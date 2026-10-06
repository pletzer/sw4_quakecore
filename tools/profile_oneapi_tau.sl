#!/bin/bash -e
#SBATCH --job-name=sw4-oneapi-tau
#SBATCH --account=nesi99999
#SBATCH --time=01:00:00         # two TAU-sampled lamb-3 runs of ~8 min each, plus margin
#SBATCH --ntasks=4              # MPI ranks
#SBATCH --cpus-per-task=2       # OpenMP threads per rank
#SBATCH --mem-per-cpu=4G
#SBATCH --output=oneapi-tau-%j.out
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#
# TAU profile (tau_exec event-based sampling + MPI + OpenMP/OMPT) of one test case with the
# oneAPI build run inside its Apptainer image, and, on the same node, of the classic Intel
# build on the host, for a like-for-like comparison. The host TAU (built against intel/2022a)
# is used in both: inside the container it is reached through the /nesi bind and preloads
# into the oneAPI sw4 with the image's Intel MPI 2021.17 (checked by job 9495796).
#
# Submit from the top of the sw4 source tree:
#
#     sbatch tools/profile_oneapi_tau.sl                  # lamb/lamb-3
#     sbatch tools/profile_oneapi_tau.sl lamb/lamb-3short
#
# Override with SW4_ONEAPI_BUILD, SW4_INTEL_BUILD (empty: oneAPI only), SW4_SIF, TAU_EBS_PERIOD.
# Profiles: <name>-tau/<jobid>/{oneapi,intel}/; compare with
#     tools/compare_tau_profiles.py <name>-tau/<jobid>/intel <name>-tau/<jobid>/oneapi --labels intel oneapi
# (pprof must be on PATH: /nesi/project/nesi99999/pletzera/tau-intel/x86_64/bin).

testcase=${1:-lamb/lamb-3}
name=$(basename "$testcase")

root=${SW4_ROOT:-$SLURM_SUBMIT_DIR}
oneapi_build=${SW4_ONEAPI_BUILD:-$root/build-oneapi}
intel_build=${SW4_INTEL_BUILD-$root/build-intel-genoa512halo-single}
sif=${SW4_SIF:-/nesi/project/nesi99999/pletzera/sifs/hdf5_oneapi.sif}

tau_root=/nesi/project/nesi99999/pletzera/tau-intel
tau_bin=$tau_root/x86_64/bin
export TAU_MAKEFILE=$tau_root/x86_64/lib/Makefile.tau-icpc-ompt-mpi-pdt-openmp
export TAU_PROFILE=1
tau_opts="-T icpc,ompt,mpi,pdt,openmp -ompt -ebs"
[ -n "$TAU_EBS_PERIOD" ] && tau_opts="$tau_opts -ebs_period=$TAU_EBS_PERIOD"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# srun places the ranks for both builds; Intel MPI's own pinning is off so that the host
# (2021.5) and container (2021.17) MPIs cannot pin differently (as tools/run_oneapi_vs_intel.sl)
export I_MPI_PIN=off
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
launch="srun --mpi=pmi2 --cpu-bind=cores"
container="apptainer exec -B /nesi -B /usr/lib64/libpmi2.so $sif"

infile="$root/pytest/reference/${testcase}.in"
outpath=$(grep -o 'path=[^ ]*' "$infile" | head -1 | cut -d= -f2)
jobid=${SLURM_JOB_ID:-local}
ntasks=${SLURM_NTASKS:-4}

for f in "$infile" "$oneapi_build/bin/sw4" "$sif" "$TAU_MAKEFILE" ${intel_build:+"$intel_build/bin/sw4"}; do
    [ -e "$f" ] || { echo "$f not found"; exit 1; }
done

echo "test case: $testcase, node: $(hostname), ranks: $ntasks, threads/rank: $OMP_NUM_THREADS"
echo "oneapi: $oneapi_build (in $sif)"
[ -n "$intel_build" ] && echo "intel:  $intel_build"

builds="oneapi"
[ -n "$intel_build" ] && builds="oneapi intel"
for build in $builds; do
(
    rundir="$root/${name}-tau/${jobid}/${build}"
    export PROFILEDIR="$rundir"
    rm -rf "$rundir"
    mkdir -p "$rundir"
    cd "$rundir"
    if [ "$build" = oneapi ]; then
        ml purge > /dev/null 2>&1    # host modules must not leak into the container
        cmd="$launch $container $tau_bin/tau_exec $tau_opts $oneapi_build/bin/sw4 $infile"
    else
        source "$intel_build/sourceme.sh" > /dev/null 2>&1
        cmd="$launch $tau_bin/tau_exec $tau_opts $intel_build/bin/sw4 $infile"
    fi
    echo
    echo "=========== $build ==========="
    echo "$cmd"
    start=$(date +%s.%N)
    rc=0
    $cmd > sw4.out 2>&1 || rc=$?
    end=$(date +%s.%N)
    grep -E "Execution time, (solver|time stepping) phase" sw4.out | tail -1 || true
    awk -v a="$start" -v b="$end" 'BEGIN { printf "wall-clock time: %.2f s\n", b - a }'
    if [ $rc -ne 0 ]; then
        echo "sw4 failed (exit $rc); tail of $rundir/sw4.out:"
        tail -30 sw4.out
        exit $rc
    fi
    [ -f "${outpath:-.}/LambErr.txt" ] && echo "last LambErr line: $(tail -1 ${outpath:-.}/LambErr.txt)"
    [ -f "${outpath:-.}/TwilightErr.txt" ] && cat "${outpath:-.}/TwilightErr.txt"

    nprof=$(ls profile.* 2>/dev/null | wc -l)
    echo "$nprof TAU profile files in $rundir"
    [ "$nprof" -gt 0 ] || { echo "no profiles written; check sw4.out"; exit 1; }
    # pprof is a host binary built with intel/2022a
    ml purge > /dev/null 2>&1
    ml intel/2022a > /dev/null 2>&1
    $tau_bin/pprof -s -m | head -60 > pprof-summary.txt
    $tau_bin/pprof -m > pprof-full.txt
    echo "pprof summary in $rundir/pprof-summary.txt"
) || echo "${build}: profiling failed"
done
