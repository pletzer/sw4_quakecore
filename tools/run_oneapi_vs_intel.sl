#!/bin/bash -e
#SBATCH --job-name=sw4-oneapi-vs-intel
#SBATCH --account=nesi99999
#SBATCH --time=01:30:00         # four lamb-3 runs of ~7 min each, plus margin
#SBATCH --ntasks=4              # MPI ranks per run
#SBATCH --cpus-per-task=2       # OpenMP threads per rank
#SBATCH --mem-per-cpu=2G
#SBATCH --output=oneapi-vs-intel-%j.out
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#
# Head-to-head timing of the classic Intel build (icpc, intel/2022a, host modules)
# against the oneAPI build (icpx, run inside an Apptainer image) on one test case,
# on the same node, in A B B A order so that drift during the job (node warm-up,
# a noisy neighbour) affects both builds equally.
#
# Submit from the top of the sw4 source tree:
#
#     sbatch tools/run_oneapi_vs_intel.sl                 # lamb/lamb-3
#     sbatch tools/run_oneapi_vs_intel.sl lamb/lamb-3short
#
# Override the builds or the image with SW4_INTEL_BUILD, SW4_ONEAPI_BUILD, SW4_SIF.
# Results: <name>-oneapi-vs-intel-<jobid>.{md,csv}, runs in <name>-runs/<jobid>/.
#
# Pinning. By default Slurm's --cpu-bind=cores places the ranks on whatever cores the
# allocation got, which on a shared node can be scattered across both sockets. With
# SW4_PIN=compact the job reserves a whole socket and packs the ranks onto contiguous
# cores within L3 domains (CCDs), so the cache and the socket's memory bandwidth are
# not shared with other jobs; the rest of the socket stays idle:
#
#     sbatch --cpus-per-task=21 --ntasks-per-socket=4 --mem-per-cpu=256M --export=ALL,SW4_PIN=compact,SW4_THREADS=2 \
#         tools/run_oneapi_vs_intel.sl
#
# (4 x 21 = the 84 cores of one EPYC 9634 socket; SW4_THREADS is the OpenMP threads per rank.)
# In both modes Intel MPI's own pinning is off, so both MPIs (host 2021.5, container 2021.17)
# get exactly the placement srun gives them, and I_MPI_DEBUG=4 logs it for every run.

testcase=${1:-lamb/lamb-3}
name=$(basename "$testcase")

root=${SW4_ROOT:-$SLURM_SUBMIT_DIR}
intel_build=${SW4_INTEL_BUILD:-$root/build-intel-genoa512halo-single}
oneapi_build=${SW4_ONEAPI_BUILD:-$root/build-oneapi}
sif=${SW4_SIF:-/nesi/project/nesi99999/pletzera/sifs/hdf5_oneapi.sif}

pin=${SW4_PIN:-slurm}
export OMP_NUM_THREADS=${SW4_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# srun does the binding; the two Intel MPI versions could otherwise re-pin a scattered
# allocation differently and penalise one build only
export I_MPI_PIN=off
export I_MPI_DEBUG=4
# Both Intel MPIs (host 2021.5, container 2021.17) get their ranks from Slurm via
# the host's PMI2 library; it only depends on libc, so it can be bound into the
# container as is. Apptainer passes the host environment through, so this and
# the OMP/I_MPI settings above reach the container too.
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
ntasks=${SLURM_NTASKS:-4}
if [ "$pin" = compact ]; then
    # Physical cores of this allocation (first hardware thread of each), grouped by
    # L3 domain as the kernel reports it; ranks are packed in order, OMP_NUM_THREADS
    # cores each, never straddling two L3 domains. One hex mask per rank for srun.
    masks=$(/usr/bin/python3 - "$ntasks" "$OMP_NUM_THREADS" <<'EOP'
import os, sys
ntasks, nthr = int(sys.argv[1]), int(sys.argv[2])
def parse(s):
    out = []
    for part in s.strip().split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out
def read(p):
    with open(p) as f:
        return f.read()
allowed = set(os.sched_getaffinity(0))
cores = sorted(c for c in allowed
               if min(parse(read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"))) == c)
groups = {}
for c in cores:
    l3 = min(parse(read(f"/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list")))
    groups.setdefault(l3, []).append(c)
chunks = []
for l3 in sorted(groups):
    g = groups[l3]
    chunks += [g[i:i + nthr] for i in range(0, len(g) - nthr + 1, nthr)]
if len(chunks) < ntasks:
    sys.exit(f"only {len(chunks)} groups of {nthr} cores within one L3 domain in {sorted(allowed)}")
print(",".join(hex(sum(1 << c for c in ch)) for ch in chunks[:ntasks]))
EOP
) || exit 1
    launch="srun --mpi=pmi2 --cpu-bind=mask_cpu:$masks"
else
    launch="srun --mpi=pmi2 --cpu-bind=cores"
fi
container="apptainer exec -B /nesi -B /usr/lib64/libpmi2.so $sif"

infile="$root/pytest/reference/${testcase}.in"
errname=TwilightErr.txt
grep -q "^testlamb" "$infile" && errname=LambErr.txt
reffile="$root/pytest/reference/${testcase}/${errname}"
outpath=$(grep -o 'path=[^ ]*' "$infile" | head -1 | cut -d= -f2)
outpath=${outpath:-.}
jobid=${SLURM_JOB_ID:-local}
rundir="$root/${name}-runs/${jobid}"
tablefile="$root/${name}-oneapi-vs-intel-${jobid}.md"
csvfile="$root/${name}-oneapi-vs-intel-${jobid}.csv"

for f in "$infile" "$intel_build/bin/sw4" "$oneapi_build/bin/sw4" "$sif"; do
    [ -e "$f" ] || { echo "$f not found"; exit 1; }
done

# same parsers as tools/run_testcase.sl
read_errors() {
    [ -f "$1" ] || { echo "- - - -"; return; }
    awk '
        /^[-+0-9.eE]+[ \t]/      { d = $2 " " $3 }
        /^Displacement variables/ { getline; d = $1 " " $2 }
        /^Attenuation variables/  { getline; a = $1 " " $2 }
        END { print (d == "" ? "- -" : d), (a == "" ? "- -" : a) }
    ' "$1"
}
solver_seconds() {
    grep -hE "Execution time, (solver|time stepping) phase" "$1" 2>/dev/null | tail -1 | awk '{
        t = 0
        for (i = 1; i < NF; i++) {
            if ($(i+1) ~ /^hour/)   t += 3600 * $i
            if ($(i+1) ~ /^minute/) t += 60 * $i
            if ($(i+1) ~ /^second/) t += $i
        }
        printf "%.2f", t
    }'
}

mkdir -p "$rundir"
echo "test case: $testcase, node: $(hostname), ranks: $ntasks, threads/rank: $OMP_NUM_THREADS"
echo "A = intel:  $intel_build"
echo "B = oneapi: $oneapi_build (in $sif)"
echo "pinning: $pin, $launch"
echo "allocated CPUs: $(taskset -cp $$ | cut -d: -f2)"

# Smoke test of MPI inside the container (the image's parallel HDF5 test), so a
# broken launch fails here in seconds rather than after the first full A run.
(
    cd "$rundir"
    ml purge > /dev/null 2>&1
    $launch $container ph5test "$rundir/ph5test.h5"
) || { echo "MPI launch inside the container failed"; exit 1; }
rm -f "$rundir/ph5test.h5"

rows=()
echo "order,build,disp_errinf,disp_errl2,step_seconds,wall_seconds,status" > "$csvfile"
order=1
for build in intel oneapi oneapi intel; do
    workdir="$rundir/${order}-${build}"
    rm -rf "$workdir"
    mkdir -p "$workdir"
    logfile="$workdir/sw4.out"
    echo "=========== run $order: $build ==========="
    status=ok
    (
        cd "$workdir"
        if [ "$build" = intel ]; then
            source "$intel_build/sourceme.sh" > /dev/null 2>&1
            cmd="$launch $intel_build/bin/sw4 $infile"
        else
            ml purge > /dev/null 2>&1    # host modules must not leak into the container
            cmd="$launch $container $oneapi_build/bin/sw4 $infile"
        fi
        echo "$cmd"
        start=$(date +%s.%N)
        rc=0
        $cmd > "$logfile" 2>&1 || rc=$?
        end=$(date +%s.%N)
        awk -v a="$start" -v b="$end" 'BEGIN { printf "%.2f\n", b - a }' > wall.txt
        exit $rc
    ) || status=failed
    wall=$(cat "$workdir/wall.txt" 2>/dev/null || echo -)
    solver=$(solver_seconds "$logfile")
    read -r dinf dl2 _ _ <<< "$(read_errors "$workdir/$outpath/$errname")"
    rows+=("| $order | $build | $dinf | $dl2 | ${solver:--} | $wall | $status |")
    echo "$order,$build,$dinf,$dl2,$solver,$wall,$status" | sed 's/,-/,/g' >> "$csvfile"
    echo "$build: errInf=$dinf errL2=$dl2, time stepping=${solver:--}s wall=${wall}s ($status)"
    # where Intel MPI says each rank ran ("Rank Pid Node Pin cpu" table)
    grep -E "MPI startup\(\): +[0-9]+ +[0-9]+ +[^ ]+ +\{" "$logfile" || true
    order=$((order + 1))
done

read -r rdinf rdl2 _ _ <<< "$(read_errors "$reffile")"
# mean time stepping per build, and B relative to A
summary=$(awk -F, 'NR > 1 && $7 == "ok" { s[$2] += $5; n[$2]++ }
    END {
        for (b in s) printf "%s mean time stepping: %.2f s (%d runs)\n", b, s[b] / n[b], n[b]
        if (n["intel"] && n["oneapi"])
            printf "oneapi / intel: %.3f\n", (s["oneapi"] / n["oneapi"]) / (s["intel"] / n["intel"])
    }' "$csvfile" | sort)
{
    echo "$testcase on $(hostname), job $jobid, $ntasks MPI ranks x $OMP_NUM_THREADS OpenMP threads"
    echo
    echo "| order | build | disp errInf | disp errL2 | time stepping (s) | wall (s) | status |"
    echo "|-------|-------|-------------|------------|-------------------|----------|--------|"
    echo "| ref | double | $rdinf | $rdl2 | - | - | - |"
    printf '%s\n' "${rows[@]}"
    echo
    echo "$summary" | sed 's/^/    /'
} > "$tablefile"

echo
cat "$tablefile"
echo
echo "table written to $tablefile and $csvfile"
