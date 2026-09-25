#!/bin/bash -e
#SBATCH --job-name=sw4-gausshill-att-3
#SBATCH --account=nesi99999
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4              # MPI ranks
#SBATCH --cpus-per-task=2       # OpenMP threads per rank, one physical core each
#SBATCH --hint=nomultithread    # no SMT: a CPU is a physical core
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=genoa
#SBATCH --output=run-gausshill-att-3-%j.out
#
# Run pytest/reference/curvimeshrefine/gausshill-att-3 with the sw4 built here by
# build.sl (GCC 15.2, Open MPI 5), and compare its errors with the reference.
#
#     cd build-best-default && sbatch run-gausshill-att-3.sl
#
# srun places the ranks: each gets cpus-per-task physical cores (--cpu-bind=cores), and
# OpenMP puts one thread on each of them. Open MPI 5 needs PMIx (it dropped PMI2).

builddir=${SLURM_SUBMIT_DIR:-$PWD}
srcdir=$(dirname "$builddir")
sw4="$builddir/bin/sw4"
infile="$srcdir/pytest/reference/curvimeshrefine/gausshill-att-3.in"
reffile="$srcdir/pytest/reference/curvimeshrefine/gausshill-att-3/TwilightErr.txt"
# sw4 writes its output to the fileio path of the input file, relative to the cwd
rundir="$builddir/run-gausshill-att-3-${SLURM_JOB_ID:-local}"

source "$builddir/sourceme.sh" > /dev/null 2>&1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p "$rundir"
cd "$rundir"
echo "node: $(hostname), ranks: ${SLURM_NTASKS:-4}, threads/rank: $OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
echo "srun --mpi=pmix --cpu-bind=cores $sw4 $infile"

start=$(date +%s.%N)
srun --mpi=pmix --cpu-bind=cores "$sw4" "$infile" > sw4.out 2>&1
end=$(date +%s.%N)

grep -E "Execution time, (solver|time stepping) phase" sw4.out | tail -1
awk -v a="$start" -v b="$end" 'BEGIN { printf "wall-clock time: %.2f s\n", b - a }'
echo "errors (errInf, errL2, solInf), this run:"
cat gausshill-att-3/TwilightErr.txt
echo "reference:"
cat "$reffile"
