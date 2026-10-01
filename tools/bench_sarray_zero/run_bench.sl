#!/bin/bash -e
#SBATCH --job-name=bench-zero
#SBATCH --account=nesi99999
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=4              # independent copies, to share memory bandwidth as the 4 sw4 ranks do
#SBATCH --cpus-per-task=4       # runs use 2 (lamb-3) or 4 (production) OpenMP threads per copy
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=genoa
#SBATCH --output=bench-zero-%j.out
#
# Build bench_set_to_zero.C with classic icpc (no arch flags, and AVX-512 as
# build-intel-genoa512auto-single) and with GCC 15 (-march=znver4, as the gcc15-genoa builds),
# then run each on the same node: one copy and 4 copies at once, with 2 and 4 OpenMP threads.
#
#     sbatch tools/bench_sarray_zero/run_bench.sl [n ...]

here=${SLURM_SUBMIT_DIR:-$PWD}/tools/bench_sarray_zero
sizes="$*"
cd "$here"

(
    ml purge > /dev/null 2>&1
    ml intel/2022a > /dev/null 2>&1
    icpc -O3 -DNDEBUG -qopenmp -o bench-icpc-generic bench_set_to_zero.C
    icpc -O3 -DNDEBUG -march=icelake-server -qoverride-limits -qopt-zmm-usage=high -qopenmp \
         -o bench-icpc-avx512 bench_set_to_zero.C
)
(
    ml purge > /dev/null 2>&1
    ml foss/2026 > /dev/null 2>&1
    g++ -O3 -DNDEBUG -march=znver4 -mtune=znver4 -mprefer-vector-width=512 -fopenmp \
        -o bench-gcc15-znver4 bench_set_to_zero.C
) || echo "GCC 15 build failed; skipping it"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "node: $(hostname)"
for flavour in icpc-generic icpc-avx512 gcc15-znver4; do
    [ -x bench-$flavour ] || continue
    # the runtime libraries (libiomp5 / libgomp, libstdc++) of the compiler that built it
    ml purge > /dev/null 2>&1
    case $flavour in
        icpc-*) ml intel/2022a > /dev/null 2>&1 ;;
        gcc15-*) ml foss/2026 > /dev/null 2>&1 ;;
    esac
    for nthreads in 2 4; do
        export OMP_NUM_THREADS=$nthreads
        echo
        echo "=========== $flavour, $nthreads threads, 1 copy ==========="
        srun -n 1 -c $nthreads --cpu-bind=cores ./bench-$flavour $sizes
        echo
        echo "=========== $flavour, $nthreads threads, 4 copies (rank 0 shown) ==========="
        srun -n 4 -c $nthreads --cpu-bind=cores bash -c "./bench-$flavour $sizes > out-$flavour.\$SLURM_PROCID"
        cat out-$flavour.0
        grep -h NONZERO out-$flavour.* || true
        rm -f out-$flavour.*
    done
done
