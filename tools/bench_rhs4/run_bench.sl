#!/bin/bash -e
#SBATCH --job-name=bench-rhs4
#SBATCH --account=nesi99999
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4              # independent copies, to share the node as the 4 sw4 ranks do
#SBATCH --cpus-per-task=2       # OpenMP threads per copy
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=genoa
#SBATCH --output=bench-rhs4-%j.out
#
# Old (git revision BASE, default HEAD) vs new (working tree) src/rhs4th3fortc.C, compiled with
# icpc as build-intel-*-single (generic, and AVX-512 as hpc3-genoa): bench_rhs4 check on random
# data (compare.py), then ms per rhs4th3fortsgstr_ci call, one copy and 4 copies at once,
# in the order old, new for each SW4_RHS_JTILE in JTILES, old -- or, with ROUNDS=r, r rounds of
# old, new for each JTILES (interleaved, so drift during the job affects all variants alike).
#
#     sbatch tools/bench_rhs4/run_bench.sl [n ...]       # from the top of the source tree
#     JTILES="8 16" ROUNDS=4 sbatch --exclusive tools/bench_rhs4/run_bench.sl 305
#
# On a reserved socket (4 x 21 = the 84 cores of one EPYC 9634), with the copies packed onto
# adjacent cores within L3 domains, as the ranks of a real run share the caches:
#
#     JTILES="8 16" ROUNDS=4 BENCH_THREADS=2 BENCH_PIN=compact sbatch --cpus-per-task=21 \
#         --ntasks-per-socket=4 --mem=48G tools/bench_rhs4/run_bench.sl 305

src=${SLURM_SUBMIT_DIR:-$PWD}
here=$src/tools/bench_rhs4
base=${BASE:-HEAD}
sizes="$*"
jtiles=${JTILES:-4 8 16 32 64}
if [ -n "$ROUNDS" ]; then
    order=$(for r in $(seq "$ROUNDS"); do echo old $jtiles; done)
else
    order="old $jtiles old"
fi
work=$here/run-${SLURM_JOB_ID:-local}
mkdir -p "$work"
cd "$work"
git -C "$src" show "$base:src/rhs4th3fortc.C" > rhs_old.C
cp "$src/src/rhs4th3fortc.C" rhs_new.C
echo "old: $base:src/rhs4th3fortc.C ($(git -C "$src" rev-parse --short "$base")), new: working tree"

ml purge > /dev/null 2>&1
ml intel/2022a > /dev/null 2>&1
inc="-I$src/src -I$src/src/float"
generic="-std=gnu++17 -O3 -DNDEBUG -qoverride-limits -qopenmp"
avx512="-std=gnu++17 -O3 -DNDEBUG -march=icelake-server -qoverride-limits -qopt-zmm-usage=high -qopenmp"
for flav in generic avx512; do
    for v in old new; do
        eval "flags=\$$flav"
        ( icpc $flags $inc "$here/bench_rhs4.C" rhs_$v.C -o b-$flav-$v > cc-$flav-$v.log 2>&1 \
          || echo "compile $flav $v failed, see $work/cc-$flav-$v.log" ) &
    done
done
wait
ls b-* > /dev/null

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=${BENCH_THREADS:-${SLURM_CPUS_PER_TASK:-2}}
ncopies=${SLURM_NTASKS:-4}
echo "node: $(hostname), $OMP_NUM_THREADS threads per copy"
if [ "${BENCH_PIN:-slurm}" = compact ]; then
    # Physical cores of the allocation, grouped by L3 domain; copies are packed in order,
    # OMP_NUM_THREADS cores each, never straddling two L3 domains (as tools/run_oneapi_vs_intel.sl)
    masks=$(/usr/bin/python3 - "$ncopies" "$OMP_NUM_THREADS" <<'EOP'
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
    bind1="--cpu-bind=mask_cpu:${masks%%,*}"
    bindn="--cpu-bind=mask_cpu:$masks"
    echo "compact pinning, masks: $masks"
else
    bind1="--cpu-bind=cores"
    bindn="--cpu-bind=cores"
fi
for flav in generic avx512; do
    echo
    ./b-$flav-old check > chk-old.bin
    for jt in 1 2 3 16; do
        echo "=========== $flav: check old vs new, SW4_RHS_JTILE=$jt ==========="
        SW4_RHS_JTILE=$jt ./b-$flav-new check > chk-new.bin
        /usr/bin/python3 "$here/compare.py" chk-old.bin chk-new.bin | tail -1
    done
    for run in $order; do
        v=new; jt=$run
        [ $run = old ] && { v=old; jt=-; }
        echo
        echo "=========== $flav $v (SW4_RHS_JTILE=$jt), 1 copy ==========="
        SW4_RHS_JTILE=$jt srun -n 1 $bind1 ./b-$flav-$v time $sizes
        echo "=========== $flav $v (SW4_RHS_JTILE=$jt), $ncopies copies (rank 0 shown) ==========="
        SW4_RHS_JTILE=$jt srun -n $ncopies $bindn bash -c "./b-$flav-$v time $sizes > out.\$SLURM_PROCID"
        cat out.0
    done
done
