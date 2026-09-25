#!/bin/bash -e
#SBATCH --job-name=sw4-build-best
#SBATCH --account=nesi99999
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=genoa
#SBATCH --output=build-%j.out
#
# Build sw4 with GCC 15.2 (foss/2026) for HPC3 genoa, double precision, preferring
# 256-bit vectors: the fastest double-precision build in the gausshill-att-3 tests.
#
# Submit from this directory, which must sit in the top of the sw4 source tree:
#
#     cd build-best-default && sbatch build.sl
#
# Build on a genoa node: foss/2026's Open MPI needs libhcoll, which is installed only
# on the compute nodes, so MPI programs do not link on the login node.

builddir=${SLURM_SUBMIT_DIR:-$PWD}
srcdir=$(dirname "$builddir")
cd "$builddir"

source ./sourceme.sh > /dev/null 2>&1
echo "$(g++ --version | head -1), $(mpirun --version 2>&1 | head -1)"

start=$(date +%s)
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SW4MOPT=OFF \
      -DSW4_TARGET=hpc3-genoa \
      -DSW4_EXTRA_RELEASE_FLAGS=-mprefer-vector-width=256 \
      "$srcdir" > cmake.log 2>&1
grep "SW4 " cmake.log
make -j "${SLURM_CPUS_PER_TASK:-16}" sw4 > make.log 2>&1
test -x bin/sw4
echo "built $builddir/bin/sw4 in $(( $(date +%s) - start )) s"
