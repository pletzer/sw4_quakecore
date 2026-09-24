#!/bin/bash -e
#SBATCH --job-name=sw4-testcase
#SBATCH --account=nesi99999
#SBATCH --time=02:00:00         # six runs of one test case
#SBATCH --ntasks=4              # MPI ranks per run
#SBATCH --cpus-per-task=2       # OpenMP threads per rank
#SBATCH --mem-per-cpu=2G
#SBATCH --output=testcase-%j.out
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#
# Run one pytest/reference test case with every (platform, precision) build on the
# same node and report the final twilight errors and the execution times.
#
# Submit from the top of the sw4 source tree, where the build-<platform>-<precision>
# directories live (or set SW4_ROOT), giving the case as <dir>/<name> relative to
# pytest/reference:
#
#     sbatch tools/run_testcase.sl meshrefine/refine-el-2
#     sbatch tools/run_testcase.sl curvimeshrefine/gausshill-att-3
#
# Each run gets its own directory <name>-runs/<platform>-<precision>/. The summary is
# written to <name>-<jobid>.md and <name>-<jobid>.csv, and a bar plot of the execution
# times to <name>-<jobid>.png (see tools/plot_testcase_timings.py).

testcase=${1:-meshrefine/refine-el-2}
name=$(basename "$testcase")

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Intel MPI (gimkl, intel): give each rank OMP_NUM_THREADS cores
export I_MPI_PIN_DOMAIN=omp

root=${SW4_ROOT:-$SLURM_SUBMIT_DIR}
infile="$root/pytest/reference/${testcase}.in"
reffile="$root/pytest/reference/${testcase}/TwilightErr.txt"
rundir="$root/${name}-runs"
jobid=${SLURM_JOB_ID:-local}
tablefile="$root/${name}-${jobid}.md"
csvfile="$root/${name}-${jobid}.csv"
plotfile="$root/${name}-${jobid}.png"
ntasks=${SLURM_NTASKS:-4}
# python with matplotlib, for the bar plot
python_module=Python/3.11.3-gimkl-2022a

platforms="gimkl intel aocc"
precs="default single"

if [ ! -f "$infile" ]; then
    echo "input file $infile not found; submit from the sw4 source tree or set SW4_ROOT"
    exit 1
fi
# sw4 writes TwilightErr.txt into the fileio path given in the input file
outpath=$(grep -o 'path=[^ ]*' "$infile" | head -1 | cut -d= -f2)
outpath=${outpath:-.}

# TwilightErr.txt has a header line "<Displacement|Attenuation> variables (errInf, errL2,
# solInf)" followed by the values -> "dispInf dispL2 attInf attL2" (- if absent)
read_errors() {
    [ -f "$1" ] || { echo "- - - -"; return; }
    awk '
        /^Displacement variables/ { getline; d = $1 " " $2 }
        /^Attenuation variables/  { getline; a = $1 " " $2 }
        END { print (d == "" ? "- -" : d), (a == "" ? "- -" : a) }
    ' "$1"
}

# "Execution time, solver|time stepping phase [h hours] [m minutes] s seconds" -> seconds
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

rows=()
echo "platform,precision,disp_errinf,disp_errl2,att_errinf,att_errl2,step_seconds,wall_seconds,status" > "$csvfile"
for platform in $platforms; do
    for prec in $precs; do
        combo="${platform}-${prec}"
        builddir="$root/build-${combo}"
        sw4="$builddir/bin/sw4"
        if [ ! -x "$sw4" ]; then
            echo "skipping ${combo}: $sw4 not found"
            rows+=("| $platform | $prec | - | - | - | - | - | - | missing |")
            echo "$platform,$prec,,,,,,,missing" >> "$csvfile"
            continue
        fi
        workdir="$rundir/$combo"
        rm -rf "$workdir"
        mkdir -p "$workdir"
        logfile="$workdir/sw4.out"

        echo "=========== running ${combo} ==========="
        status=ok
        # subshell: the module environment doesn't leak into the next build
        (
            cd "$workdir"
            source "$builddir/sourceme.sh" > /dev/null 2>&1
            if mpirun --version 2>&1 | grep -q "Open MPI"; then
                launch="mpirun -np $ntasks --map-by slot:PE=$OMP_NUM_THREADS --bind-to core"
            else
                launch="mpirun -np $ntasks"
            fi
            echo "$launch $sw4 $infile"
            start=$(date +%s.%N)
            rc=0
            $launch "$sw4" "$infile" > "$logfile" 2>&1 || rc=$?
            end=$(date +%s.%N)
            awk -v a="$start" -v b="$end" 'BEGIN { printf "%.2f\n", b - a }' > wall.txt
            exit $rc
        ) || status=failed

        wall=$(cat "$workdir/wall.txt" 2>/dev/null || echo -)
        solver=$(solver_seconds "$logfile")
        read -r dinf dl2 ainf al2 <<< "$(read_errors "$workdir/$outpath/TwilightErr.txt")"
        rows+=("| $platform | $prec | $dinf | $dl2 | $ainf | $al2 | ${solver:--} | $wall | $status |")
        echo "$platform,$prec,$dinf,$dl2,$ainf,$al2,$solver,$wall,$status" | sed 's/,-/,/g' >> "$csvfile"
        echo "${combo}: disp errInf=$dinf errL2=$dl2, att errInf=$ainf errL2=$al2," \
             "time stepping=${solver:--}s wall=${wall}s ($status)"
    done
done

read -r rdinf rdl2 rainf ral2 <<< "$(read_errors "$reffile")"
{
    echo "$testcase on $(hostname), job $jobid," \
         "$ntasks MPI ranks x $OMP_NUM_THREADS OpenMP threads"
    echo
    echo "| platform | precision | disp errInf | disp errL2 | att errInf | att errL2 | time stepping (s) | wall (s) | status |"
    echo "|----------|-----------|-------------|------------|------------|-----------|-------------------|----------|--------|"
    echo "| reference | double | $rdinf | $rdl2 | $rainf | $ral2 | - | - | - |"
    printf '%s\n' "${rows[@]}"
} > "$tablefile"

echo
cat "$tablefile"
echo
echo "table written to $tablefile and $csvfile"

(
    ml purge > /dev/null 2>&1
    ml "$python_module" > /dev/null 2>&1
    python3 "$root/tools/plot_testcase_timings.py" "$csvfile" -o "$plotfile" \
        --title "$testcase ($ntasks ranks x $OMP_NUM_THREADS threads, $(hostname -s))"
) || echo "bar plot failed; rerun tools/plot_testcase_timings.py $csvfile by hand"
