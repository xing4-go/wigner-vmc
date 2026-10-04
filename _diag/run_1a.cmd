@echo off
cd /d "E:\physics notes\condensed matter physics\quantum neural network\wigner_vmc_clean"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
python examples/phase_competition_2/crystal_seed_search.py --budget quick --rs 90 --trials 4 > _diag/run_1a.log 2> _diag/run_1a.err
echo EXIT=%ERRORLEVEL% >> _diag/run_1a.log
