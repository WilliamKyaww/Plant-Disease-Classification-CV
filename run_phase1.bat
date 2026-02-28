@echo off
setlocal

cd /d "%~dp0"

echo [1/5] Running integrity checks and writing JSON/TXT artifacts with Python 3.13...
py -3.13 -m src.integrity_report
if errorlevel 1 goto :fail

echo [2/5] Generating frozen splits with Python 3.13...
py -3.13 -m src.prepare_splits
if errorlevel 1 goto :fail

echo [3/5] Running minimal tests with Python 3.13...
py -3.13 -m unittest discover -s tests -v
if errorlevel 1 goto :fail

echo [4/5] Running script-based baseline smoke training (15-class)...
py -3.13 -m src.run_baseline_smoke --epochs 1 --batch-size 32 --max-train 128 --max-val 64 --max-test 64
if errorlevel 1 goto :fail

echo [5/5] Running Colab path sanity smoke report...
py -3.13 -m src.colab_smoke --repo-main .
if errorlevel 1 goto :fail

echo.
echo Phase 1 runner completed successfully.
exit /b 0

:fail
echo.
echo Phase 1 runner failed.
exit /b 1
