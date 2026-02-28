@echo off
setlocal

cd /d "%~dp0"

echo [1/3] Running integrity checks and writing JSON/TXT artifacts with Python 3.13...
py -3.13 -m src.integrity_report
if errorlevel 1 goto :fail

echo [2/3] Generating frozen splits with Python 3.13...
py -3.13 -m src.prepare_splits
if errorlevel 1 goto :fail

echo [3/3] Running minimal tests with Python 3.13...
py -3.13 -m unittest discover -s tests -v
if errorlevel 1 goto :fail

echo.
echo Phase 1 runner completed successfully.
exit /b 0

:fail
echo.
echo Phase 1 runner failed.
exit /b 1
