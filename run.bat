@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set "VENV_DIR=venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [AirStudio] venv not found. Run install.bat first.
  popd
  pause
  exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
  echo [AirStudio] Failed to activate venv.
  popd
  pause
  exit /b 1
)

python -m airstudio.main %*

popd
pause
