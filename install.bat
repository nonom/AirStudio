@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
set "PROJECT_DIR=%CD%"

set "VENV_DIR=venv"

if not exist "models" mkdir "models"
if not exist "cache" mkdir "cache"
if not exist "runs" mkdir "runs"

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  set "PYTHON=py -3"
) else (
  set "PYTHON=python"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [AirStudio] Creating venv in "%VENV_DIR%"...
  %PYTHON% -m venv "%VENV_DIR%"
  if %ERRORLEVEL% NEQ 0 goto :fail
)

call "%VENV_DIR%\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 goto :fail

python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 goto :fail

if exist "pyproject.toml" goto :install_pyproject
if exist "requirements.txt" goto :install_requirements
goto :after_deps

:install_pyproject
python -m pip install -e "%PROJECT_DIR%" --upgrade --force-reinstall
if %ERRORLEVEL% NEQ 0 goto :fail
goto :after_deps

:install_requirements
python -m pip install -r requirements.txt --upgrade --force-reinstall
if %ERRORLEVEL% NEQ 0 goto :fail

:after_deps
if /I "%AIRSTUDIO_TORCH%"=="cpu" goto :after_torch
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
echo [AirStudio] Installing CUDA-enabled PyTorch from %TORCH_INDEX_URL%...
python -m pip install --index-url %TORCH_INDEX_URL% torch torchvision torchaudio --upgrade --force-reinstall
if %ERRORLEVEL% NEQ 0 goto :fail_torch

:after_torch
echo [AirStudio] Installing protobuf...
python -m pip install protobuf --upgrade
if %ERRORLEVEL% NEQ 0 goto :fail

python -m pip uninstall -y pynvml >nul 2>&1
python -m pip install "nvidia-ml-py" --upgrade

if /I "%AIRSTUDIO_BNB%"=="skip" goto :skip_bnb
echo [AirStudio] Installing bitsandbytes (optional)...
python -m pip install bitsandbytes --upgrade
if %ERRORLEVEL% NEQ 0 (
  echo [AirStudio] bitsandbytes install failed. low_vram/balanced will be unavailable.
) else (
  echo [AirStudio] bitsandbytes installed.
)

:skip_bnb
python -c "import bitsandbytes as bnb; print('bitsandbytes:', bnb.__version__)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [AirStudio] bitsandbytes import check failed. low_vram/balanced may not work.
) else (
  echo [AirStudio] bitsandbytes import check OK.
)

echo [AirStudio] Install complete.
popd
pause
exit /b 0

:fail_torch
echo [AirStudio] Failed to install CUDA PyTorch. Check your drivers and internet connection.

:fail
echo [AirStudio] Install failed.
popd
pause
exit /b 1
