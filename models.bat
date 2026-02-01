@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set "LOG=%SCRIPT_DIR%models.log"
>"%LOG%" echo [AirStudio] models.bat start

echo [AirStudio] Working dir: %SCRIPT_DIR%>>"%LOG%"

echo [AirStudio] Log: %LOG%>>"%LOG%"

set "VENV_DIR=venv"
set "MODELS_DIR=models"
set "PY=%SCRIPT_DIR%venv\Scripts\python.exe"

echo [AirStudio] Checking venv at %PY%>>"%LOG%"
if not exist "%PY%" (
  echo [AirStudio] venv not found. Run install.bat first.
  echo [AirStudio] venv not found.>>"%LOG%"
  popd
  pause
  exit /b 1
)

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

echo [AirStudio] Using: %PY%>>"%LOG%"

echo [AirStudio] Checking huggingface_hub...>>"%LOG%"
"%PY%" -m pip show huggingface_hub >nul 2>&1
set "ERR=%ERRORLEVEL%"
echo [AirStudio] pip show exit: %ERR%>>"%LOG%"
if %ERR% NEQ 0 (
  echo [AirStudio] Installing huggingface_hub...
  echo [AirStudio] Installing huggingface_hub...>>"%LOG%"
  "%PY%" -m pip install "huggingface_hub" >>"%LOG%" 2>&1
  set "ERR=%ERRORLEVEL%"
  echo [AirStudio] pip install exit: %ERR%>>"%LOG%"
  if %ERR% NEQ 0 (
    echo [AirStudio] Failed to install huggingface_hub. See models.log
    popd
    pause
    exit /b 1
  )
)

if "%HF_TOKEN%"=="" (
  echo [AirStudio] HF_TOKEN not set. Continuing without token.>>"%LOG%"
) else (
  echo [AirStudio] HF_TOKEN provided.>>"%LOG%"
)

set "M0_REPO=mistralai/Mistral-7B-Instruct-v0.3"
set "M0_DIR=%MODELS_DIR%\Mistral-7B-Instruct-v0.3"
set "M1_REPO=meta-llama/Llama-3.1-8B-Instruct"
set "M1_DIR=%MODELS_DIR%\Llama-3.1-8B-Instruct"

set "MODELS=%~1"
if "%MODELS%"=="" set "MODELS=mistral7b,llama8b"

echo [AirStudio] Download list: %MODELS%>>"%LOG%"

call :download_model mistral7b "%M0_REPO%" "%M0_DIR%" "%MODELS%"
call :download_model llama8b "%M1_REPO%" "%M1_DIR%" "%MODELS%"

echo [AirStudio] Done.>>"%LOG%"
popd
pause
exit /b 0

:download_model
set "KEY=%~1"
set "REPO=%~2"
set "OUTDIR=%~3"
set "LIST=%~4"

echo %LIST% | findstr /I /C:%KEY% >nul
if %ERRORLEVEL% NEQ 0 goto :eof

echo [AirStudio] Downloading %REPO%...
echo [AirStudio] Downloading %REPO%...>>"%LOG%"
if "%HF_TOKEN%"=="" (
  "%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=r'%REPO%', local_dir=r'%OUTDIR%', token=None)"
) else (
  "%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=r'%REPO%', local_dir=r'%OUTDIR%', token=r'%HF_TOKEN%')"
)
if %ERRORLEVEL% NEQ 0 (
  echo [AirStudio] Failed to download %REPO%. See models.log
  pause
)

goto :eof
