@echo off

setlocal enabledelayedexpansion
set NO_ALBUMENTATIONS_UPDATE=1
set "file=.\config\python_path.txt"
set "python_exec="
for /f "usebackq delims=" %%i in ("%file%") do (
set "python_exec=!python_exec! %%i"
)
echo %python_exec%

%python_exec% -s ComfyUI\main_mixlab.py --extra-model-paths-config ./config/extra_model_paths.yaml --port 8188 --listen 0.0.0.0 --multi-user --enable-cors-header
pause