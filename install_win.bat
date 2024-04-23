@echo off


setlocal enabledelayedexpansion
set "file=.\config\python_path.txt"
set "python_exec="
for /f "usebackq delims=" %%i in ("%file%") do (
set "python_exec=!python_exec! %%i"
)
echo %python_exec%



set "requirements_txt=%~dp0\config\requirements.txt"
@REM set "python_exec=.\python_embeded\python.exe"

echo Installing ComfyUI's Mixlab Nodes..

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    for /f "delims=" %%i in (%requirements_txt%) do (
        %python_exec% -s -m pip install "%%i" -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
) else (
    echo Installing with system Python
    for /f "delims=" %%i in (%requirements_txt%) do (
        pip install "%%i" -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
)

pause