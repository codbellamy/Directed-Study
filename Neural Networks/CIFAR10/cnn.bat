@echo off
title Big Brain
if not exist "%USERPROFILE%\Documents\Big Brain" (
    mkdir "%USERPROFILE%\Documents\Big Brain"
    python --version 2>nul
    if errorlevel 1 goto errorNoPython
    pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
    cls
)

echo This script will guess what object is in an image.
echo It can only check for one object per image.
echo.
echo Please use images from these catagories:
echo Plane, car, bird, cat, deer, dog, frog, horse, ship, and truck.
echo.

:start
set filename=""
set /p filename="Filename: "
cls
if %filename% == "" exit
python ./ShipAnalyzer.py %filename%
echo.
echo Press ENTER or close the window to exit.
goto start

:errorNoPython
color 04
echo You must have python installed for this script to work.
pause
exit