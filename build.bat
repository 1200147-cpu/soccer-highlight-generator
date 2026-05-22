@echo off
echo =========================================
echo  Soccer Highlight GUI - EXE Build
echo =========================================
echo.

echo [1/3] Installing libraries...
pip install PyQt6 opencv-python librosa ultralytics pyinstaller

echo.
echo [2/3] Building EXE...
pyinstaller SoccerHighlight.spec --noconfirm --clean

echo.
echo [3/3] Copying files...

IF EXIST yolo11n.pt (
    copy yolo11n.pt dist\SoccerHighlight\yolo11n.pt
    echo   yolo11n.pt copied.
) ELSE (
    echo   yolo11n.pt not found. Please copy manually.
)

IF EXIST ffmpeg.exe (
    copy ffmpeg.exe dist\SoccerHighlight\ffmpeg.exe
    echo   ffmpeg.exe copied.
) ELSE (
    echo   ffmpeg.exe not found. Please copy manually.
)

echo.
echo =========================================
echo  Build complete!
echo  dist\SoccerHighlight\SoccerHighlight.exe
echo =========================================
pause
