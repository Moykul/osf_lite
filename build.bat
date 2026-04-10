@echo off
echo ================================================
echo  OSF Lite -- EXE Builder
echo ================================================
echo.

REM Check PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [!] PyInstaller not found. Installing...
    pip install pyinstaller
)

echo [1/3] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist\osf_lite rmdir /s /q dist\osf_lite

echo [2/3] Building exe...
pyinstaller osf_lite.spec

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Check output above.
    pause
    exit /b 1
)

echo [3/3] Build complete!
echo.
echo Output: dist\osf_lite\osf_lite.exe
echo.
echo To run:
echo   dist\osf_lite\osf_lite.exe
echo   dist\osf_lite\osf_lite.exe -c 1
echo   dist\osf_lite\osf_lite.exe --vis 0 --silent 1
echo.
echo To distribute: zip up the entire dist\osf_lite\ folder.
echo.
pause
