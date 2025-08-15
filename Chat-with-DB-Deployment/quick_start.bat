@echo off
REM Chat with DB - Quick Start Script for Windows
REM This script sets up the development environment quickly

echo 🚀 Starting Chat with DB Quick Setup...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo ✅ Python found

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)
echo ✅ Node.js found

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm not found! Please install npm
    pause
    exit /b 1
)
echo ✅ npm found

echo.
echo 🔧 Setting up Python environment...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo ✅ Python dependencies installed

REM Create .env file if it doesn't exist
if not exist ".env" (
    if exist "deployment.env.example" (
        echo Creating .env file from template...
        copy deployment.env.example .env
        echo ⚠️  Please update .env file with your configuration
    ) else (
        echo ⚠️  No .env template found. Please create .env file manually.
    )
) else (
    echo ✅ .env file found
)

echo.
echo 🎨 Setting up frontend...

REM Install Node.js dependencies
cd agent-ui
echo Installing Node.js dependencies...
call npm install
echo ✅ Node.js dependencies installed

REM Build frontend
echo Building frontend...
call npm run build
echo ✅ Frontend built successfully

cd ..

echo.
echo 🚀 Starting backend server...

REM Start backend
echo Starting backend server on http://localhost:8000...
start "Chat with DB Backend" cmd /k "venv\Scripts\activate.bat && python start_backend.py"

echo.
echo ⏳ Waiting for backend to start...
timeout /t 5 /nobreak >nul

REM Test backend
echo Testing backend connection...
curl -s http://localhost:8000/api/v1/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Backend may still be starting. Please wait a moment and try:
    echo     curl http://localhost:8000/api/v1/health
) else (
    echo ✅ Backend is running successfully!
)

echo.
echo 🎯 Setup Complete!
echo.
echo 📊 Your Chat with DB system is now running:
echo    - Backend: http://localhost:8000
echo    - API Docs: http://localhost:8000/docs
echo    - Frontend: Built and ready in agent-ui/build/
echo.
echo 🔧 To stop the backend, close the backend command window
echo 🔧 To restart, run this script again
echo.
echo 📖 For more information, see DEPLOYMENT_README.md
echo.

pause
