# Chat with DB - PowerShell Deployment Script
# This script automates the deployment process on Windows

param(
    [switch]$Dev,
    [switch]$Docker
)

# Error handling
$ErrorActionPreference = "Stop"

# Configuration
$ProjectName = "Chat-with-DB"
$BackendPort = 8000
$FrontendPort = 3000

Write-Host "🚀 Starting deployment of $ProjectName..." -ForegroundColor Blue

# Function to check if command exists
function Test-Command {
    param($CommandName)
    try {
        Get-Command $CommandName -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check if port is available
function Test-Port {
    param($Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $false  # Port is in use
    }
    catch {
        return $true   # Port is available
    }
}

# Function to print status
function Write-Status {
    param($Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Blue

# Check Python
if (Test-Command "python") {
    $PythonCmd = "python"
    Write-Status "Python found"
} elseif (Test-Command "python3") {
    $PythonCmd = "python3"
    Write-Status "Python 3 found"
} else {
    Write-Error "Python not found. Please install Python 3.8+"
    exit 1
}

# Check Node.js
if (Test-Command "node") {
    Write-Status "Node.js found"
} else {
    Write-Warning "Node.js not found. Frontend deployment will be skipped."
    $SkipFrontend = $true
}

# Check npm
if (Test-Command "npm") {
    Write-Status "npm found"
} else {
    Write-Warning "npm not found. Frontend deployment will be skipped."
    $SkipFrontend = $true
}

# Check Docker (optional)
if (Test-Command "docker") {
    Write-Status "Docker found - containerized deployment available"
    $DockerAvailable = $true
} else {
    Write-Warning "Docker not found - using local deployment only"
    $DockerAvailable = $false
}

# Check if ports are available
if (Test-Port $BackendPort) {
    Write-Status "Backend port $BackendPort is available"
} else {
    Write-Warning "Backend port $BackendPort is in use"
}

if (-not $SkipFrontend -and (Test-Port $FrontendPort)) {
    Write-Status "Frontend port $FrontendPort is available"
} elseif (-not $SkipFrontend) {
    Write-Warning "Frontend port $FrontendPort is in use"
}

# Create virtual environment
Write-Host "🐍 Setting up Python environment..." -ForegroundColor Blue

if (-not (Test-Path "venv")) {
    & $PythonCmd -m venv venv
    Write-Status "Virtual environment created"
} else {
    Write-Status "Virtual environment already exists"
}

# Activate virtual environment
& "venv\Scripts\Activate.ps1"
Write-Status "Virtual environment activated"

# Install Python dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Blue
& $PythonCmd -m pip install --upgrade pip
& $PythonCmd -m pip install -r requirements.txt
Write-Status "Python dependencies installed"

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚙️  Creating .env file from template..." -ForegroundColor Blue
    if (Test-Path "deployment.env.example") {
        Copy-Item "deployment.env.example" ".env"
        Write-Warning "Please update .env file with your configuration"
    } else {
        Write-Warning "No .env template found. Please create .env file manually."
    }
} else {
    Write-Status ".env file found"
}

# Deploy backend
Write-Host "🔧 Deploying backend..." -ForegroundColor Blue

# Check if backend is already running
$BackendProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*start_backend.py*" }
if ($BackendProcesses) {
    Write-Warning "Backend is already running. Stopping existing processes..."
    $BackendProcesses | Stop-Process -Force
    Start-Sleep -Seconds 2
}

# Start backend in background
Write-Host "🚀 Starting backend server..." -ForegroundColor Blue
Start-Process -FilePath $PythonCmd -ArgumentList "start_backend.py" -WindowStyle Hidden -RedirectStandardOutput "backend.log" -RedirectStandardError "backend.log"
Start-Sleep -Seconds 5

# Check if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$BackendPort/api/v1/health" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Status "Backend is running on port $BackendPort"
    } else {
        throw "Backend health check failed"
    }
} catch {
    Write-Error "Backend failed to start. Check backend.log for details."
    exit 1
}

# Deploy frontend (if available)
if (-not $SkipFrontend) {
    Write-Host "🎨 Deploying frontend..." -ForegroundColor Blue
    
    Set-Location "agent-ui"
    
    # Install Node.js dependencies
    Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Blue
    & npm install
    Write-Status "Node.js dependencies installed"
    
    # Build frontend
    Write-Host "🔨 Building frontend..." -ForegroundColor Blue
    & npm run build
    Write-Status "Frontend built successfully"
    
    # Start frontend development server (optional)
    if ($Dev) {
        Write-Host "🚀 Starting frontend development server..." -ForegroundColor Blue
        Start-Process -FilePath "npm" -ArgumentList "start" -WindowStyle Hidden
        Write-Status "Frontend development server started on port $FrontendPort"
    }
    
    Set-Location ".."
} else {
    Write-Warning "Frontend deployment skipped"
}

# Docker deployment (if available)
if ($DockerAvailable -and $Docker) {
    Write-Host "🐳 Deploying with Docker..." -ForegroundColor Blue
    
    # Build and start services
    & docker-compose up -d --build
    
    Write-Status "Docker services started"
    Write-Status "Backend: http://localhost:8000"
    Write-Status "Frontend: http://localhost:80 (via nginx)"
    Write-Status "Ollama: http://localhost:11434"
}

# Final status
Write-Host "📊 Deployment Summary" -ForegroundColor Blue
Write-Host "✅ Backend: http://localhost:$BackendPort" -ForegroundColor Green
if (-not $SkipFrontend) {
    if ($Dev) {
        Write-Host "✅ Frontend: http://localhost:$FrontendPort" -ForegroundColor Green
    } else {
        Write-Host "✅ Frontend: Built and ready" -ForegroundColor Green
    }
}
if ($DockerAvailable) {
    Write-Host "✅ Docker: Available for containerized deployment" -ForegroundColor Green
}

Write-Host "🎯 Next steps:" -ForegroundColor Blue
Write-Host "1. Test the API: Invoke-WebRequest http://localhost:$BackendPort/api/v1/health"
Write-Host "2. View API docs: http://localhost:$BackendPort/docs"
Write-Host "3. Check logs: Get-Content backend.log -Wait"
Write-Host "4. Stop services: .\stop.ps1"

Write-Status "Deployment completed successfully!"
