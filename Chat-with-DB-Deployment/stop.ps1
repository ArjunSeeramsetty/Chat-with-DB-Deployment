# Chat with DB - PowerShell Stop Services Script
# This script gracefully stops all running services on Windows

# Error handling
$ErrorActionPreference = "Continue"

Write-Host "🛑 Stopping Chat with DB services..." -ForegroundColor Blue

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

# Stop backend service
Write-Host "🔄 Stopping backend service..." -ForegroundColor Blue

$BackendProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { 
    $_.CommandLine -like "*start_backend.py*" -or $_.ProcessName -eq "python" 
}

if ($BackendProcesses) {
    Write-Host "Found $($BackendProcesses.Count) backend process(es)" -ForegroundColor Yellow
    
    foreach ($process in $BackendProcesses) {
        try {
            Write-Host "Stopping process PID: $($process.Id)" -ForegroundColor Yellow
            $process.Kill()
            Start-Sleep -Seconds 1
            
            # Force kill if still running
            if (-not $process.HasExited) {
                Write-Warning "Force stopping process PID: $($process.Id)"
                $process.Kill($true)
            }
        }
        catch {
            Write-Warning "Error stopping process PID: $($process.Id): $($_.Exception.Message)"
        }
    }
    
    Write-Status "Backend processes stopped"
} else {
    Write-Status "No backend processes found"
}

# Stop frontend service
Write-Host "🔄 Stopping frontend service..." -ForegroundColor Blue

$FrontendProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { 
    $_.CommandLine -like "*react-scripts start*" -or $_.ProcessName -eq "node" 
}

if ($FrontendProcesses) {
    Write-Host "Found $($FrontendProcesses.Count) frontend process(es)" -ForegroundColor Yellow
    
    foreach ($process in $FrontendProcesses) {
        try {
            Write-Host "Stopping process PID: $($process.Id)" -ForegroundColor Yellow
            $process.Kill()
            Start-Sleep -Seconds 1
            
            # Force kill if still running
            if (-not $process.HasExited) {
                Write-Warning "Force stopping process PID: $($process.Id)"
                $process.Kill($true)
            }
        }
        catch {
            Write-Warning "Error stopping process PID: $($process.Id): $($_.Exception.Message)"
        }
    }
    
    Write-Status "Frontend processes stopped"
} else {
    Write-Status "No frontend processes found"
}

# Stop Docker services (if running)
if (Test-Command "docker-compose") {
    if (Test-Path "docker-compose.yml") {
        Write-Host "🐳 Stopping Docker services..." -ForegroundColor Blue
        try {
            & docker-compose down
            Write-Status "Docker services stopped"
        }
        catch {
            Write-Warning "Error stopping Docker services: $($_.Exception.Message)"
        }
    }
}

# Clean up log files
if (Test-Path "backend.log") {
    Write-Host "🧹 Cleaning up log files..." -ForegroundColor Blue
    try {
        Remove-Item "backend.log" -Force
        Write-Status "Log files cleaned up"
    }
    catch {
        Write-Warning "Error cleaning up log files: $($_.Exception.Message)"
    }
}

# Check if any processes are still running
Write-Host "🔍 Checking for remaining processes..." -ForegroundColor Blue

$RemainingBackend = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { 
    $_.CommandLine -like "*start_backend.py*" 
}
$RemainingFrontend = Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { 
    $_.CommandLine -like "*react-scripts start*" 
}

if (-not $RemainingBackend -and -not $RemainingFrontend) {
    Write-Status "All services stopped successfully"
} else {
    Write-Warning "Some processes may still be running:"
    if ($RemainingBackend) {
        Write-Host "  Backend PIDs: $($RemainingBackend.Id -join ', ')" -ForegroundColor Yellow
    }
    if ($RemainingFrontend) {
        Write-Host "  Frontend PIDs: $($RemainingFrontend.Id -join ', ')" -ForegroundColor Yellow
    }
}

# Final status
Write-Host "📊 Service Status" -ForegroundColor Blue
Write-Host "✅ Backend: Stopped" -ForegroundColor Green
Write-Host "✅ Frontend: Stopped" -ForegroundColor Green
Write-Host "✅ Docker: Stopped (if running)" -ForegroundColor Green

Write-Status "All services stopped successfully!"
