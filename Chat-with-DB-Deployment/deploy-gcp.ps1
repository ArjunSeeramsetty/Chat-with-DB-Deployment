# Chat with DB - Google Cloud Platform Deployment Script (PowerShell)
# This script sets up and deploys the application to Google Cloud

param(
    [string]$ProjectId = "",
    [string]$Region = "us-central1",
    [string]$BackendService = "chat-with-db-backend",
    [string]$FrontendService = "chat-with-db-frontend"
)

# Error handling
$ErrorActionPreference = "Stop"

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

# Function to check if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    if (-not (Test-Command "gcloud")) {
        Write-Error "gcloud CLI is not installed. Please install it first:"
        Write-Host "https://cloud.google.com/sdk/docs/install" -ForegroundColor $Yellow
        exit 1
    }
    
    if (-not (Test-Command "docker")) {
        Write-Error "Docker is not installed. Please install it first:"
        Write-Host "https://docs.docker.com/get-docker/" -ForegroundColor $Yellow
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

# Function to get project ID
function Get-ProjectId {
    if ([string]::IsNullOrEmpty($ProjectId)) {
        Write-Status "Getting current project ID..."
        $ProjectId = gcloud config get-value project 2>$null
        
        if ([string]::IsNullOrEmpty($ProjectId)) {
            Write-Error "No project ID set. Please set it:"
            Write-Host "gcloud config set project YOUR_PROJECT_ID" -ForegroundColor $Yellow
            exit 1
        }
    }
    
    Write-Success "Using project: $ProjectId"
}

# Function to enable required APIs
function Enable-APIs {
    Write-Status "Enabling required Google Cloud APIs..."
    
    gcloud services enable `
        cloudbuild.googleapis.com `
        run.googleapis.com `
        secretmanager.googleapis.com `
        artifactregistry.googleapis.com `
        cloudresourcemanager.googleapis.com
    
    Write-Success "APIs enabled successfully"
}

# Function to create Artifact Registry repositories
function New-ArtifactRegistry {
    Write-Status "Creating Artifact Registry repositories..."
    
    # Create backend repository
    try {
        gcloud artifacts repositories create chat-with-db-backend `
            --repository-format=docker `
            --location=$Region `
            --description="Chat with DB Backend Images" `
            --quiet
        Write-Success "Backend repository created"
    }
    catch {
        Write-Warning "Backend repository may already exist"
    }
    
    # Create frontend repository
    try {
        gcloud artifacts repositories create chat-with-db-frontend `
            --repository-format=docker `
            --location=$Region `
            --description="Chat with DB Frontend Images" `
            --quiet
        Write-Success "Frontend repository created"
    }
    catch {
        Write-Warning "Frontend repository may already exist"
    }
    
    Write-Success "Artifact Registry repositories created"
}

# Function to create secrets
function New-Secrets {
    Write-Status "Setting up secrets in Secret Manager..."
    
    # Create MSSQL password secret
    try {
        $secret = gcloud secrets describe MSSQL_PASSWORD --project=$ProjectId 2>$null
        Write-Warning "MSSQL_PASSWORD secret already exists"
    }
    catch {
        $MSSQLPassword = Read-Host "Please enter your MSSQL password" -AsSecureString
        $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($MSSQLPassword)
        $PlainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
        
        $PlainPassword | gcloud secrets create MSSQL_PASSWORD --data-file=-
        Write-Success "MSSQL_PASSWORD secret created"
    }
    
    # Create LLM API key secret (optional)
    try {
        $secret = gcloud secrets describe LLM_API_KEY --project=$ProjectId 2>$null
        Write-Warning "LLM_API_KEY secret already exists"
    }
    catch {
        $CreateLLMSecret = Read-Host "Do you want to create LLM_API_KEY secret? (y/n)"
        if ($CreateLLMSecret -eq "y" -or $CreateLLMSecret -eq "Y") {
            $LLMApiKey = Read-Host "Please enter your LLM API key" -AsSecureString
            $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($LLMApiKey)
            $PlainLLMApiKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
            
            $PlainLLMApiKey | gcloud secrets create LLM_API_KEY --data-file=-
            Write-Success "LLM_API_KEY secret created"
        }
    }
    
    # Create Vector DB API key secret (optional)
    try {
        $secret = gcloud secrets describe VECTOR_DB_API_KEY --project=$ProjectId 2>$null
        Write-Warning "VECTOR_DB_API_KEY secret already exists"
    }
    catch {
        $CreateVectorSecret = Read-Host "Do you want to create VECTOR_DB_API_KEY secret? (y/n)"
        if ($CreateVectorSecret -eq "y" -or $CreateVectorSecret -eq "Y") {
            $VectorApiKey = Read-Host "Please enter your Vector DB API key" -AsSecureString
            $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($VectorApiKey)
            $PlainVectorApiKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
            
            $PlainVectorApiKey | gcloud secrets create VECTOR_DB_API_KEY --data-file=-
            Write-Success "VECTOR_DB_API_KEY secret created"
        }
    }
}

# Function to configure Cloud Build
function Set-CloudBuild {
    Write-Status "Configuring Cloud Build..."
    
    # Grant Cloud Build service account necessary permissions
    $ProjectNumber = gcloud projects describe $ProjectId --format="value(projectNumber)"
    $CloudBuildSA = "$ProjectNumber@cloudbuild.gserviceaccount.com"
    
    gcloud projects add-iam-policy-binding $ProjectId `
        --member="serviceAccount:$CloudBuildSA" `
        --role="roles/run.admin"
    
    gcloud projects add-iam-policy-binding $ProjectId `
        --member="serviceAccount:$CloudBuildSA" `
        --role="roles/iam.serviceAccountUser"
    
    gcloud projects add-iam-policy-binding $ProjectId `
        --member="serviceAccount:$CloudBuildSA" `
        --role="roles/secretmanager.secretAccessor"
    
    Write-Success "Cloud Build configured successfully"
}

# Function to deploy using Cloud Build
function Deploy-WithCloudBuild {
    Write-Status "Deploying with Cloud Build..."
    
    # Submit build
    gcloud builds submit `
        --config=cloudbuild.yaml `
        --project=$ProjectId `
        --substitutions=_REGION=$Region,_BACKEND_SERVICE=$BackendService,_FRONTEND_SERVICE=$FrontendService
    
    Write-Success "Deployment completed successfully!"
}

# Function to get service URLs
function Get-ServiceUrls {
    Write-Status "Getting service URLs..."
    
    $BackendUrl = gcloud run services describe $BackendService `
        --region=$Region `
        --format="value(status.url)" `
        --project=$ProjectId
    
    $FrontendUrl = gcloud run services describe $FrontendService `
        --region=$Region `
        --format="value(status.url)" `
        --project=$ProjectId
    
    Write-Host ""
    Write-Success "Deployment completed!"
    Write-Host ""
    Write-Host "Service URLs:"
    Write-Host "  Backend:  $BackendUrl"
    Write-Host "  Frontend: $FrontendUrl"
    Write-Host ""
    Write-Host "Health Check:"
    Write-Host "  Backend:  $BackendUrl/api/v1/health"
    Write-Host "  Frontend: $FrontendUrl/health"
    Write-Host ""
}

# Main deployment function
function Main {
    Write-Host "==========================================" -ForegroundColor $Blue
    Write-Host "Chat with DB - Google Cloud Deployment" -ForegroundColor $Blue
    Write-Host "==========================================" -ForegroundColor $Blue
    Write-Host ""
    
    Test-Prerequisites
    Get-ProjectId
    Enable-APIs
    New-ArtifactRegistry
    New-Secrets
    Set-CloudBuild
    Deploy-WithCloudBuild
    Get-ServiceUrls
    
    Write-Host "==========================================" -ForegroundColor $Blue
    Write-Success "Deployment completed successfully!"
    Write-Host "==========================================" -ForegroundColor $Blue
}

# Run main function
try {
    Main
}
catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    exit 1
}
