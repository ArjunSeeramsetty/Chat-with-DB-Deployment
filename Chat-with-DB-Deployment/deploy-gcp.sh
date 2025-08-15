#!/bin/bash

# Chat with DB - Google Cloud Platform Deployment Script
# This script sets up and deploys the application to Google Cloud with Azure SQL support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=""
REGION="us-central1"
BACKEND_SERVICE="chat-with-db-backend"
FRONTEND_SERVICE="chat-with-db-frontend"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists gcloud; then
        print_error "gcloud CLI is not installed. Please install it first:"
        echo "https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install it first:"
        echo "https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to get project ID
get_project_id() {
    if [ -z "$PROJECT_ID" ]; then
        print_status "Getting current project ID..."
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        
        if [ -z "$PROJECT_ID" ]; then
            print_error "No project ID set. Please set it:"
            echo "gcloud config set project YOUR_PROJECT_ID"
            exit 1
        fi
    fi
    
    print_success "Using project: $PROJECT_ID"
}

# Function to enable required APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        secretmanager.googleapis.com \
        artifactregistry.googleapis.com \
        cloudresourcemanager.googleapis.com
    
    print_success "APIs enabled successfully"
}

# Function to create Artifact Registry repositories
create_artifact_registry() {
    print_status "Creating Artifact Registry repositories..."
    
    # Create backend repository
    gcloud artifacts repositories create chat-with-db-backend \
        --repository-format=docker \
        --location=$REGION \
        --description="Chat with DB Backend Images" \
        --quiet || print_warning "Backend repository may already exist"
    
    # Create frontend repository
    gcloud artifacts repositories create chat-with-db-frontend \
        --repository-format=docker \
        --location=$REGION \
        --description="Chat with DB Frontend Images" \
        --quiet || print_warning "Frontend repository may already exist"
    
    print_success "Artifact Registry repositories created"
}

# Function to create secrets
create_secrets() {
    print_status "Setting up secrets in Secret Manager..."
    
    # Create MSSQL password secret for Azure SQL
    if ! gcloud secrets describe MSSQL_PASSWORD --project=$PROJECT_ID >/dev/null 2>&1; then
        echo "Please enter your Azure SQL password:"
        read -s MSSQL_PASSWORD
        echo "$MSSQL_PASSWORD" | gcloud secrets create MSSQL_PASSWORD \
            --data-file=- \
            --project=$PROJECT_ID
        print_success "MSSQL_PASSWORD secret created for Azure SQL"
    else
        print_warning "MSSQL_PASSWORD secret already exists"
    fi
    
    # Create LLM API key secret (optional)
    if ! gcloud secrets describe LLM_API_KEY --project=$PROJECT_ID >/dev/null 2>&1; then
        echo "Do you want to create LLM_API_KEY secret? (y/n):"
        read -r CREATE_LLM_SECRET
        if [[ $CREATE_LLM_SECRET =~ ^[Yy]$ ]]; then
            echo "Please enter your LLM API key:"
            read -s LLM_API_KEY
            echo "$LLM_API_KEY" | gcloud secrets create LLM_API_KEY \
                --data-file=- \
                --project=$PROJECT_ID
            print_success "LLM_API_KEY secret created"
        fi
    else
        print_warning "LLM_API_KEY secret already exists"
    fi
    
    # Create Vector DB API key secret (optional)
    if ! gcloud secrets describe VECTOR_DB_API_KEY --project=$PROJECT_ID >/dev/null 2>&1; then
        echo "Do you want to create VECTOR_DB_API_KEY secret? (y/n):"
        read -r CREATE_VECTOR_SECRET
        if [[ $CREATE_VECTOR_SECRET =~ ^[Yy]$ ]]; then
            echo "Please enter your Vector DB API key:"
            read -s VECTOR_DB_API_KEY
            echo "$VECTOR_DB_API_KEY" | gcloud secrets create VECTOR_DB_API_KEY \
                --data-file=- \
                --project=$PROJECT_ID
            print_success "VECTOR_DB_API_KEY secret created"
        fi
    else
        print_warning "VECTOR_DB_API_KEY secret already exists"
    fi
}

# Function to configure Cloud Build
configure_cloud_build() {
    print_status "Configuring Cloud Build..."
    
    # Grant Cloud Build service account necessary permissions
    PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
    CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${CLOUDBUILD_SA}" \
        --role="roles/run.admin"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${CLOUDBUILD_SA}" \
        --role="roles/iam.serviceAccountUser"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${CLOUDBUILD_SA}" \
        --role="roles/secretmanager.secretAccessor"
    
    print_success "Cloud Build configured successfully"
}

# Function to test Azure SQL connection locally
test_azure_sql_connection() {
    print_status "Testing Azure SQL connection locally..."
    
    if [ -f "test_azure_connection.py" ]; then
        print_status "Running Azure SQL connection test..."
        if python test_azure_connection.py; then
            print_success "Azure SQL connection test passed"
        else
            print_warning "Azure SQL connection test failed - deployment may still work if credentials are correct"
            echo "Do you want to continue with deployment? (y/n):"
            read -r CONTINUE_DEPLOYMENT
            if [[ ! $CONTINUE_DEPLOYMENT =~ ^[Yy]$ ]]; then
                print_error "Deployment aborted by user"
                exit 1
            fi
        fi
    else
        print_warning "Azure SQL connection test script not found - skipping local test"
    fi
}

# Function to deploy using Cloud Build
deploy_with_cloud_build() {
    print_status "Deploying with Cloud Build..."
    
    # Submit build
    gcloud builds submit \
        --config=cloudbuild.yaml \
        --project=$PROJECT_ID \
        --substitutions=_REGION=$REGION,_BACKEND_SERVICE=$BACKEND_SERVICE,_FRONTEND_SERVICE=$FRONTEND_SERVICE
    
    print_success "Deployment completed successfully!"
}

# Function to get service URLs
get_service_urls() {
    print_status "Getting service URLs..."
    
    BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE \
        --region=$REGION \
        --format="value(status.url)" \
        --project=$PROJECT_ID)
    
    FRONTEND_URL=$(gcloud run services describe $FRONTEND_SERVICE \
        --region=$REGION \
        --format="value(status.url)" \
        --project=$PROJECT_ID)
    
    echo ""
    print_success "Deployment completed!"
    echo ""
    echo "Service URLs:"
    echo "  Backend:  $BACKEND_URL"
    echo "  Frontend: $FRONTEND_URL"
    echo ""
    echo "Health Check:"
    echo "  Backend:  $BACKEND_URL/api/v1/health"
    echo "  Frontend: $FRONTEND_URL/health"
    echo ""
    echo "Azure SQL Status:"
    echo "  Azure SQL Status: $BACKEND_URL/api/v1/azure-sql/status"
    echo ""
}

# Function to verify Azure SQL connection in deployed service
verify_azure_sql_deployment() {
    print_status "Verifying Azure SQL connection in deployed service..."
    
    BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE \
        --region=$REGION \
        --format="value(status.url)" \
        --project=$PROJECT_ID)
    
    print_status "Testing Azure SQL connection at: $BACKEND_URL/api/v1/health"
    
    # Wait a bit for service to be ready
    sleep 10
    
    # Test health endpoint
    if curl -f "$BACKEND_URL/api/v1/health" >/dev/null 2>&1; then
        print_success "Backend service is responding"
        
        # Test Azure SQL specific endpoint
        if curl -f "$BACKEND_URL/api/v1/azure-sql/status" >/dev/null 2>&1; then
            print_success "Azure SQL status endpoint is working"
        else
            print_warning "Azure SQL status endpoint is not responding - check logs"
        fi
    else
        print_warning "Backend service is not responding - check logs"
    fi
}

# Main deployment function
main() {
    echo "=========================================="
    echo "Chat with DB - Google Cloud Deployment"
    echo "with Azure SQL Server Support"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    get_project_id
    enable_apis
    create_artifact_registry
    create_secrets
    configure_cloud_build
    
    # Test Azure SQL connection locally before deployment
    test_azure_sql_connection
    
    deploy_with_cloud_build
    get_service_urls
    
    # Verify deployment
    verify_azure_sql_deployment
    
    echo "=========================================="
    print_success "Deployment completed successfully!"
    echo "=========================================="
}

# Run main function
main "$@"
