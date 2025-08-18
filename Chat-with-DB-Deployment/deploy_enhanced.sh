#!/bin/bash

# =============================================================================
# ENHANCED CHAT-WITH-DB DEPLOYMENT SCRIPT
# =============================================================================
# This script deploys the complete enhanced AI/ML application to Google Cloud

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="powerflow-467113"
REGION="us-central1"
REPOSITORY="chat-with-db-repo"
BACKEND_SERVICE="chat-with-db-backend"
FRONTEND_SERVICE="chat-with-db-frontend"

echo -e "${BLUE}🚀 Enhanced Chat-with-DB Deployment Script${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    if ! command_exists gcloud; then
        echo -e "${RED}❌ Google Cloud CLI (gcloud) not found. Please install it first.${NC}"
        exit 1
    fi
    
    if ! command_exists docker; then
        echo -e "${RED}❌ Docker not found. Please install it first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Function to authenticate with Google Cloud
authenticate_gcloud() {
    echo -e "${YELLOW}🔐 Authenticating with Google Cloud...${NC}"
    
    # Check if already authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        echo -e "${GREEN}✅ Already authenticated with Google Cloud${NC}"
    else
        echo -e "${YELLOW}Please authenticate with Google Cloud...${NC}"
        gcloud auth login
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    echo -e "${GREEN}✅ Project set to: $PROJECT_ID${NC}"
}

# Function to enable required APIs
enable_apis() {
    echo -e "${YELLOW}🔌 Enabling required Google Cloud APIs...${NC}"
    
    local apis=(
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "artifactregistry.googleapis.com"
        "secretmanager.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        echo -e "${BLUE}Enabling $api...${NC}"
        gcloud services enable $api --quiet
    done
    
    echo -e "${GREEN}✅ All required APIs enabled${NC}"
}

# Function to create Artifact Registry repository
create_artifact_repository() {
    echo -e "${YELLOW}🏗️  Creating Artifact Registry repository...${NC}"
    
    if gcloud artifacts repositories describe $REPOSITORY --location=$REGION >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Repository $REPOSITORY already exists${NC}"
    else
        echo -e "${BLUE}Creating repository $REPOSITORY...${NC}"
        gcloud artifacts repositories create $REPOSITORY \
            --repository-format=docker \
            --location=$REGION \
            --description="Chat-with-DB Docker images"
        echo -e "${GREEN}✅ Repository created successfully${NC}"
    fi
}

# Function to configure Docker for Artifact Registry
configure_docker() {
    echo -e "${YELLOW}🐳 Configuring Docker for Artifact Registry...${NC}"
    
    gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
    echo -e "${GREEN}✅ Docker configured for Artifact Registry${NC}"
}

# Function to build and push backend image
build_backend() {
    echo -e "${YELLOW}🔨 Building backend Docker image...${NC}"
    
    local image_tag="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/backend:latest"
    
    echo -e "${BLUE}Building image: $image_tag${NC}"
    docker build -t $image_tag -f Dockerfile .
    
    echo -e "${BLUE}Pushing image to Artifact Registry...${NC}"
    docker push $image_tag
    
    echo -e "${GREEN}✅ Backend image built and pushed successfully${NC}"
}

# Function to build and push frontend image
build_frontend() {
    echo -e "${YELLOW}🔨 Building frontend Docker image...${NC}"
    
    local image_tag="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/frontend:latest"
    
    echo -e "${BLUE}Building image: $image_tag${NC}"
    docker build -t $image_tag -f agent-ui/Dockerfile.frontend agent-ui/
    
    echo -e "${BLUE}Pushing image to Artifact Registry...${NC}"
    docker push $image_tag
    
    echo -e "${GREEN}✅ Frontend image built and pushed successfully${NC}"
}

# Function to deploy backend to Cloud Run
deploy_backend() {
    echo -e "${YELLOW}🚀 Deploying backend to Cloud Run...${NC}"
    
    local image_url="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/backend:latest"
    
    gcloud run deploy $BACKEND_SERVICE \
        --image=$image_url \
        --platform=managed \
        --region=$REGION \
        --allow-unauthenticated \
        --memory=4Gi \
        --cpu=2 \
        --max-instances=10 \
        --timeout=900 \
        --concurrency=80 \
        --set-env-vars="ENVIRONMENT=production,APP_VERSION=latest" \
        --quiet
    
    # Get the service URL
    local service_url=$(gcloud run services describe $BACKEND_SERVICE --region=$REGION --format="value(status.url)")
    echo -e "${GREEN}✅ Backend deployed successfully${NC}"
    echo -e "${BLUE}🌐 Backend URL: $service_url${NC}"
    
    # Store the URL for frontend configuration
    echo $service_url > .backend_url
}

# Function to deploy frontend to Cloud Run
deploy_frontend() {
    echo -e "${YELLOW}🚀 Deploying frontend to Cloud Run...${NC}"
    
    local image_url="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/frontend:latest"
    local backend_url=$(cat .backend_url)
    
    gcloud run deploy $FRONTEND_SERVICE \
        --image=$image_url \
        --platform=managed \
        --region=$REGION \
        --allow-unauthenticated \
        --memory=1Gi \
        --cpu=1 \
        --max-instances=5 \
        --timeout=300 \
        --concurrency=40 \
        --set-env-vars="REACT_APP_API_BASE=$backend_url" \
        --quiet
    
    # Get the service URL
    local service_url=$(gcloud run services describe $FRONTEND_SERVICE --region=$REGION --format="value(status.url)")
    echo -e "${GREEN}✅ Frontend deployed successfully${NC}"
    echo -e "${BLUE}🌐 Frontend URL: $service_url${NC}"
}

# Function to run health checks
health_check() {
    echo -e "${YELLOW}🏥 Running health checks...${NC}"
    
    local backend_url=$(cat .backend_url)
    local frontend_url=$(gcloud run services describe $FRONTEND_SERVICE --region=$REGION --format="value(status.url)")
    
    echo -e "${BLUE}Checking backend health...${NC}"
    if curl -f "$backend_url/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend health check passed${NC}"
    else
        echo -e "${RED}❌ Backend health check failed${NC}"
    fi
    
    echo -e "${BLUE}Checking frontend...${NC}"
    if curl -f "$frontend_url" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend check passed${NC}"
    else
        echo -e "${RED}❌ Frontend check failed${NC}"
    fi
}

# Function to display deployment summary
deployment_summary() {
    echo -e "${BLUE}📊 Deployment Summary${NC}"
    echo -e "${BLUE}==================${NC}"
    
    local backend_url=$(cat .backend_url)
    local frontend_url=$(gcloud run services describe $FRONTEND_SERVICE --region=$REGION --format="value(status.url)")
    
    echo -e "${GREEN}✅ Backend Service: $BACKEND_SERVICE${NC}"
    echo -e "${BLUE}   URL: $backend_url${NC}"
    echo -e "${BLUE}   Region: $REGION${NC}"
    echo -e "${BLUE}   Memory: 4Gi${NC}"
    echo -e "${BLUE}   CPU: 2${NC}"
    
    echo -e "${GREEN}✅ Frontend Service: $FRONTEND_SERVICE${NC}"
    echo -e "${BLUE}   URL: $frontend_url${NC}"
    echo -e "${BLUE}   Region: $REGION${NC}"
    echo -e "${BLUE}   Memory: 1Gi${NC}"
    echo -e "${BLUE}   CPU: 1${NC}"
    
    echo -e "${GREEN}✅ Artifact Registry: $REPOSITORY${NC}"
    echo -e "${BLUE}   Location: $REGION${NC}"
    
    echo -e "${YELLOW}🎯 Next Steps:${NC}"
    echo -e "${BLUE}   1. Configure your domain names${NC}"
    echo -e "${BLUE}   2. Set up SSL certificates${NC}"
    echo -e "${BLUE}   3. Configure monitoring and alerting${NC}"
    echo -e "${BLUE}   4. Set up CI/CD pipeline${NC}"
}

# Main deployment flow
main() {
    echo -e "${BLUE}Starting enhanced deployment...${NC}"
    
    check_prerequisites
    authenticate_gcloud
    enable_apis
    create_artifact_repository
    configure_docker
    build_backend
    build_frontend
    deploy_backend
    deploy_frontend
    health_check
    deployment_summary
    
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
}

# Run main function
main "$@"
