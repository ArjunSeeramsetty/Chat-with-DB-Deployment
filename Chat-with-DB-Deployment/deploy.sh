#!/bin/bash

# Chat with DB - Deployment Script
# This script automates the deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Chat-with-DB"
BACKEND_PORT=8000
FRONTEND_PORT=3000

echo -e "${BLUE}🚀 Starting deployment of ${PROJECT_NAME}...${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
port_available() {
    ! nc -z localhost $1 2>/dev/null
}

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

# Check Python
if command_exists python3; then
    PYTHON_CMD="python3"
    print_status "Python 3 found"
elif command_exists python; then
    PYTHON_CMD="python"
    print_status "Python found"
else
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# Check Node.js
if command_exists node; then
    print_status "Node.js found"
else
    print_warning "Node.js not found. Frontend deployment will be skipped."
    SKIP_FRONTEND=true
fi

# Check npm
if command_exists npm; then
    print_status "npm found"
else
    print_warning "npm not found. Frontend deployment will be skipped."
    SKIP_FRONTEND=true
fi

# Check Docker (optional)
if command_exists docker; then
    print_status "Docker found - containerized deployment available"
    DOCKER_AVAILABLE=true
else
    print_warning "Docker not found - using local deployment only"
    DOCKER_AVAILABLE=false
fi

# Check if ports are available
if port_available $BACKEND_PORT; then
    print_status "Backend port $BACKEND_PORT is available"
else
    print_warning "Backend port $BACKEND_PORT is in use"
fi

if [ "$SKIP_FRONTEND" != "true" ] && port_available $FRONTEND_PORT; then
    print_status "Frontend port $FRONTEND_PORT is available"
elif [ "$SKIP_FRONTEND" != "true" ]; then
    print_warning "Frontend port $FRONTEND_PORT is in use"
fi

# Create virtual environment
echo -e "${BLUE}🐍 Setting up Python environment...${NC}"

if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Install Python dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
print_status "Python dependencies installed"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${BLUE}⚙️  Creating .env file from template...${NC}"
    if [ -f "deployment.env.example" ]; then
        cp deployment.env.example .env
        print_warning "Please update .env file with your configuration"
    else
        print_warning "No .env template found. Please create .env file manually."
    fi
else
    print_status ".env file found"
fi

# Deploy backend
echo -e "${BLUE}🔧 Deploying backend...${NC}"

# Check if backend is already running
if pgrep -f "start_backend.py" > /dev/null; then
    print_warning "Backend is already running. Stopping existing process..."
    pkill -f "start_backend.py"
    sleep 2
fi

# Start backend in background
echo -e "${BLUE}🚀 Starting backend server...${NC}"
nohup $PYTHON_CMD start_backend.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${BLUE}⏳ Waiting for backend to start...${NC}"
sleep 5

# Check if backend is running
if curl -s http://localhost:$BACKEND_PORT/api/v1/health > /dev/null; then
    print_status "Backend is running on port $BACKEND_PORT"
    echo $BACKEND_PID > backend.pid
else
    print_error "Backend failed to start. Check backend.log for details."
    exit 1
fi

# Deploy frontend (if available)
if [ "$SKIP_FRONTEND" != "true" ]; then
    echo -e "${BLUE}🎨 Deploying frontend...${NC}"
    
    cd agent-ui
    
    # Install Node.js dependencies
    echo -e "${BLUE}📦 Installing Node.js dependencies...${NC}"
    npm install
    print_status "Node.js dependencies installed"
    
    # Build frontend
    echo -e "${BLUE}🔨 Building frontend...${NC}"
    npm run build
    print_status "Frontend built successfully"
    
    # Start frontend development server (optional)
    if [ "$1" = "--dev" ]; then
        echo -e "${BLUE}🚀 Starting frontend development server...${NC}"
        npm start &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > ../frontend.pid
        print_status "Frontend development server started on port $FRONTEND_PORT"
    fi
    
    cd ..
else
    print_warning "Frontend deployment skipped"
fi

# Docker deployment (if available)
if [ "$DOCKER_AVAILABLE" = "true" ] && [ "$1" = "--docker" ]; then
    echo -e "${BLUE}🐳 Deploying with Docker...${NC}"
    
    # Build and start services
    docker-compose up -d --build
    
    print_status "Docker services started"
    print_status "Backend: http://localhost:8000"
    print_status "Frontend: http://localhost:80 (via nginx)"
    print_status "Ollama: http://localhost:11434"
fi

# Final status
echo -e "${BLUE}📊 Deployment Summary${NC}"
echo -e "${GREEN}✅ Backend: http://localhost:$BACKEND_PORT${NC}"
if [ "$SKIP_FRONTEND" != "true" ]; then
    if [ "$1" = "--dev" ]; then
        echo -e "${GREEN}✅ Frontend: http://localhost:$FRONTEND_PORT${NC}"
    else
        echo -e "${GREEN}✅ Frontend: Built and ready${NC}"
    fi
fi
if [ "$DOCKER_AVAILABLE" = "true" ]; then
    echo -e "${GREEN}✅ Docker: Available for containerized deployment${NC}"
fi

echo -e "${BLUE}🎯 Next steps:${NC}"
echo -e "1. Test the API: curl http://localhost:$BACKEND_PORT/api/v1/health"
echo -e "2. View API docs: http://localhost:$BACKEND_PORT/docs"
echo -e "3. Check logs: tail -f backend.log"
echo -e "4. Stop services: ./stop.sh"

print_status "Deployment completed successfully!"
