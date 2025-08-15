#!/bin/bash

# Chat with DB - Stop Services Script
# This script gracefully stops all running services

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping Chat with DB services...${NC}"

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

# Stop backend service
if [ -f "backend.pid" ]; then
    BACKEND_PID=$(cat backend.pid)
    if ps -p $BACKEND_PID > /dev/null; then
        echo -e "${BLUE}🔄 Stopping backend service (PID: $BACKEND_PID)...${NC}"
        kill $BACKEND_PID
        sleep 2
        
        # Force kill if still running
        if ps -p $BACKEND_PID > /dev/null; then
            echo -e "${YELLOW}⚠️  Force stopping backend...${NC}"
            kill -9 $BACKEND_PID
        fi
        
        rm -f backend.pid
        print_status "Backend service stopped"
    else
        print_warning "Backend service not running (PID file exists but process not found)"
        rm -f backend.pid
    fi
else
    # Try to find and stop backend process
    BACKEND_PROCESSES=$(pgrep -f "start_backend.py" || true)
    if [ ! -z "$BACKEND_PROCESSES" ]; then
        echo -e "${BLUE}🔄 Stopping backend processes...${NC}"
        echo $BACKEND_PROCESSES | xargs kill
        sleep 2
        
        # Force kill if still running
        REMAINING=$(pgrep -f "start_backend.py" || true)
        if [ ! -z "$REMAINING" ]; then
            echo -e "${YELLOW}⚠️  Force stopping remaining backend processes...${NC}"
            echo $REMAINING | xargs kill -9
        fi
        
        print_status "Backend processes stopped"
    else
        print_status "No backend processes found"
    fi
fi

# Stop frontend service
if [ -f "frontend.pid" ]; then
    FRONTEND_PID=$(cat frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        echo -e "${BLUE}🔄 Stopping frontend service (PID: $FRONTEND_PID)...${NC}"
        kill $FRONTEND_PID
        sleep 2
        
        # Force kill if still running
        if ps -p $FRONTEND_PID > /dev/null; then
            echo -e "${YELLOW}⚠️  Force stopping frontend...${NC}"
            kill -9 $FRONTEND_PID
        fi
        
        rm -f frontend.pid
        print_status "Frontend service stopped"
    else
        print_warning "Frontend service not running (PID file exists but process not found)"
        rm -f frontend.pid
    fi
else
    # Try to find and stop frontend process
    FRONTEND_PROCESSES=$(pgrep -f "react-scripts start" || true)
    if [ ! -z "$FRONTEND_PROCESSES" ]; then
        echo -e "${BLUE}🔄 Stopping frontend processes...${NC}"
        echo $FRONTEND_PROCESSES | xargs kill
        sleep 2
        
        # Force kill if still running
        REMAINING=$(pgrep -f "react-scripts start" || true)
        if [ ! -z "$REMAINING" ]; then
            echo -e "${YELLOW}⚠️  Force stopping remaining frontend processes...${NC}"
            echo $REMAINING | xargs kill -9
        fi
        
        print_status "Frontend processes stopped"
    else
        print_status "No frontend processes found"
    fi
fi

# Stop Docker services (if running)
if command -v docker-compose >/dev/null 2>&1; then
    if [ -f "docker-compose.yml" ]; then
        echo -e "${BLUE}🐳 Stopping Docker services...${NC}"
        docker-compose down
        print_status "Docker services stopped"
    fi
fi

# Clean up log files
if [ -f "backend.log" ]; then
    echo -e "${BLUE}🧹 Cleaning up log files...${NC}"
    rm -f backend.log
    print_status "Log files cleaned up"
fi

# Check if any processes are still running
echo -e "${BLUE}🔍 Checking for remaining processes...${NC}"

REMAINING_BACKEND=$(pgrep -f "start_backend.py" || true)
REMAINING_FRONTEND=$(pgrep -f "react-scripts start" || true)

if [ -z "$REMAINING_BACKEND" ] && [ -z "$REMAINING_FRONTEND" ]; then
    print_status "All services stopped successfully"
else
    print_warning "Some processes may still be running:"
    if [ ! -z "$REMAINING_BACKEND" ]; then
        echo -e "${YELLOW}  Backend PIDs: $REMAINING_BACKEND${NC}"
    fi
    if [ ! -z "$REMAINING_FRONTEND" ]; then
        echo -e "${YELLOW}  Frontend PIDs: $REMAINING_FRONTEND${NC}"
    fi
fi

# Final status
echo -e "${BLUE}📊 Service Status${NC}"
echo -e "${GREEN}✅ Backend: Stopped${NC}"
echo -e "${GREEN}✅ Frontend: Stopped${NC}"
echo -e "${GREEN}✅ Docker: Stopped (if running)${NC}"

print_status "All services stopped successfully!"
