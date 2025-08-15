# Chat with DB - Production Deployment Guide

This guide covers advanced production deployment scenarios including load balancing, monitoring, security, and scaling.

## 🚀 **Production Architecture Overview**

```
Internet → Load Balancer → Nginx → Backend Cluster → Database
                    ↓
                Frontend (CDN)
```

## 🔧 **Prerequisites**

- **Server**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Resources**: Minimum 2GB RAM, 2 vCPUs
- **Storage**: 20GB+ available space
- **Network**: Public IP with ports 80, 443, 8000 open
- **Domain**: SSL certificate for HTTPS

## 🐳 **Docker Production Deployment**

### **1. Environment Setup**

Create production environment file:
```bash
cp deployment.env.example .env.production
```

Update `.env.production`:
```env
# Production settings
NODE_ENV=production
LOG_LEVEL=WARNING
ENABLE_DEBUG=false
ENABLE_CORS=false

# Database
DATABASE_PATH=/data/power_data.db

# Security
ENABLE_QUERY_SAFETY_CHECK=true
MAX_QUERY_LENGTH=500

# Performance
API_WORKERS=4
SQL_TIMEOUT=15
MAX_ROWS=500
```

### **2. Production Docker Compose**

```bash
# Start production services
docker-compose --env-file .env.production -f docker-compose.yml --profile prod up -d

# Or start with specific services
docker-compose --env-file .env.production -f docker-compose.yml --profile prod --profile cache up -d
```

### **3. SSL Certificate Setup**

```bash
# Create SSL directory
mkdir -p ssl

# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/key.pem -out ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# For production, use Let's Encrypt or commercial certificates
```

## 🏗️ **Manual Server Deployment**

### **1. Server Preparation**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv nginx redis-server

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install PM2 for process management
sudo npm install -g pm2
```

### **2. Application Deployment**

```bash
# Clone repository
git clone <your-repo-url> /opt/chat-with-db
cd /opt/chat-with-db

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Build frontend
cd agent-ui
npm install
npm run build
cd ..
```

### **3. Systemd Service Setup**

Create `/etc/systemd/system/chat-with-db.service`:
```ini
[Unit]
Description=Chat with DB Backend
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/chat-with-db
Environment=PATH=/opt/chat-with-db/venv/bin
ExecStart=/opt/chat-with-db/venv/bin/gunicorn backend.main:app \
    -w 4 -k uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --access-logfile /var/log/chat-with-db/access.log \
    --error-logfile /var/log/chat-with-db/error.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable chat-with-db
sudo systemctl start chat-with-db
```

### **4. Nginx Configuration**

Copy nginx config:
```bash
sudo cp nginx.conf /etc/nginx/sites-available/chat-with-db
sudo ln -s /etc/nginx/sites-available/chat-with-db /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
```

Update nginx config for your domain and SSL certificates.

## 📊 **Monitoring & Observability**

### **1. Application Metrics**

Install Prometheus client:
```bash
pip install prometheus-client
```

Add metrics endpoint to backend:
```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### **2. Logging Configuration**

Create `/etc/logrotate.d/chat-with-db`:
```
/var/log/chat-with-db/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload chat-with-db
    endscript
}
```

### **3. Health Checks**

```bash
# Backend health
curl -f http://localhost:8000/api/v1/health

# Nginx health
curl -f http://localhost/health

# Database connectivity
python3 -c "import sqlite3; sqlite3.connect('/path/to/db').close()"
```

## 🔒 **Security Hardening**

### **1. Firewall Configuration**

```bash
# UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### **2. Application Security**

```python
# Security headers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])
app.add_middleware(CORSMiddleware, allow_origins=["https://yourdomain.com"])
```

### **3. Rate Limiting**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, query: str):
    # Your implementation
    pass
```

## 📈 **Scaling & Performance**

### **1. Load Balancing**

Multiple backend instances:
```bash
# Start multiple workers
gunicorn backend.main:app -w 8 -k uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 --max-requests 1000 --max-requests-jitter 100
```

### **2. Caching Strategy**

Redis caching:
```python
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(ttl=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### **3. Database Optimization**

```sql
-- Add indexes for common queries
CREATE INDEX idx_fact_dates ON FactAllIndiaDailySummary(DateID);
CREATE INDEX idx_fact_regions ON FactAllIndiaDailySummary(RegionID);
CREATE INDEX idx_fact_states ON FactStateDailyEnergy(StateID);

-- Analyze table statistics
ANALYZE;
```

## 🚨 **Backup & Recovery**

### **1. Database Backup**

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_PATH="/path/to/power_data.db"

# Create backup
sqlite3 "$DB_PATH" ".backup '$BACKUP_DIR/backup_$DATE.db'"

# Compress backup
gzip "$BACKUP_DIR/backup_$DATE.db"

# Keep only last 7 days
find "$BACKUP_DIR" -name "backup_*.db.gz" -mtime +7 -delete
```

### **2. Application Backup**

```bash
# Backup application code
tar -czf "/backups/app_$(date +%Y%m%d).tar.gz" /opt/chat-with-db

# Backup configuration
cp /etc/nginx/sites-available/chat-with-db /backups/nginx_config_$(date +%Y%m%d)
cp /etc/systemd/system/chat-with-db.service /backups/service_$(date +%Y%m%d)
```

## 🔄 **CI/CD Pipeline**

### **1. GitHub Actions**

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to server
        uses: appleboy/ssh-action@v0.1.5
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.KEY }}
          script: |
            cd /opt/chat-with-db
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt
            cd agent-ui && npm install && npm run build
            cd ..
            sudo systemctl restart chat-with-db
            sudo systemctl reload nginx
```

## 📋 **Deployment Checklist**

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring setup complete
- [ ] Backup strategy implemented
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Logging configured
- [ ] Health checks working
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Team trained on deployment

## 🆘 **Troubleshooting**

### **Common Issues**

1. **High Memory Usage**
   - Check for memory leaks in Python processes
   - Reduce worker count
   - Monitor with `htop` or `top`

2. **Slow Response Times**
   - Check database performance
   - Enable query logging
   - Monitor network latency

3. **Service Won't Start**
   - Check logs: `journalctl -u chat-with-db`
   - Verify file permissions
   - Check port availability

### **Emergency Procedures**

```bash
# Quick service restart
sudo systemctl restart chat-with-db

# Rollback to previous version
cd /opt/chat-with-db
git checkout HEAD~1
sudo systemctl restart chat-with-db

# Emergency maintenance mode
echo "System under maintenance" | sudo tee /usr/share/nginx/html/maintenance.html
sudo systemctl stop chat-with-db
```

---

**Status**: ✅ **Production Ready** - Follow this guide for enterprise-grade deployment.

**Last Updated**: August 2025
