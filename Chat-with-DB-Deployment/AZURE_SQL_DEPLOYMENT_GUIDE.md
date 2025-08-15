# Azure SQL Server Deployment Guide

This guide provides comprehensive instructions for deploying the Chat with DB application with Azure SQL Server integration.

## 🚀 Quick Start

### Prerequisites

1. **Azure SQL Database** instance created and running
2. **Google Cloud Account** with billing enabled
3. **Google Cloud CLI** (gcloud) installed and configured
4. **Docker** installed locally
5. **Python 3.11+** with required dependencies

### 1. Azure SQL Database Setup

#### Create Azure SQL Database

1. **Azure Portal Setup**:
   ```bash
   # Go to Azure Portal: https://portal.azure.com
   # Create SQL Database with these settings:
   - Server: powerflow-server (or your preferred name)
   - Database: Powerflow
   - Pricing tier: Basic (for development) or Standard (for production)
   - Authentication: SQL authentication
   ```

2. **Configure Firewall Rules**:
   ```bash
   # Allow your IP address
   # Allow Azure services (for Cloud Run access)
   # Note: Cloud Run IPs are dynamic, consider using VPC connector
   ```

3. **Create Database User**:
   ```sql
   -- Connect to your Azure SQL Database
   CREATE USER app_user WITH PASSWORD = 'your_secure_password_here';
   GRANT CONNECT TO app_user;
   GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA::dbo TO app_user;
   ```

### 2. Environment Configuration

#### Copy and Configure Environment File

```bash
# Copy the template
cp env.template .env

# Edit .env with your Azure SQL credentials
nano .env
```

#### Required Azure SQL Environment Variables

```bash
# Database Configuration
DATABASE_TYPE=mssql
MSSQL_SERVER=powerflow-server
MSSQL_DATABASE=Powerflow
MSSQL_USERNAME=app_user
MSSQL_PASSWORD=your_secure_password_here

# Azure SQL specific settings
MSSQL_ENCRYPT=true
MSSQL_TRUST_SERVER_CERTIFICATE=false
MSSQL_CONNECTION_TIMEOUT=30
MSSQL_COMMAND_TIMEOUT=30

# Frontend Configuration (update with your domains)
FRONTEND_ORIGIN=https://your-frontend-domain.com
CORS_ORIGINS=https://your-frontend-domain.com

# LLM Configuration (choose your provider)
LLM_PROVIDER_TYPE=openai
LLM_API_KEY=your-openai-api-key
LLM_MODEL=gpt-4
```

### 3. Test Azure SQL Connection Locally

#### Run Connection Test

```bash
# Test Azure SQL connection before deployment
python test_azure_connection.py
```

**Expected Output**:
```
🚀 Azure SQL Connection Test
============================================================
🔍 Checking environment variables...
============================================================
   ✅ DATABASE_TYPE: mssql
   ✅ MSSQL_SERVER: powerflow-server
   ✅ MSSQL_DATABASE: Powerflow
   ✅ MSSQL_USERNAME: app_user
   ✅ MSSQL_PASSWORD: ********

============================================================
✅ All required environment variables are set.
🔍 Checking ODBC Driver availability...
============================================================
   ✅ Found ODBC Driver 18: ODBC Driver 18 for SQL Server

🔍 Testing Azure SQL connection...
============================================================
1. Testing basic database connection...
   ✅ Basic connection: SUCCESS

2. Getting database health information...
   ✅ Database type: mssql
   ✅ Connection status: healthy
   ✅ Is Azure SQL: True
   ✅ Azure server: powerflow-server
   ✅ Azure database: Powerflow

3. Testing Azure SQL specific features...
   ✅ Azure connection: SUCCESS
   ✅ Server: powerflow-server
   ✅ Database: Powerflow
   ✅ Engine Edition: 5
   ✅ Is Azure SQL: True

============================================================
🎉 All tests passed! Azure SQL connection is working correctly.
```

### 4. Deploy to Google Cloud

#### Automated Deployment

```bash
# Make script executable
chmod +x deploy-gcp.sh

# Run deployment
./deploy-gcp.sh
```

#### Manual Deployment Steps

If you prefer manual deployment:

```bash
# 1. Set project
gcloud config set project YOUR_PROJECT_ID

# 2. Enable APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com

# 3. Create secrets
echo "your_azure_sql_password" | gcloud secrets create MSSQL_PASSWORD --data-file=-

# 4. Deploy with Cloud Build
gcloud builds submit --config=cloudbuild.yaml
```

## 🔧 Azure SQL Specific Features

### 1. Enhanced Health Checks

The application now includes Azure SQL specific health monitoring:

```bash
# Basic health check
curl https://your-backend-url/api/v1/health

# Azure SQL specific status
curl https://your-backend-url/api/v1/azure-sql/status

# Table information
curl https://your-backend-url/api/v1/azure-sql/tables/your_table_name
```

### 2. Azure SQL Performance Optimizations

The application automatically applies Azure SQL optimizations:

- **Connection Pooling**: Optimized for Azure SQL with 20 connections
- **Session Settings**: ANSI compliance and performance settings
- **MARS Support**: Multiple Active Result Sets enabled
- **Connection Recycling**: 30-minute connection recycling
- **Command Timeouts**: Configurable query timeouts

### 3. Azure SQL Monitoring

Monitor your Azure SQL performance:

```bash
# Get performance metrics
curl -X GET "https://your-backend-url/api/v1/azure-sql/status"

# Execute custom queries (read-only)
curl -X POST "https://your-backend-url/api/v1/azure-sql/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT COUNT(*) FROM your_table", "timeout": 30}'
```

## 🔐 Security Best Practices

### 1. Azure SQL Security

- **Encryption**: Always enabled (Encrypt=yes)
- **Certificate Validation**: TrustServerCertificate=no
- **Firewall Rules**: Restrict to necessary IPs
- **User Permissions**: Least privilege access
- **Connection Timeouts**: Configurable timeouts

### 2. Google Cloud Security

- **Secret Manager**: Store all credentials securely
- **IAM Roles**: Least privilege access
- **VPC Connector**: Consider for private connectivity
- **HTTPS**: All external communication encrypted

### 3. Application Security

- **SQL Injection Protection**: Parameterized queries
- **Input Validation**: Query safety checks
- **CORS Configuration**: Restrict to your domains
- **Rate Limiting**: Built-in protection

## 📊 Performance Optimization

### 1. Azure SQL Configuration

```sql
-- Recommended Azure SQL settings
ALTER DATABASE Powerflow SET COMPATIBILITY_LEVEL = 150;
ALTER DATABASE Powerflow SET AUTO_CLOSE OFF;
ALTER DATABASE Powerflow SET AUTO_SHRINK OFF;
```

### 2. Connection Pooling

```python
# Application automatically configures:
- Pool size: 20 connections
- Max overflow: 20 connections
- Connection recycling: 30 minutes
- Pre-ping: Enabled
- Timeout: 30 seconds
```

### 3. Query Optimization

```python
# Use Azure SQL utilities for optimized queries
from backend.core.azure_sql_utils import execute_azure_query

result = execute_azure_query(
    "SELECT * FROM your_table WHERE id = :id",
    {"id": 123},
    timeout=30
)
```

## 🚨 Troubleshooting

### 1. Connection Issues

**Problem**: Cannot connect to Azure SQL
```bash
# Check firewall rules
# Verify credentials in Secret Manager
# Test connection locally first
python test_azure_connection.py
```

**Problem**: ODBC Driver not found
```bash
# Install ODBC Driver 18
# Verify in Docker container
docker run --rm your-image odbcinst -q -d
```

### 2. Performance Issues

**Problem**: Slow queries
```bash
# Check connection pool usage
curl "https://your-backend-url/api/v1/azure-sql/status"

# Monitor Azure SQL metrics in Azure Portal
# Check for connection timeouts
```

**Problem**: Connection timeouts
```bash
# Increase timeout values in .env
MSSQL_CONNECTION_TIMEOUT=60
MSSQL_COMMAND_TIMEOUT=60
```

### 3. Deployment Issues

**Problem**: Build fails
```bash
# Check Docker build logs
# Verify all files committed
# Check resource limits in cloudbuild.yaml
```

**Problem**: Service not responding
```bash
# Check Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision" --limit=50

# Verify environment variables
# Check Azure SQL connectivity
```

## 📈 Monitoring and Alerting

### 1. Azure SQL Monitoring

- **Azure Portal**: Database metrics and performance
- **Query Performance Insights**: Identify slow queries
- **Connection Monitoring**: Track active connections
- **Resource Usage**: CPU, memory, and storage

### 2. Application Monitoring

- **Health Checks**: Regular endpoint monitoring
- **Performance Metrics**: Response times and throughput
- **Error Tracking**: Failed queries and connections
- **Resource Usage**: Memory and CPU consumption

### 3. Cloud Run Monitoring

- **Request Metrics**: Latency and throughput
- **Instance Scaling**: Auto-scaling behavior
- **Error Rates**: 4xx and 5xx responses
- **Resource Utilization**: Memory and CPU usage

## 🔄 Maintenance and Updates

### 1. Regular Tasks

- **Security Updates**: Monthly dependency updates
- **Performance Monitoring**: Weekly performance reviews
- **Backup Verification**: Monthly backup tests
- **Cost Optimization**: Monthly cost reviews

### 2. Update Process

```bash
# 1. Update code
git pull origin main

# 2. Test locally
python test_azure_connection.py

# 3. Deploy to staging (if available)
gcloud builds submit --config=cloudbuild.yaml --substitutions=_ENV=staging

# 4. Deploy to production
gcloud builds submit --config=cloudbuild.yaml --substitutions=_ENV=production
```

### 3. Backup and Recovery

- **Azure SQL Backups**: Automatic backups enabled
- **Point-in-Time Recovery**: Available for all tiers
- **Geo-Replication**: Consider for production
- **Application Data**: Regular exports and testing

## 💰 Cost Optimization

### 1. Azure SQL Costs

- **Basic Tier**: $5/month (5 DTUs, 2GB storage)
- **Standard Tier**: $30/month (10 DTUs, 250GB storage)
- **Premium Tier**: $465/month (125 DTUs, 500GB storage)

### 2. Google Cloud Costs

- **Cloud Run**: Pay-per-use ($0.00002400 per 100ms)
- **Artifact Registry**: $0.10 per GB per month
- **Secret Manager**: $0.06 per secret per month
- **Cloud Build**: $0.003 per build-minute

### 3. Cost Reduction Tips

1. **Use Basic Tier** for development
2. **Scale down** during off-hours
3. **Monitor usage** and optimize queries
4. **Use reserved instances** for predictable workloads

## 🆘 Support Resources

### 1. Documentation

- [Azure SQL Documentation](https://docs.microsoft.com/azure/azure-sql/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [ODBC Driver Documentation](https://docs.microsoft.com/sql/connect/odbc/)

### 2. Community

- [Azure SQL Community](https://techcommunity.microsoft.com/t5/azure-sql-database/bd-p/AzureSQLDatabase)
- [Google Cloud Community](https://cloud.google.com/community)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/azure-sql)

### 3. Support Tiers

- **Community Support**: Free
- **Azure Support**: $29+/month
- **Google Cloud Support**: $100+/month

---

## 🎯 Next Steps

1. **Test locally** with `python test_azure_connection.py`
2. **Deploy to staging** for testing
3. **Monitor performance** and optimize
4. **Set up alerting** for critical issues
5. **Plan backup and recovery** procedures

For questions or issues, please refer to the troubleshooting section or create an issue in the repository.
