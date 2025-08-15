# Chat with DB - Cloud Deployment Guide

This guide provides step-by-step instructions for deploying the Chat with DB application to Google Cloud Platform with SQL Server support.

## 🚀 Quick Start

### Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud CLI** (gcloud) installed and configured
3. **Docker** installed locally
4. **Git** repository cloned locally

### 1. Initial Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Chat-with-DB-Deployment

# Set your Google Cloud project
gcloud config set project YOUR_PROJECT_ID

# Make deployment script executable
chmod +x deploy-gcp.sh
```

### 2. Environment Configuration

Copy the environment template and configure it:

```bash
# Copy the template
cp env.template .env

# Edit .env with your values
nano .env
```

**Required Environment Variables:**

```bash
# Database Configuration
DATABASE_TYPE=mssql
MSSQL_SERVER=your-server-name
MSSQL_DATABASE=your-database-name
MSSQL_USERNAME=your-username
MSSQL_PASSWORD=your-password

# Frontend Configuration
FRONTEND_ORIGIN=https://your-domain.com

# LLM Configuration
LLM_PROVIDER_TYPE=openai  # or vertex, anthropic, ollama
LLM_API_KEY=your-api-key
```

### 3. Automated Deployment

Run the automated deployment script:

```bash
./deploy-gcp.sh
```

The script will:
- ✅ Check prerequisites
- ✅ Enable required Google Cloud APIs
- ✅ Create Artifact Registry repositories
- ✅ Set up secrets in Secret Manager
- ✅ Configure Cloud Build permissions
- ✅ Deploy backend and frontend to Cloud Run
- ✅ Display service URLs

## 🔧 Manual Deployment Steps

If you prefer manual deployment or need to troubleshoot:

### Step 1: Enable Google Cloud APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudresourcemanager.googleapis.com
```

### Step 2: Create Artifact Registry

```bash
# Backend repository
gcloud artifacts repositories create chat-with-db-backend \
  --repository-format=docker \
  --location=us-central1 \
  --description="Chat with DB Backend Images"

# Frontend repository
gcloud artifacts repositories create chat-with-db-frontend \
  --repository-format=docker \
  --location=us-central1 \
  --description="Chat with DB Frontend Images"
```

### Step 3: Set Up Secrets

```bash
# MSSQL Password
echo "your-mssql-password" | gcloud secrets create MSSQL_PASSWORD --data-file=-

# LLM API Key (if using OpenAI/Anthropic)
echo "your-llm-api-key" | gcloud secrets create LLM_API_KEY --data-file=-

# Vector DB API Key (if using Qdrant Cloud)
echo "your-vector-db-api-key" | gcloud secrets create VECTOR_DB_API_KEY --data-file=-
```

### Step 4: Configure Cloud Build Permissions

```bash
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
```

### Step 5: Deploy with Cloud Build

```bash
gcloud builds submit --config=cloudbuild.yaml
```

## 🐳 Local Development with Docker

### Using Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.cloud.yml up -d

# View logs
docker-compose -f docker-compose.cloud.yml logs -f

# Stop services
docker-compose -f docker-compose.cloud.yml down
```

### Individual Services

```bash
# Build and run backend
docker build -t chat-with-db-backend .
docker run -p 8000:8000 --env-file .env chat-with-db-backend

# Build and run frontend
cd agent-ui
docker build -t chat-with-db-frontend .
docker run -p 3000:80 chat-with-db-frontend
```

## 🔍 Monitoring and Troubleshooting

### Health Checks

- **Backend**: `https://your-backend-url/api/v1/health`
- **Frontend**: `https://your-frontend-url/health`

### View Logs

```bash
# Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision" --limit=50

# Specific service logs
gcloud logging read "resource.labels.service_name=chat-with-db-backend" --limit=50
```

### Common Issues

#### 1. Database Connection Failed

**Symptoms**: Backend health check fails, database errors in logs

**Solutions**:
- Verify MSSQL credentials in Secret Manager
- Check Azure SQL firewall rules
- Ensure ODBC Driver 18 is installed in container

#### 2. Frontend Can't Connect to Backend

**Symptoms**: Frontend loads but API calls fail

**Solutions**:
- Verify `REACT_APP_API_BASE` environment variable
- Check CORS configuration
- Ensure backend service is running

#### 3. Build Failures

**Symptoms**: Cloud Build fails during Docker build

**Solutions**:
- Check Dockerfile syntax
- Verify all files are committed to repository
- Check resource limits in cloudbuild.yaml

## 🔐 Security Best Practices

### 1. Secrets Management

- ✅ Use Secret Manager for all sensitive data
- ✅ Never commit secrets to version control
- ✅ Rotate secrets regularly
- ✅ Use least privilege access

### 2. Network Security

- ✅ Use HTTPS for all external communication
- ✅ Configure CORS to restrict origins
- ✅ Use VPC connectors for private services
- ✅ Implement rate limiting

### 3. Container Security

- ✅ Use non-root users in containers
- ✅ Keep base images updated
- ✅ Scan images for vulnerabilities
- ✅ Use multi-stage builds

## 📊 Performance Optimization

### Backend (Cloud Run)

- **Memory**: Start with 2GB, monitor usage
- **CPU**: 1 vCPU for most workloads
- **Concurrency**: 80-200 depending on workload
- **Instances**: 0-10 for cost optimization

### Frontend (Cloud Run)

- **Memory**: 512MB sufficient for React app
- **CPU**: 0.5 vCPU for static serving
- **Instances**: 0-5 for cost optimization

### Database (Azure SQL)

- **Tier**: Start with Basic/Standard
- **DTUs**: Monitor usage and scale as needed
- **Connection Pooling**: Enable in application
- **Indexes**: Optimize for query patterns

## 💰 Cost Optimization

### Estimated Monthly Costs (US)

- **Cloud Run Backend**: $20-60 (depending on traffic)
- **Cloud Run Frontend**: $5-15 (depending on traffic)
- **Artifact Registry**: $1-5 (depending on storage)
- **Secret Manager**: $0.06 per secret per month
- **Cloud Build**: $0.003 per build-minute
- **Logging**: $0.50 per GB ingested

### Cost Reduction Tips

1. **Use min-instances=0** for non-critical services
2. **Set memory limits** based on actual usage
3. **Use regional resources** to avoid egress costs
4. **Monitor and alert** on spending
5. **Use committed use discounts** for predictable workloads

## 🚀 Scaling and High Availability

### Auto-scaling

- **Backend**: 0-10 instances based on demand
- **Frontend**: 0-5 instances for redundancy
- **Database**: Azure SQL auto-scaling

### Load Balancing

- **Cloud Run** provides built-in load balancing
- **Global load balancing** available with Cloud CDN
- **Custom domains** with SSL certificates

### Disaster Recovery

- **Multi-region deployment** for critical applications
- **Database backups** with Azure SQL
- **Cross-region replication** for static assets

## 📝 Maintenance and Updates

### Regular Tasks

1. **Security updates**: Monthly
2. **Dependency updates**: Quarterly
3. **Performance monitoring**: Weekly
4. **Cost review**: Monthly
5. **Backup verification**: Monthly

### Update Process

```bash
# 1. Update code
git pull origin main

# 2. Test locally
docker-compose -f docker-compose.cloud.yml up --build

# 3. Deploy to staging (if available)
gcloud builds submit --config=cloudbuild.yaml --substitutions=_ENV=staging

# 4. Deploy to production
gcloud builds submit --config=cloudbuild.yaml --substitutions=_ENV=production
```

## 🆘 Support and Resources

### Documentation

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Azure SQL Documentation](https://docs.microsoft.com/azure/azure-sql/)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)

### Community

- [Google Cloud Community](https://cloud.google.com/community)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-platform)
- [GitHub Issues](https://github.com/your-repo/issues)

### Support Tiers

- **Community Support**: Free
- **Google Cloud Support**: $100+/month
- **Azure Support**: $29+/month

---

## 🎯 Next Steps

1. **Deploy to staging environment** for testing
2. **Set up monitoring and alerting**
3. **Configure custom domains and SSL**
4. **Implement CI/CD pipeline**
5. **Add performance monitoring**
6. **Set up backup and recovery procedures**

For questions or issues, please refer to the troubleshooting section or create an issue in the repository.
