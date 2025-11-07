# F1 Analytics Backend - Deployment Guide

Complete guide for deploying the Python FastAPI backend with automated data synchronization.

## Table of Contents

1. [Quick Start (Local)](#quick-start-local)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Monitoring & Logs](#monitoring--logs)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- pip
- Supabase account with project created

### Steps

1. **Clone and navigate to backend**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Supabase credentials
   ```

5. **Run the server**
   ```bash
   python main.py
   ```

6. **Wait for initialization**
   - Server starts on http://localhost:8000
   - Automated agent checks database
   - If empty, populates 2023-2024 F1 data (5-10 minutes)
   - Background scheduler starts
   - API ready at http://localhost:8000/docs

---

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Create .env file**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Build and run**
   ```bash
   docker-compose up -d
   ```

3. **Check logs**
   ```bash
   docker-compose logs -f backend
   ```

4. **Access API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

5. **Stop services**
   ```bash
   docker-compose down
   ```

### Development Mode with Hot Reload

```bash
docker-compose -f docker-compose.dev.yml up
```

This mounts your source code and enables auto-reload on changes.

### Using Docker Directly

1. **Build image**
   ```bash
   docker build -t f1-analytics-backend .
   ```

2. **Run container**
   ```bash
   docker run -d \
     --name f1-analytics-backend \
     -p 8000:8000 \
     --env-file .env \
     f1-analytics-backend
   ```

3. **View logs**
   ```bash
   docker logs -f f1-analytics-backend
   ```

---

## Cloud Deployment

### Railway

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login**
   ```bash
   railway login
   ```

3. **Initialize project**
   ```bash
   railway init
   ```

4. **Set environment variables**
   ```bash
   railway variables set SUPABASE_URL=your-url
   railway variables set SUPABASE_KEY=your-key
   railway variables set SUPABASE_SERVICE_KEY=your-service-key
   ```

5. **Deploy**
   ```bash
   railway up
   ```

6. **Get URL**
   ```bash
   railway domain
   ```

### Render

1. **Create account** at https://render.com

2. **New Web Service**
   - Connect your GitHub repository
   - Select `backend` directory
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python main.py`

3. **Environment Variables**
   - Add all variables from `.env.example`
   - Set your Supabase credentials

4. **Deploy**
   - Render will automatically deploy
   - Monitor logs in dashboard

### Fly.io

1. **Install Fly CLI**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**
   ```bash
   fly auth login
   ```

3. **Launch app**
   ```bash
   cd backend
   fly launch
   ```

4. **Set secrets**
   ```bash
   fly secrets set SUPABASE_URL=your-url
   fly secrets set SUPABASE_KEY=your-key
   fly secrets set SUPABASE_SERVICE_KEY=your-service-key
   ```

5. **Deploy**
   ```bash
   fly deploy
   ```

### AWS (EC2)

1. **Launch EC2 instance**
   - Ubuntu 22.04 LTS
   - t3.medium or larger
   - Open port 8000

2. **SSH into instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Docker**
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose
   sudo usermod -aG docker ubuntu
   ```

4. **Clone repository**
   ```bash
   git clone your-repo-url
   cd your-repo/backend
   ```

5. **Configure and run**
   ```bash
   cp .env.example .env
   nano .env  # Edit with your credentials
   docker-compose up -d
   ```

6. **Setup reverse proxy (Nginx)**
   ```bash
   sudo apt install -y nginx
   sudo nano /etc/nginx/sites-available/f1-analytics
   ```

   Add configuration:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

   Enable and restart:
   ```bash
   sudo ln -s /etc/nginx/sites-available/f1-analytics /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

---

## Environment Configuration

### Required Variables

```bash
# Supabase (Required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
```

### Optional Variables

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=False

# FastF1
FASTF1_CACHE_DIR=./fastf1_cache
FASTF1_ENABLE_CACHE=True

# Data Sync
AUTO_SYNC_ON_STARTUP=True
SYNC_INTERVAL_HOURS=24
HISTORICAL_YEARS=[2023,2024]

# Models
MODEL_DIR=./models
MODEL_CACHE_DIR=./models/cache
```

### Getting Supabase Credentials

1. Go to https://supabase.com/dashboard
2. Select your project
3. Go to Settings > API
4. Copy:
   - **URL**: Project URL
   - **anon/public key**: For SUPABASE_KEY
   - **service_role key**: For SUPABASE_SERVICE_KEY (keep secret!)

---

## Monitoring & Logs

### View Logs (Docker)

```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f backend
```

### View Logs (Local)

```bash
# Application logs
tail -f logs/f1_analytics.log

# Real-time monitoring
watch -n 1 'tail -20 logs/f1_analytics.log'
```

### Health Check

```bash
# Check if service is healthy
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"1.0.0","service":"F1 Analytics API"}
```

### Monitor Data Sync

```bash
# Check if database has data
curl http://localhost:8000/api/v1/drivers | jq 'length'

# Check races
curl http://localhost:8000/api/v1/races | jq 'length'

# Check if model is trained
curl http://localhost:8000/api/v1/predictions/train
```

---

## Troubleshooting

### Database Not Populating

**Problem**: Server starts but database remains empty

**Solution**:
1. Check logs for errors:
   ```bash
   docker-compose logs backend | grep ERROR
   ```

2. Verify Supabase credentials:
   ```bash
   echo $SUPABASE_URL
   echo $SUPABASE_KEY
   ```

3. Manually trigger data sync:
   ```bash
   curl -X POST http://localhost:8000/api/v1/sync/initialize
   ```

4. Check Supabase connection:
   ```bash
   curl http://localhost:8000/health
   ```

### FastF1 Cache Issues

**Problem**: FastF1 fails to fetch data

**Solution**:
1. Clear cache:
   ```bash
   rm -rf fastf1_cache/
   docker-compose restart backend
   ```

2. Check FastF1 service status:
   ```bash
   curl https://livetiming.formula1.com/static/StreamingStatus.json
   ```

3. Increase timeout in config.py

### Memory Issues

**Problem**: Container crashes with OOM

**Solution**:
1. Increase Docker memory limit:
   ```bash
   docker-compose down
   # Edit docker-compose.yml, add:
   # services:
   #   backend:
   #     mem_limit: 2g
   docker-compose up -d
   ```

2. Reduce historical years:
   ```bash
   # In .env, change:
   HISTORICAL_YEARS=[2024]
   ```

### Port Already in Use

**Problem**: Port 8000 is already in use

**Solution**:
1. Find process using port:
   ```bash
   lsof -i :8000
   ```

2. Kill process:
   ```bash
   kill -9 <PID>
   ```

3. Or change port in .env:
   ```bash
   PORT=8001
   ```

### Model Training Fails

**Problem**: ML model training errors

**Solution**:
1. Ensure database has race results:
   ```bash
   curl http://localhost:8000/api/v1/races/1/results
   ```

2. Manually train model:
   ```bash
   curl -X POST http://localhost:8000/api/v1/predictions/train
   ```

3. Check model directory permissions:
   ```bash
   ls -la models/
   ```

### Slow Initial Load

**Problem**: First startup takes very long

**Solution**:
- This is normal! FastF1 fetches 2 years of data
- Expected time: 5-10 minutes
- Monitor progress in logs:
  ```bash
  docker-compose logs -f backend | grep "Processing"
  ```

---

## Performance Optimization

### Production Settings

```bash
# .env for production
DEBUG=False
AUTO_SYNC_ON_STARTUP=True
SYNC_INTERVAL_HOURS=24
FASTF1_ENABLE_CACHE=True
```

### Scaling

For high traffic, use multiple workers:

```bash
# In Dockerfile, change CMD to:
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

Add to requirements.txt:
```
gunicorn==21.2.0
```

### Database Optimization

1. Add indexes in Supabase:
   ```sql
   CREATE INDEX idx_races_date ON races(date);
   CREATE INDEX idx_race_results_race_id ON race_results(race_id);
   CREATE INDEX idx_race_results_driver_id ON race_results(driver_id);
   ```

2. Enable connection pooling in config.py

---

## Security Best Practices

1. **Never commit .env file**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use secrets management**
   - Railway: Use Railway secrets
   - AWS: Use AWS Secrets Manager
   - Docker: Use Docker secrets

3. **Enable HTTPS**
   - Use Nginx with Let's Encrypt
   - Or use cloud provider's SSL

4. **Restrict CORS origins**
   ```python
   # In config.py
   CORS_ORIGINS = ["https://yourdomain.com"]
   ```

5. **Rate limiting**
   - Add rate limiting middleware
   - Use cloud provider's WAF

---

## Backup & Recovery

### Backup FastF1 Cache

```bash
docker-compose exec backend tar -czf /tmp/fastf1_cache.tar.gz fastf1_cache/
docker cp f1-analytics-backend:/tmp/fastf1_cache.tar.gz ./backup/
```

### Backup ML Models

```bash
docker-compose exec backend tar -czf /tmp/models.tar.gz models/
docker cp f1-analytics-backend:/tmp/models.tar.gz ./backup/
```

### Restore

```bash
docker cp ./backup/fastf1_cache.tar.gz f1-analytics-backend:/tmp/
docker-compose exec backend tar -xzf /tmp/fastf1_cache.tar.gz
docker-compose restart backend
```

---

## Support

For issues:
- Check logs first
- Review API docs: http://localhost:8000/docs
- FastF1 docs: https://docs.fastf1.dev/
- Supabase docs: https://supabase.com/docs

## License

MIT