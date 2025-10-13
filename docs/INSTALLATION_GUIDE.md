# 🏎️ F1 Tire Thermal Platform

## Installation & Deployment Guide

**Version**: 2.0  
**Last Updated**: September 2025  
**Target Audience**: System Administrators, DevOps Engineers, F1 IT Teams

---

## 📋 **Table of Contents**

1. [System Requirements](#system-requirements)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration Management](#configuration-management)
7. [Monitoring & Logging](#monitoring--logging)
8. [Security Configuration](#security-configuration)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## 🖥️ **System Requirements**

### **Minimum Requirements**

- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 20 GB SSD
- **Network**: 100 Mbps
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### **Recommended Requirements**

- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 100 GB NVMe SSD
- **Network**: 1 Gbps
- **OS**: Ubuntu 20.04 LTS, Windows Server 2019+

### **Software Dependencies**

- **Python**: 3.8 or higher
- **Node.js**: 14+ (for frontend tools)
- **Git**: Latest version
- **Docker**: 20.10+ (optional)
- **Kubernetes**: 1.20+ (optional)

---

## 🛠️ **Local Development Setup**

### **Prerequisites Installation**

#### **Ubuntu/Debian**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install -y python3.9 python3.9-pip python3.9-venv python3.9-dev
sudo apt install -y git curl wget build-essential

# Install Node.js (for frontend tools)
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt install -y nodejs

# Install additional dependencies
sudo apt install -y libhdf5-dev libhdf5-serial-dev
sudo apt install -y pkg-config libfreetype6-dev libpng-dev
```

#### **macOS**

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and development tools
brew install python@3.9 git curl wget
brew install node

# Install additional dependencies
brew install hdf5 freetype pkg-config
```

#### **Windows**

```powershell
# Install Chocolatey (if not already installed)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python and development tools
choco install python39 git curl wget nodejs

# Install additional dependencies
choco install visualstudio2019buildtools
```

### **Project Setup**

```bash
# Clone repository
git clone https://github.com/ac12644/f1-tire-temp-prototype.git
cd f1-tire-temp-prototype

# Create virtual environment
python3.9 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify installation
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
```

### **Development Server**

```bash
# Start development server
streamlit run src/app_streamlit.py --server.port 8501 --server.headless false

# Or with custom configuration
streamlit run src/app_streamlit.py \
  --server.port 8501 \
  --server.headless false \
  --server.runOnSave true \
  --browser.gatherUsageStats false
```

---

## 🚀 **Production Deployment**

### **System Preparation**

#### **Create Application User**

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash f1tire
sudo usermod -aG sudo f1tire

# Switch to application user
sudo su - f1tire
```

#### **Install Production Dependencies**

```bash
# Install Python and system dependencies
sudo apt update
sudo apt install -y python3.9 python3.9-pip python3.9-venv
sudo apt install -y nginx supervisor redis-server
sudo apt install -y postgresql postgresql-contrib
sudo apt install -y certbot python3-certbot-nginx
```

### **Application Deployment**

```bash
# Create application directory
mkdir -p /opt/f1-tire-system
cd /opt/f1-tire-system

# Clone repository
git clone https://github.com/ac12644/f1-tire-temp-prototype.git .

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set proper permissions
sudo chown -R f1tire:f1tire /opt/f1-tire-system
chmod +x scripts/*.sh
```

### **Configuration Files**

#### **Environment Configuration**

```bash
# Create environment file
cat > /opt/f1-tire-system/.env << EOF
# Application Configuration
F1_TIRE_ENV=production
F1_TIRE_DEBUG=false
F1_TIRE_LOG_LEVEL=INFO

# Database Configuration
F1_TIRE_DB_PATH=/opt/f1-tire-system/data/system.db
F1_TIRE_DB_BACKUP_PATH=/opt/f1-tire-system/backups

# Security Configuration
F1_TIRE_SECRET_KEY=your-secret-key-here
F1_TIRE_ALLOWED_HOSTS=your-domain.com,localhost

# Performance Configuration
F1_TIRE_MAX_WORKERS=4
F1_TIRE_MAX_CONNECTIONS=100
F1_TIRE_CACHE_SIZE=1000

# Monitoring Configuration
F1_TIRE_METRICS_ENABLED=true
F1_TIRE_HEALTH_CHECK_INTERVAL=30
EOF
```

#### **Supervisor Configuration**

```bash
# Create supervisor configuration
sudo cat > /etc/supervisor/conf.d/f1-tire-system.conf << EOF
[program:f1-tire-system]
command=/opt/f1-tire-system/venv/bin/streamlit run src/app_streamlit.py --server.port 8501 --server.headless true
directory=/opt/f1-tire-system
user=f1tire
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/f1-tire-system.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=PATH="/opt/f1-tire-system/venv/bin"
EOF

# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start f1-tire-system
```

### **Nginx Configuration**

```bash
# Create Nginx configuration
sudo cat > /etc/nginx/sites-available/f1-tire-system << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }

    # Static files
    location /static/ {
        alias /opt/f1-tire-system/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/f1-tire-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### **SSL Certificate**

```bash
# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Test certificate renewal
sudo certbot renew --dry-run
```

---

## 🐳 **Docker Deployment**

### **Dockerfile**

```dockerfile
# Multi-stage build for production
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libhdf5-serial-dev \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Create application user
RUN useradd -m -s /bin/bash f1tire

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directory
RUN mkdir -p /app/data && chown -R f1tire:f1tire /app

# Switch to application user
USER f1tire

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start application
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.port", "8501", "--server.headless", "true"]
```

### **Docker Compose**

```yaml
version: "3.8"

services:
  f1-tire-system:
    build: .
    ports:
      - "8501:8501"
    environment:
      - F1_TIRE_ENV=production
      - F1_TIRE_DB_PATH=/app/data/system.db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - f1-tire-system
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### **Docker Deployment Commands**

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f f1-tire-system

# Scale application
docker-compose up -d --scale f1-tire-system=3

# Update application
docker-compose pull
docker-compose up -d
```

---

## ☁️ **Cloud Deployment**

### **AWS Deployment**

#### **EC2 Instance Setup**

```bash
# Launch EC2 instance (Ubuntu 20.04 LTS)
# Instance type: t3.medium or larger
# Security group: Allow HTTP (80), HTTPS (443), SSH (22)

# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu

# Clone and deploy
git clone https://github.com/ac12644/f1-tire-temp-prototype.git
cd f1-tire-temp-prototype
docker-compose up -d
```

#### **Application Load Balancer**

```bash
# Create ALB target group
aws elbv2 create-target-group \
  --name f1-tire-targets \
  --protocol HTTP \
  --port 8501 \
  --vpc-id vpc-xxxxxxxx \
  --target-type instance

# Create ALB
aws elbv2 create-load-balancer \
  --name f1-tire-alb \
  --subnets subnet-xxxxxxxx subnet-yyyyyyyy \
  --security-groups sg-xxxxxxxx
```

### **Azure Deployment**

#### **Azure Container Instances**

```bash
# Create resource group
az group create --name f1-tire-rg --location eastus

# Deploy container instance
az container create \
  --resource-group f1-tire-rg \
  --name f1-tire-system \
  --image your-registry/f1-tire-system:latest \
  --ports 8501 \
  --environment-variables F1_TIRE_ENV=production
```

### **Google Cloud Deployment**

#### **Cloud Run**

```bash
# Build and push image
gcloud builds submit --tag gcr.io/your-project/f1-tire-system

# Deploy to Cloud Run
gcloud run deploy f1-tire-system \
  --image gcr.io/your-project/f1-tire-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## ⚙️ **Configuration Management**

### **Environment Variables**

```bash
# Application Configuration
F1_TIRE_ENV=production
F1_TIRE_DEBUG=false
F1_TIRE_LOG_LEVEL=INFO
F1_TIRE_SECRET_KEY=your-secret-key

# Database Configuration
F1_TIRE_DB_PATH=/app/data/system.db
F1_TIRE_DB_BACKUP_PATH=/app/backups
F1_TIRE_DB_RETENTION_DAYS=30

# Performance Configuration
F1_TIRE_MAX_WORKERS=4
F1_TIRE_MAX_CONNECTIONS=100
F1_TIRE_CACHE_SIZE=1000
F1_TIRE_CACHE_TTL=3600

# Security Configuration
F1_TIRE_ALLOWED_HOSTS=your-domain.com,localhost
F1_TIRE_CORS_ORIGINS=https://your-domain.com
F1_TIRE_RATE_LIMIT=1000

# Monitoring Configuration
F1_TIRE_METRICS_ENABLED=true
F1_TIRE_HEALTH_CHECK_INTERVAL=30
F1_TIRE_LOG_RETENTION_DAYS=7
```

### **Configuration Files**

#### **System Configuration**

```python
# src/config.py
import os
from dataclasses import dataclass

@dataclass
class SystemConfig:
    # Environment
    env: str = os.getenv('F1_TIRE_ENV', 'development')
    debug: bool = os.getenv('F1_TIRE_DEBUG', 'false').lower() == 'true'
    log_level: str = os.getenv('F1_TIRE_LOG_LEVEL', 'INFO')

    # Database
    db_path: str = os.getenv('F1_TIRE_DB_PATH', 'data/system.db')
    db_backup_path: str = os.getenv('F1_TIRE_DB_BACKUP_PATH', 'backups')
    db_retention_days: int = int(os.getenv('F1_TIRE_DB_RETENTION_DAYS', '30'))

    # Performance
    max_workers: int = int(os.getenv('F1_TIRE_MAX_WORKERS', '4'))
    max_connections: int = int(os.getenv('F1_TIRE_MAX_CONNECTIONS', '100'))
    cache_size: int = int(os.getenv('F1_TIRE_CACHE_SIZE', '1000'))
    cache_ttl: int = int(os.getenv('F1_TIRE_CACHE_TTL', '3600'))

    # Security
    secret_key: str = os.getenv('F1_TIRE_SECRET_KEY', 'default-secret-key')
    allowed_hosts: list = os.getenv('F1_TIRE_ALLOWED_HOSTS', 'localhost').split(',')
    cors_origins: list = os.getenv('F1_TIRE_CORS_ORIGINS', 'http://localhost:8501').split(',')
    rate_limit: int = int(os.getenv('F1_TIRE_RATE_LIMIT', '1000'))

    # Monitoring
    metrics_enabled: bool = os.getenv('F1_TIRE_METRICS_ENABLED', 'true').lower() == 'true'
    health_check_interval: int = int(os.getenv('F1_TIRE_HEALTH_CHECK_INTERVAL', '30'))
    log_retention_days: int = int(os.getenv('F1_TIRE_LOG_RETENTION_DAYS', '7'))

# Global configuration instance
system_config = SystemConfig()
```

---

## 📊 **Monitoring & Logging**

### **Application Monitoring**

#### **Health Check Endpoint**

```python
# Add to src/app_streamlit.py
import time
import psutil

def health_check():
    """Health check endpoint for monitoring"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'uptime': time.time() - start_time,
        'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
        'cpu_usage': psutil.cpu_percent(),
        'disk_usage': psutil.disk_usage('/').percent
    }

# Add health check route
if st.query_params.get('health') == 'check':
    st.json(health_check())
    st.stop()
```

#### **Prometheus Metrics**

```python
# Install prometheus client
pip install prometheus-client

# Add metrics collection
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
REQUEST_COUNT = Counter('f1_tire_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('f1_tire_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('f1_tire_active_connections', 'Active connections')
TEMPERATURE_READINGS = Gauge('f1_tire_temperature_celsius', 'Temperature readings', ['corner', 'node'])

# Start metrics server
start_http_server(8000)
```

### **Logging Configuration**

#### **Structured Logging**

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id

        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/f1-tire-system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### **Log Rotation**

```bash
# Configure logrotate
sudo cat > /etc/logrotate.d/f1-tire-system << EOF
/var/log/f1-tire-system.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 f1tire f1tire
    postrotate
        supervisorctl restart f1-tire-system
    endscript
}
EOF
```

---

## 🔒 **Security Configuration**

### **Application Security**

#### **Security Headers**

```python
# Add security headers
import streamlit as st

# Security headers
st.set_page_config(
    page_title="F1 Tire Management System",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add security middleware
def add_security_headers():
    st.markdown("""
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';">
    <meta http-equiv="X-Content-Type-Options" content="nosniff">
    <meta http-equiv="X-Frame-Options" content="DENY">
    <meta http-equiv="X-XSS-Protection" content="1; mode=block">
    """, unsafe_allow_html=True)
```

#### **Authentication (Optional)**

```python
# Simple authentication example
import streamlit_authenticator as stauth

# Configure authentication
authenticator = stauth.Authenticate(
    credentials={
        'usernames': {
            'admin': {
                'name': 'Administrator',
                'password': stauth.Hasher(['admin123']).generate()[0]
            }
        }
    },
    cookie_name='f1_tire_auth',
    key='f1_tire_secret_key',
    cookie_expiry_days=30
)

# Add authentication check
name, authentication_status, username = authenticator.login('Login', 'main')

if not authentication_status:
    st.error('Username/password is incorrect')
    st.stop()
```

### **Network Security**

#### **Firewall Configuration**

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8501/tcp  # Block direct access to Streamlit
```

#### **SSL/TLS Configuration**

```bash
# Strong SSL configuration in Nginx
sudo cat >> /etc/nginx/sites-available/f1-tire-system << EOF

# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# Security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options DENY always;
add_header X-Content-Type-Options nosniff always;
add_header X-XSS-Protection "1; mode=block" always;
EOF
```

---

## 💾 **Backup & Recovery**

### **Database Backup**

#### **Automated Backup Script**

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/opt/f1-tire-system/backups"
DB_PATH="/opt/f1-tire-system/data/system.db"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/f1_tire_backup_$DATE.db"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
sqlite3 $DB_PATH ".backup '$BACKUP_FILE'"

# Compress backup
gzip $BACKUP_FILE

# Remove old backups (keep last 30 days)
find $BACKUP_DIR -name "f1_tire_backup_*.db.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

#### **Cron Job Setup**

```bash
# Add to crontab
crontab -e

# Add backup job (daily at 2 AM)
0 2 * * * /opt/f1-tire-system/scripts/backup.sh >> /var/log/f1-tire-backup.log 2>&1
```

### **Configuration Backup**

```bash
#!/bin/bash
# scripts/config_backup.sh

CONFIG_DIR="/opt/f1-tire-system/config"
BACKUP_DIR="/opt/f1-tire-system/backups/config"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf "$BACKUP_DIR/config_backup_$DATE.tar.gz" -C $CONFIG_DIR .

# Keep last 7 days of config backups
find $BACKUP_DIR -name "config_backup_*.tar.gz" -mtime +7 -delete
```

### **Recovery Procedures**

#### **Database Recovery**

```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_FILE=$1
DB_PATH="/opt/f1-tire-system/data/system.db"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
sudo supervisorctl stop f1-tire-system

# Restore database
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | sqlite3 $DB_PATH
else
    sqlite3 $DB_PATH < $BACKUP_FILE
fi

# Start application
sudo supervisorctl start f1-tire-system

echo "Database restored from $BACKUP_FILE"
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Application Won't Start**

```bash
# Check logs
sudo supervisorctl status f1-tire-system
tail -f /var/log/f1-tire-system.log

# Check port availability
sudo netstat -tlnp | grep 8501

# Check Python environment
source /opt/f1-tire-system/venv/bin/activate
python -c "import streamlit; print('OK')"
```

#### **Performance Issues**

```bash
# Check system resources
htop
df -h
free -h

# Check application metrics
curl http://localhost:8501/health

# Monitor logs for errors
tail -f /var/log/f1-tire-system.log | grep ERROR
```

#### **Database Issues**

```bash
# Check database integrity
sqlite3 /opt/f1-tire-system/data/system.db "PRAGMA integrity_check;"

# Check database size
ls -lh /opt/f1-tire-system/data/system.db

# Repair database if needed
sqlite3 /opt/f1-tire-system/data/system.db ".recover" | sqlite3 recovered.db
```

### **Performance Optimization**

#### **System Tuning**

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.conf
sysctl -p
```

#### **Application Tuning**

```python
# Optimize Streamlit configuration
# .streamlit/config.toml
[server]
port = 8501
headless = true
runOnSave = false
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#e10600"
backgroundColor = "#1e1e2e"
secondaryBackgroundColor = "#2d2d44"
textColor = "#ffffff"
```

---

## 📞 **Support & Maintenance**

### **Regular Maintenance Tasks**

#### **Daily Tasks**

- Check application health and logs
- Monitor system resources
- Verify backup completion
- Check for security updates

#### **Weekly Tasks**

- Review performance metrics
- Clean up old log files
- Update system packages
- Test backup restoration

#### **Monthly Tasks**

- Security audit and updates
- Performance optimization review
- Capacity planning assessment
- Disaster recovery testing

### **Support Contacts**

- **Technical Issues**: GitHub Issues
- **Documentation**: Check `/docs` directory
- **Community**: GitHub Discussions
- **Professional Support**: Contact via GitHub profile

---

**Installation Guide Version**: 2.0  
**Last Updated**: September 2025  
**Next Review**: September 2026

_This installation guide provides comprehensive instructions for deploying the F1 Tire Thermal Platform in various environments._
