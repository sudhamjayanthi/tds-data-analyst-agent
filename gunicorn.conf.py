import os

# Gunicorn configuration for production deployment
bind = f"0.0.0.0:{os.getenv('PORT', 8000)}"
workers = 1  # Free tier usually has memory limits
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 180  # 3 minutes for data analysis tasks
keepalive = 2
max_requests = 1000
max_requests_jitter = 50
preload_app = True
