import logging.config
import os
import logging

bind = "0.0.0.0:8080"  # Cloud Run listens on port 8080
workers = 1            # Single worker per container (concurrency=1)
worker_class = "sync"  # Synchronous worker since concurrency=1
timeout = 60           # Match Cloud Run request timeout (60s)
keepalive = 30         # Keep connections alive longer to reduce overhead
capture_output = True
enable_stdio_inheritance = True
accesslog = "-"
errorlog = "-"
loglevel = "info"      # Use 'info' to reduce excessive logging
preload_app = True     # Preload the model at startup for faster requests


# Initialize logging early
logger = logging.getLogger('gunicorn.error')

def on_starting(server):
    """Log when Gunicorn is starting."""
    if os.getenv('K_SERVICE'):
        import google.cloud.logging
        client = google.cloud.logging.Client()
        client.setup_logging(log_level=logging.DEBUG, name=os.getenv('K_SERVICE'))
        logger.info("Cloud Logging initialized for Gunicorn")
    logger.info("Gunicorn is starting")

def when_ready(server):
    """Log when Gunicorn is ready."""
    logger.info("Gunicorn is ready. Initializing application...")

def post_fork(server, worker):
    """Initialize Firebase after worker fork."""
    logger.info(f"Initializing worker {worker.pid}")
    from api.config.firebase_config import initialize_firebase
    try:
        initialize_firebase()
        logger.info(f"Worker {worker.pid} initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize worker {worker.pid}: {e}")
        raise

def pre_request(worker, req):
    worker.log.debug(f"Handling request: {req.uri}")

def post_request(worker, req, environ, resp):
    worker.log.debug(f"Completed request: {req.uri} - Status: {resp.status}")
