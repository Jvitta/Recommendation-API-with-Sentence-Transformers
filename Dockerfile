# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies required for numpy and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV LOG_LEVEL=INFO
ENV GOOGLE_CLOUD_PROJECT=astn-mvp-v1
ENV HF_HOME=/app/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory and set permissions
RUN mkdir -p /app/hf_cache && \
    chmod -R 777 /app/hf_cache

# Copy the rest of the application
COPY . .

# Create a logging configuration file
RUN echo '{\
    "version": 1,\
    "disable_existing_loggers": false,\
    "formatters": {\
        "standard": {\
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"\
        }\
    },\
    "handlers": {\
        "console": {\
            "class": "logging.StreamHandler",\
            "formatter": "standard",\
            "stream": "ext://sys.stdout"\
        }\
    },\
    "loggers": {\
        "": {\
            "handlers": ["console"],\
            "level": "INFO"\
        },\
        "gunicorn.error": {\
            "handlers": ["console"],\
            "level": "INFO",\
            "propagate": true\
        },\
        "gunicorn.access": {\
            "handlers": ["console"],\
            "level": "INFO",\
            "propagate": true\
        }\
    }\
}' > /app/logging.json

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--worker-class", "sync", \
     "--timeout", "60", \
     "--keep-alive", "30", \
     "--max-requests", "100", \
     "--max-requests-jitter", "10", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "--enable-stdio-inheritance", \
     "--config", "python:api.config.gunicorn_config", \
     "api.app:app"]
