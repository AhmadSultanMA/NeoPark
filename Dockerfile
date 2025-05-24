# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Modify sources.list to include contrib and non-free
RUN sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list.d/debian.sources \
    || sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    # Pastikan Anda menerima EULA untuk mscorefonts
    && echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections \
    && apt-get install -y --no-install-recommends ttf-mscorefonts-installer \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file from the root of the build context
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Server files (termasuk neopark_server.py dan fine-best.pt)
COPY Server/ /app/

# Copy Website files to /app/static (jika Nginx tidak dipakai untuk static files)
COPY Website/ /app/static/

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=neopark_server.py
ENV FLASK_ENV=production

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check (sudah ada dan baik)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/combined/status || exit 1

# Run the application
CMD ["python", "neopark_server.py"]