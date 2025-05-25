# ==============================================================================
# STAGE 1: BUILD STAGE (Python 3.9 Compatible)
# ==============================================================================
FROM python:3.9-slim as builder

# Set working directory for build stage
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip dan install wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch versi yang kompatibel dengan Python 3.9
RUN pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Buat requirements_modified.txt tanpa torch (karena sudah diinstall)
RUN grep -v "torch" requirements.txt > requirements_modified.txt || touch requirements_modified.txt

# Install requirements lainnya
RUN pip install --no-cache-dir -r requirements_modified.txt

# ==============================================================================
# STAGE 2: RUNTIME STAGE
# ==============================================================================
FROM python:3.9-slim as runtime

# Set working directory
WORKDIR /app

# Copy virtual environment dari build stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Modify sources.list to include contrib and non-free
RUN sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list.d/debian.sources \
    || sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections \
    && apt-get install -y --no-install-recommends ttf-mscorefonts-installer \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY Server/ /app/
COPY Website/ /app/static/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=neopark_server.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/combined/status || exit 1

# Run the application
CMD ["python", "neopark_server.py"]