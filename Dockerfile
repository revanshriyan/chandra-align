# Chandra-Align production image — OSGeo GDAL base (GDAL + Python preinstalled)
FROM osgeo/gdal:ubuntu-small-3.6.3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for the pipeline (rasterio wheels need GDAL headers present in base;
# opencv needs libgl; node needed for the React viewer build)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgdal-dev \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 (NodeSource) for building the React viewer
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (layer cache)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Build the React viewer
COPY web/ ./web/
RUN cd web && npm install --no-audit --no-fund && npm run build

# Engine + API + scripts + configs
COPY chandra_align/ ./chandra_align/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY vendored/ ./vendored/
COPY tests/ ./tests/

# data/, outputs/, reports/ are mounted at runtime (never baked in)
RUN mkdir -p data outputs reports

EXPOSE 8000

ENV CHANDRA_ALIGN_WEB_DIST=/app/web/dist

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
