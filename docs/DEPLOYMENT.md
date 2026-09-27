# Chandra-Align Deployment Guide

## Local deployment (Docker)

```bash
docker compose up --build
```

This builds the production image and starts the service:

| Component | What happens |
|---|---|
| Base image | `osgeo/gdal:ubuntu-small-latest` (GDAL + Python preinstalled) |
| Python deps | installed from `requirements.txt` (numpy, opencv, rasterio, fastapi, torch/lightglue, …) |
| React viewer | built from `web/` with Node.js 20 → static assets in `web/dist` |
| Server | FastAPI via `uvicorn api.main:app` on port **8000** |

Volumes mounted (dynamic, never baked into the image):

| Host path | Container path | Purpose |
|---|---|---|
| `./data` | `/app/data` | `download_log.csv`, products, NAC frames (read-write for logging) |
| `./outputs` | `/app/outputs` | metrics.json, match_points.geojson, registered COGs |
| `./reports` | `/app/reports` | run reports |

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | React viewer (built static assets; SPA fallback to an info page when not built) |
| `GET /health` | liveness check |
| `GET /docs` | OpenAPI/Swagger UI |
| `GET /api/runs` | list all runs — including log entries with no run directory (shown with `status: logged-not-run`, `trust_flag: Not Trusted`, `UNMEASURED` metrics) |
| `POST /register` | upload image pair + config → COG + match points + metrics + trust flag |
| `GET /runs/{id}/registered.tif` | registered COG (or JSON `UNMEASURED` payload if not generated) |
| `GET /runs/{id}/match_points.geojson` | match points sidecar (or `UNMEASURED` payload) |
| `GET /runs/{id}/metrics.json` | metrics bundle (or `UNMEASURED` payload) |

Graceful degradation: a run ID present in `data/download_log.csv` but with no generated
artifacts returns a **200 JSON payload with explicit `UNMEASURED` states and
`trust_flag: "Not Trusted"`** — never a 500.

## Local development (no Docker)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # POSIX: .venv/bin/pip
pytest                                            # 36 tests
cd web && npm install && npm run build && cd ..   # build viewer (optional)
uvicorn api.main:app --port 8000 --reload         # serve API + viewer
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CHANDRA_ALIGN_WEB_DIST` | `<repo>/web/dist` | where the built viewer lives (set to `/app/web/dist` in the container) |

All paths are relative and resolved from the repo root; no absolute paths are required.

## Pipeline run

```bash
.venv/Scripts/python scripts/run_pipeline.py \
    config/ohrc.yaml fixtures/ref.tif fixtures/mov.tif outputs/run1 outputs/run1/metrics.json
```

Real-data runs (paths outside `fixtures/`) require a matching `data/download_log.csv`
entry — the pipeline exits with code 3 and an explicit JSON error otherwise.

## Honest-claims note

All metrics are from real runs or explicitly labelled `UNMEASURED`. Trust thresholds
are `UNCALIBRATED` until recalibrated on real data. No invented numbers anywhere.
