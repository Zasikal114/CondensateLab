---
title: CondensateLab
sdk: docker
app_port: 7860
---

# CondensateLab

Software developed by Yixuan Zou (邹依璇).

CPU application for phase separation, light control / FRAP and droplet nuclear transport. The desktop interface and hosted interface use the same FastAPI backend.

## Run

Python 3.12: `pip install -r requirements.txt`, then `uvicorn app:app --host 127.0.0.1 --port 7860`.
Docker: `docker build -t condensatelab .` then `docker run --rm -p 7860:7860 condensatelab`.
Hugging Face: create a Docker Space and upload this directory. Expose port 7860. The default CPU configuration does not require CUDA.
Windows: launch CondensateLab.exe next to its complete `_internal` folder. WebView2 Runtime must already be installed. See docs/OFFLINE_AND_ERRORS.md.

## Models and outputs

Manual phase grid: 16–256. Light/FRAP is based on Li Pengbo's implementation. The only nuclear model is the interactive droplet transport model; see docs/droplet_transport.md for equations and limitations. GIFs contain actual computed frames. CSV, NPZ and parameter exports vary by module.

## Operations

Job files expire after 24 hours; ephemeral hosted storage may reset on restart. Download needed outputs. Use one Uvicorn worker: the task queue is process-local. MAX_PENDING_JOBS limits the total pending queue. This release has not been load-tested for 100–200 simultaneous users.

## Attribution and release

See CONTRIBUTORS.md and MODEL_SOURCES.json. Confirm licenses for all contributed model files before a formal open-source award release. The Wiki repository's documentation license does not automatically resolve software-component licensing.
