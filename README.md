# CondensateLab

CPU-based desktop & web application for phase-separation modeling: manual and light-controlled condensate formation, FRAP analysis, and interactive droplet nuclear transport. The desktop interface and the hosted web interface share the same FastAPI backend.

Developed by Yixuan Zou (邹依璇). Light control / FRAP module: Pengbo Li — see [source/CONTRIBUTORS.md](source/CONTRIBUTORS.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `source/` | Full source code (FastAPI backend + web UI), `Dockerfile`, and deployment docs. See [source/README.md](source/README.md). |
| `Windows/` | Ready-to-run Windows build (PyInstaller). No Python installation required. |
| `demos/` | Sample outputs of the nuclear transport module (GIF, PNG, CSV, parameters). |
| `docs/zh/` | Original Chinese release notes (使用说明 / 本次更新说明). |

## Quick start

**Windows (packaged build):** open `Windows/CondensateLab` and double-click `CondensateLab.exe`. Keep the `_internal` folder next to the executable; when sharing, copy the whole folder. Details in [USER_GUIDE.md](USER_GUIDE.md).

**From source (Python 3.12):**

```bash
pip install -r source/requirements.txt
cd source
uvicorn app:app --host 127.0.0.1 --port 7860
```

**Docker:** `docker build -t condensatelab source && docker run --rm -p 7860:7860 condensatelab`

**Hugging Face:** create a Docker Space and upload the contents of `source/` (port 7860; default CPU configuration, no CUDA required).

## Documentation

- [USER_GUIDE.md](USER_GUIDE.md) — running the Windows build, web sharing, deployment, and verification status
- [RELEASE_NOTES.md](RELEASE_NOTES.md) — what changed in this release (nuclear-entry module)
- [source/docs/droplet_transport.md](source/docs/droplet_transport.md) — equations, implementation, and limitations of the droplet transport model
- [source/docs/OFFLINE_AND_ERRORS.md](source/docs/OFFLINE_AND_ERRORS.md) — offline behavior and error handling

## Interface languages

The UI ships in Chinese, English, French, Spanish, Portuguese, and Arabic (right-to-left layout for Arabic). Raw results, errors, and some method hints returned by upstream services are kept in their original language.

## Verification status

The Windows build has passed local service startup and front-page checks. Visual inspection of the desktop window is not complete yet.
