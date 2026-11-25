# Windows Installation & Setup Guide

RoadGuard AI is engineered and tested natively on **Windows 10/11** with PowerShell and Command Prompt.

---

## 1. Prerequisites

- **Python 3.10 – 3.13** (`python --version`)
- **Node.js 18+ & npm** (`node --version`, `npm.cmd --version`)
- *(Optional)* **NVIDIA GPU Driver** for local CUDA acceleration (CPU inference works automatically out-of-the-box)

> **Windows PowerShell ExecutionPolicy Note:**
> Many Windows installations restrict unsigned `.ps1` scripts (`PSSecurityException`). RoadGuard AI provides `.cmd` launchers in `scripts/` that invoke `powershell -NoProfile -ExecutionPolicy Bypass` automatically so you never need to weaken system-wide security policies.

---

## 2. Automated Installation

Open PowerShell or Command Prompt in the project root:

```powershell
cd C:\Users\Dhruv\Downloads\RoadGuardAI
.\scripts\setup_windows.cmd
```

Or run the PowerShell script explicitly:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

What setup performs:
1. Creates an isolated virtual environment at `RoadGuardAI\.venv`.
2. Installs FastAPI, SQLAlchemy, OpenCV, Ultralytics YOLO, Albumentations, Scikit-Image, ReportLab, Kaggle CLI, and Pytest into `.venv`.
3. Copies `.env.example` to `.env` if not already present.
4. Runs `npm.cmd install` and `npm.cmd run build` inside `frontend/`.

---

## 3. Secure Kaggle API Configuration (Optional for Local Dataset Download)

Never paste Kaggle API tokens into source files or `.env`. To authenticate the Kaggle CLI locally on Windows:

1. Download `kaggle.json` from [https://www.kaggle.com/settings](https://www.kaggle.com/settings).
2. Move it into `%USERPROFILE%\.kaggle\kaggle.json`:
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null
   Move-Item -Force -Path "$env:USERPROFILE\Downloads\kaggle.json" -Destination "$env:USERPROFILE\.kaggle\kaggle.json"
   ```
3. Verify authentication safely:
   ```powershell
   .\.venv\Scripts\python.exe -m ml.dataset.cli check-auth
   ```

---

## 4. Running the Application Locally

Start the backend and frontend in two separate terminals:

**Terminal 1 — FastAPI Backend (`http://127.0.0.1:8000`):**
```powershell
cd C:\Users\Dhruv\Downloads\RoadGuardAI
.\scripts\start_backend.cmd
```

**Terminal 2 — Vite + React Frontend (`http://localhost:5173`):**
```powershell
cd C:\Users\Dhruv\Downloads\RoadGuardAI
.\scripts\start_frontend.cmd
```

Open **`http://localhost:5173`** in your browser.
