# setup_venv.ps1
# veco-ai environment setup
# Stack: Python 3.13 | faster-whisper | Gemma4 | PySceneDetect | trimesh | FAISS | Ollama

#Requires -Version 7.0
$ErrorActionPreference = "Stop"

# ── Change to script directory so relative paths work ─────────────────────────
Set-Location $PSScriptRoot

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "── $msg" -ForegroundColor Cyan
}
function Write-Ok([string]$msg)   { Write-Host "  [OK]   $msg" -ForegroundColor Green  }
function Write-Warn([string]$msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg)  { Write-Host "  [ERR]  $msg" -ForegroundColor Red    }
function Write-Info([string]$msg) { Write-Host "         $msg" -ForegroundColor Gray   }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Python 3.13
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Python 3.13"

$launcher = Get-Command "py" -ErrorAction SilentlyContinue
if (-not $launcher) {
    Write-Err "'py' launcher not found. Install Python 3.13 from https://python.org"
    exit 1
}

$verRaw = & py -3.13 -c "import sys; print(sys.version)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "Python 3.13 not found. Install it from https://python.org"
    exit 1
}
Write-Ok "Python $verRaw"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Virtual environment
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Virtual environment (.venv)"

if (Test-Path ".venv") {
    $recreate = Read-Host "  .venv already exists. Recreate? [y/N]"
    if ($recreate -match "^[yY]") {
        Remove-Item -Recurse -Force ".venv"
        Write-Ok "Removed existing .venv."
    } else {
        Write-Ok "Reusing existing .venv."
    }
}

if (-not (Test-Path ".venv")) {
    Write-Host "  Creating .venv ..."
    & py -3.13 -m venv .venv
    Write-Ok ".venv created."
}

$venvPy = (Resolve-Path ".venv\Scripts\python.exe").Path
. ".venv\Scripts\Activate.ps1"
Write-Ok "Activated: $venvPy"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Upgrade pip / setuptools / wheel
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Upgrading pip / setuptools / wheel"
& $venvPy -m pip install --upgrade pip setuptools wheel --quiet
Write-Ok "Done."

# ─────────────────────────────────────────────────────────────────────────────
# 4. PyTorch  — installed first with the right index URL
#    (requirements.txt pins >=2.6; pip will skip re-install if already satisfied)
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "PyTorch + torchaudio"

# Detect CUDA via nvidia-smi
$cudaVersion  = $null
$installCuda  = $false
$wheelTag     = "cpu"

if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    $smiOut = & nvidia-smi 2>&1 | Out-String
    if ($smiOut -match "CUDA Version:\s*(\d+)\.(\d+)") {
        $cudaMajor    = [int]$Matches[1]
        $cudaMinor    = [int]$Matches[2]
        $cudaVersion  = "$cudaMajor.$cudaMinor"
        Write-Host "  Detected CUDA $cudaVersion via nvidia-smi."
    }
}

if ($cudaVersion) {
    $answer = Read-Host "  Install PyTorch with CUDA support? [Y/n]"
    $installCuda = $answer -notmatch "^[nN]"
} else {
    Write-Warn "CUDA not detected — will install CPU build."
}

if ($installCuda) {
    # Map driver-reported CUDA version to the nearest available PyTorch wheel
    $cudaInt = $cudaMajor * 10 + $cudaMinor
    $wheelTag = switch ($true) {
		#($cudaInt -ge 130) { "cu130" }   # future-proof
        ($cudaInt -ge 128) { "cu128" }   # future-proof
        ($cudaInt -ge 126) { "cu126" }
        default {
            Write-Warn "CUDA $cudaVersion predates PyTorch 2.6 wheels — falling back to CPU."
            $installCuda = $false
            "cpu"
        }
    }
}

$torchIndex = "https://download.pytorch.org/whl/$wheelTag"
Write-Host "  Wheel index: $torchIndex"
& $venvPy -m pip install torch torchaudio --index-url $torchIndex --quiet

if ($installCuda) {
    Write-Ok "PyTorch installed (CUDA $wheelTag)."
} else {
    Write-Ok "PyTorch installed (CPU)."
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. requirements.txt
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Installing requirements.txt"

if (-not (Test-Path "requirements.txt")) {
    Write-Err "requirements.txt not found in $PSScriptRoot"
    exit 1
}

# torch/torchaudio already installed above; pip will skip them automatically.
& $venvPy -m pip install -r requirements.txt --quiet
Write-Ok "All Python packages installed."

# ─────────────────────────────────────────────────────────────────────────────
# 6. Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Smoke test (key imports)"

$imports = @(
    @{ name = "torch";                   stmt = "import torch; print(torch.__version__)" }
    @{ name = "faster_whisper";          stmt = "from faster_whisper import WhisperModel" }
    @{ name = "sentence_transformers";   stmt = "from sentence_transformers import SentenceTransformer" }
    @{ name = "faiss";                   stmt = "import faiss" }
    @{ name = "unstructured";            stmt = "from unstructured.partition.auto import partition" }
    @{ name = "scenedetect";             stmt = "from scenedetect import detect" }
    @{ name = "trimesh";                 stmt = "import trimesh" }
    @{ name = "ollama";                  stmt = "import ollama" }
)

foreach ($imp in $imports) {
    $out = & $venvPy -c $imp.stmt 2>&1
    if ($LASTEXITCODE -eq 0) {
        $detail = if ($out) { " ($out)" } else { "" }
        Write-Ok "$($imp.name)$detail"
    } else {
        Write-Warn "$($imp.name) — import failed (check installation)"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. System dependencies
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "System dependencies"

# ffmpeg (required)
if (Get-Command "ffmpeg" -ErrorAction SilentlyContinue) {
    $ffVer = (& ffmpeg -version 2>&1 | Select-Object -First 1) -replace "ffmpeg version ", ""
    Write-Ok "ffmpeg $ffVer"
} else {
    Write-Err "ffmpeg NOT found — audio/video processing will fail."
    Write-Info "Install:  winget install Gyan.FFmpeg"
    Write-Info "      or: choco install ffmpeg"
    Write-Info "      or: https://ffmpeg.org/download.html"
}

# tesseract (optional)
if (Get-Command "tesseract" -ErrorAction SilentlyContinue) {
    $tessVer = (& tesseract --version 2>&1 | Select-Object -First 1)
    Write-Ok "tesseract $tessVer"
} else {
    Write-Warn "tesseract not found (optional — OCR fallback for unstructured)."
    Write-Info "Install:  winget install UB-Mannheim.TesseractOCR"
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. Ollama
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Ollama"

if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
    Write-Ok "ollama found."
    $ollamaList = & ollama list 2>&1 | Out-String
    foreach ($model in @("gemma4", "gemma3")) {
        if ($ollamaList -match $model) {
            Write-Ok "Model '$model' available."
        } else {
            Write-Warn "Model '$model' not pulled yet."
            Write-Info "Pull with:  ollama pull $model"
        }
    }
} else {
    Write-Warn "ollama not found. Install from https://ollama.ai"
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. HuggingFace token (speaker diarization)
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "HuggingFace token (speaker diarization)"

$hfToken = $env:HF_TOKEN ?? $env:HUGGINGFACE_TOKEN
if ($hfToken) {
    Write-Ok "HF_TOKEN is set ($($hfToken.Substring(0, [Math]::Min(8, $hfToken.Length)))...)."
} else {
    Write-Warn "HF_TOKEN not set — speaker diarization will be disabled at runtime."
    Write-Info "1. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1"
    Write-Info "2. Create token: https://huggingface.co/settings/tokens"
    Write-Info "3. Add to your PowerShell profile:"
    Write-Info '      $env:HF_TOKEN = "hf_..."'
}

$pyannoteOk = & $venvPy -c "from pyannote.audio import Pipeline" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Ok "pyannote.audio available — speaker diarization can be enabled."
} else {
    Write-Info "pyannote.audio not installed — speaker diarization stays optional."
    Write-Info "If your Python 3.13 environment can resolve lightning, install with:"
    Write-Info "    pip install pyannote.audio"
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. CAD (STEP/IGES via pythonOCC — optional, conda only)
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "CAD: pythonOCC (optional — STEP/IGES)"

$occOk = & $venvPy -c "from OCC.Extend.DataExchange import read_step_file" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Ok "pythonOCC available — STEP/IGES support enabled."
} else {
    Write-Info "pythonOCC not installed — STEP/IGES will be skipped (trimesh still handles STL/OBJ/PLY)."
    Write-Info "To enable STEP/IGES support, run in a conda environment:"
    Write-Info "    conda install -c conda-forge pythonocc-core"
}

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  veco-ai environment ready." -ForegroundColor Green
Write-Host ""
Write-Host "  Activate:    .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  Run (doc):   python veco_ai\veco_ai.py report.pdf --compress" -ForegroundColor White
Write-Host "  Run (audio): python veco_ai\veco_ai.py interview.wav --diarize true" -ForegroundColor White
Write-Host "  Run (video): python veco_ai\veco_ai.py meeting.mp4" -ForegroundColor White
Write-Host "  Run (CAD):   python veco_ai\veco_ai.py model.step" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
