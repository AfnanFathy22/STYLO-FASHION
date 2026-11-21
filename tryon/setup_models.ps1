# tryon/setup_models.ps1
# --------------------------------------------------------------------------
# One-time setup script that clones IDM-VTON and OOTDiffusion into
# third_party/, installs their extra Python requirements into your CURRENT
# activated venv, and downloads their checkpoints from Hugging Face.
#
# Run this from the project root (same folder as run_video.py), with your
# venv already activated:
#
#     cd C:\Users\Admin\Desktop\Stage1\Stage1
#     .\tryon\setup_models.ps1
#
# Requirements before running:
#   - git installed and on PATH
#   - git-lfs installed (https://git-lfs.com) -> run `git lfs install` once
#   - Python venv activated (you already have (venv) in your prompt)
#   - ~40-50 GB free disk space (both models' weights combined)
# --------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot   # project root (parent of tryon/)
$thirdParty = Join-Path $root "third_party"
New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

Write-Host "=== 1/4: Installing shared Python packages (diffusers, transformers, huggingface_hub, accelerate) ===" -ForegroundColor Cyan
pip install --upgrade diffusers transformers accelerate huggingface_hub safetensors

# --------------------------------------------------------------------------
# IDM-VTON
# --------------------------------------------------------------------------
Write-Host "`n=== 2/4: Cloning IDM-VTON ===" -ForegroundColor Cyan
$idmDir = Join-Path $thirdParty "IDM-VTON"
if (-not (Test-Path $idmDir)) {
    git clone https://github.com/yisol/IDM-VTON.git $idmDir
} else {
    Write-Host "third_party/IDM-VTON already exists, skipping clone."
}

Write-Host "Installing IDM-VTON's own requirements..."
pip install -r (Join-Path $idmDir "requirements.txt")

Write-Host "Downloading IDM-VTON weights from Hugging Face (yisol/IDM-VTON)..."
Write-Host "(this uses tryon\download_idm_vton.py -- resumable, safe to Ctrl+C and re-run)"
pip install hf_transfer --quiet
python (Join-Path $PSScriptRoot "download_idm_vton.py")

# --------------------------------------------------------------------------
# OOTDiffusion
# --------------------------------------------------------------------------
Write-Host "`n=== 3/4: Cloning OOTDiffusion ===" -ForegroundColor Cyan
$ootdDir = Join-Path $thirdParty "OOTDiffusion"
if (-not (Test-Path $ootdDir)) {
    git clone https://github.com/levihsu/OOTDiffusion.git $ootdDir
} else {
    Write-Host "third_party/OOTDiffusion already exists, skipping clone."
}

Write-Host "Installing OOTDiffusion's own requirements..."
pip install -r (Join-Path $ootdDir "requirements.txt")

Write-Host "Downloading OOTDiffusion weights from Hugging Face (levihsu/OOTDiffusion)..."
Write-Host "(this uses tryon\download_ootd.py -- resumable, safe to Ctrl+C and re-run)"
python (Join-Path $PSScriptRoot "download_ootd.py")

Write-Host "`n=== 4/4: Done ===" -ForegroundColor Green
Write-Host "IDM-VTON      -> $idmDir"
Write-Host "OOTDiffusion  -> $ootdDir"
Write-Host ""
Write-Host "NEXT STEPS (important, read tryon/README.md):"
Write-Host "  1. Both repos' own READMEs may list a couple of EXTRA manual downloads"
Write-Host "     (e.g. DensePose checkpoint for IDM-VTON, human-parsing checkpoints)."
Write-Host "     Open third_party\IDM-VTON\README.md and third_party\OOTDiffusion\README.md"
Write-Host "     and follow any 'Download checkpoints' steps that snapshot_download above"
Write-Host "     did not already cover."
Write-Host "  2. Test EACH model's own official demo/inference script first, standalone,"
Write-Host "     BEFORE trying tryon/idm_vton_wrapper.py or tryon/ootd_wrapper.py."
Write-Host "     This confirms the environment itself works before we add our glue code."
