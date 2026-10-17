param([switch]$NoPause)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot 'install_log.txt'
"=== install $(Get-Date) ===" | Set-Content $log -Encoding utf8
function Step($title, [scriptblock]$cmd) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    "=== $title ===" | Add-Content $log -Encoding utf8
    & $cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "$title failed (exit code $LASTEXITCODE)" }
}
try {
    $uv = $null
    $c = Get-Command uv -ErrorAction SilentlyContinue
    if ($c) { $uv = $c.Source } else { $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
    if (-not (Test-Path $uv)) {
        Write-Host "Installing uv (small Python manager)..." -ForegroundColor Cyan
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    if (-not (Test-Path $uv)) { throw "uv was not found at $uv after install" }
    $py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    Step 'Create Python 3.11 environment' { & $uv venv --python 3.11 --allow-existing .venv }
    Step 'Install packages (Whisper, voice models, overlay, CUDA libraries ~1.5 GB)' {
        & $uv pip install --python $py faster-whisper sherpa-onnx PyAudioWPatch PySide6-Essentials soxr numpy soundfile `
            nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
    }
    New-Item -ItemType Directory -Force models | Out-Null
    $rel = 'https://github.com/k2-fsa/sherpa-onnx/releases/download'
    foreach ($m in @(@('asr-models', 'silero_vad.onnx'), @('speaker-recongition-models', 'nemo_en_titanet_small.onnx'))) {
        $dst = Join-Path 'models' $m[1]
        if (-not (Test-Path $dst)) {
            Step "Download $($m[1])" { Invoke-WebRequest "$rel/$($m[0])/$($m[1])" -OutFile $dst -UseBasicParsing; $global:LASTEXITCODE = 0 }
        }
    }
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
    Step 'Download Whisper large-v3-turbo (~1.6 GB)' {
        & $py -c "from faster_whisper import download_model; print(download_model('large-v3-turbo', cache_dir='models'))"
    }
    Step 'Test on the GPU with a recorded sample call' {
        & $py -u selftest.py test\conversation.flac --truth test\conversation.json
    }
    Step 'Desktop icon' { powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'make_shortcut.ps1') }
    Write-Host "`nSetup finished. Start it from the 'D&D Subtitles' icon on your desktop." -ForegroundColor Green
    "SETUP OK" | Add-Content $log -Encoding utf8
} catch {
    Write-Host "`nSETUP FAILED: $_" -ForegroundColor Red
    "SETUP FAILED: $_" | Add-Content $log -Encoding utf8
    if ($NoPause) { exit 1 }
}
if (-not $NoPause) { Read-Host "Press Enter to close" }
