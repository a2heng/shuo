$ErrorActionPreference = "Stop"
$7z = "C:\Program Files\7-Zip\7z.exe"

# 0. Pre-check: generate locale files
Write-Host "Generating locale files..."
python -m i18n.build_i18n_locales generate
if ($LASTEXITCODE -ne 0) {
    Write-Error "FAIL: locale generation failed"; exit 1
}
Write-Host "PASS: locale files generated"

# 1. Timestamp
$date = Get-Date -Format 'yyyy-MM-dd-HHmm'
Set-Content build_version.py "BUILD_DATE = `"$date`"" -Encoding UTF8 -NoNewline

# 2. PyInstaller
# 排斥钩子：在 Hook 阶段阻断 torch/triton 收集，不跑 hook-torch.py
$hooksDir = (Get-Item "hooks").FullName
pyinstaller --clean --name Shuo --noconsole --icon=shuo.ico `
    --additional-hooks-dir="$hooksDir" `
    --hidden-import=pyaudio --hidden-import=tokenizers `
    --hidden-import=PySide6.QtOpenGL --hidden-import=PySide6.QtOpenGLWidgets `
    --add-data="locales;locales" `
    --add-data="Qwen3.5-2B-ONNX-OPT;Qwen3.5-2B-ONNX-OPT" `
    --add-data="Qwen3-ASR-0.6B-ONNX-CPU;Qwen3-ASR-0.6B-ONNX-CPU" `
    --add-data="onnx_infer/denoise.onnx;onnx_infer/" `
    --exclude-module=torch --exclude-module=triton `
    -y asr_gui.py

# 3. Cleanup — remove packages pulled in by dependency chains
$target = "dist\Shuo\_internal"
Remove-Item "$target\tcl86t.dll", "$target\tk86t.dll", "$target\_tkinter.pyd" -Force -ErrorAction SilentlyContinue
Remove-Item "$target\_tcl_data", "$target\_tk_data", "$target\tcl8" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$target\PySide6\opengl32sw.dll" -Force -ErrorAction SilentlyContinue
Remove-Item "$target\PySide6\Qt6Quick.dll", "$target\PySide6\Qt6Qml.dll", "$target\PySide6\Qt6Pdf.dll", "$target\PySide6\Qt6DataVisualization.dll" -Force -ErrorAction SilentlyContinue
Remove-Item "$target\PySide6\QtOpenGL.pyd", "$target\PySide6\QtQuick.pyd", "$target\PySide6\QtQml.pyd", "$target\PySide6\QtPdf.pyd" -Force -ErrorAction SilentlyContinue
# 安全兜底：检查 torch 是否漏网
if (Test-Path "$target\torch") { Remove-Item "$target\torch" -Recurse -Force }

# 4. Verify torch excluded
if (Test-Path "$target\torch") {
    Write-Error "FAIL: torch still bundled"; exit 1
}
Write-Host "PASS: torch excluded"

# 5. SFX archive
Push-Location dist
try {
    & $7z a "Shuo_$date.7z" "Shuo"
    & cmd /c "copy /b ..\7z\7z.sfx + ..\7z\sfx_config.txt + Shuo_$date.7z Shuo_$date.exe >nul"
    Remove-Item "Shuo_$date.7z"
} finally {
    Pop-Location
}

# 6. NoModel — 用 robocopy 直接排除模型目录，零冗余读写
$nomodel = "dist\Shuo_nomodel"
robocopy "dist\Shuo" $nomodel /E /XD "Qwen3.5-2B-ONNX-OPT" "Qwen3-ASR-0.6B-ONNX-CPU" /NDL /NFL /NJH /NJS
# onnx_infer/denoise.onnx 保留在 nomodel 包中
Push-Location dist
try {
    & $7z a "Shuo_nomodel_$date.7z" "Shuo_nomodel"
    & cmd /c "copy /b ..\7z\7z.sfx + ..\7z\sfx_config.txt + Shuo_nomodel_$date.7z Shuo_nomodel_$date.exe >nul"
    Remove-Item "Shuo_nomodel_$date.7z"
} finally {
    Pop-Location
}
Remove-Item $nomodel -Recurse -Force
Write-Host "  + dist\Shuo_nomodel_$date.exe"

Remove-Item build_version.py -Force
Write-Host "DONE: dist\Shuo_$date.exe"