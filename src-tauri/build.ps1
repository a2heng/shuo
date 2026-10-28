# 说 · Shuo — Windows 打包脚本（Tauri 3 / NSIS）
#
# 前置：
#   - Rust stable（rustup）+ MSVC Build Tools（"使用 C++ 的桌面开发"）
#   - WebView2 Runtime（Win10/11 一般自带；缺失时安装包会提示）
#   - 模型不随包分发，放到 %USERPROFILE%\.shuo\models 或设置 asr_model_dir/llm_model_dir
#
# 用法（在仓库任意位置）：
#   powershell -ExecutionPolicy Bypass -File src-tauri\build.ps1
#   powershell -File src-tauri\build.ps1 -Debug
#
# 产物：
#   src-tauri\target\release\bundle\nsis\Shuo_<版本>_x64-setup.exe

param(
  [switch]$Debug,
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
  throw "找不到 cargo：请先安装 Rust（https://rustup.rs）"
}

if (-not $SkipInstall) {
  $tauri = Get-Command cargo-tauri -ErrorAction SilentlyContinue
  if (-not $tauri) {
    Write-Host "安装 tauri-cli 3.0.0-alpha.5 ..."
    cargo install tauri-cli --version "3.0.0-alpha.5" --locked
  } else {
    $ver = (& cargo tauri --version) -join " "
    Write-Host "cargo-tauri: $ver"
  }
}

# 降噪模型直接来自仓库，打包时会被复制到资源目录
if (-not (Test-Path "src-tauri\resources\denoise.onnx")) {
  throw "缺少 src-tauri\resources\denoise.onnx（仓库自带，检查是否被误删）"
}

if ($Debug) {
  Write-Host "开发模式：cargo tauri dev"
  cargo tauri dev
  exit $LASTEXITCODE
}

Write-Host "构建 NSIS 安装包 ..."
cargo tauri build
if ($LASTEXITCODE -ne 0) { throw "cargo tauri build 失败（$LASTEXITCODE）" }

$out = Get-ChildItem "src-tauri\target\release\bundle\nsis\*.exe" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if ($out) {
  Write-Host ""
  Write-Host "完成：$($out.FullName)"
  Write-Host "（体积 $([math]::Round($out.Length / 1MB, 1)) MB）"
} else {
  throw "没找到 NSIS 产物，检查 src-tauri\target\release\bundle\nsis"
}
