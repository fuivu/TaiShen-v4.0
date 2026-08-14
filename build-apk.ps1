# ============================================================
#  Local AI Painter v4.0.0 — Windows 一键编译脚本 (PowerShell)
#  用法: 在 PowerShell 中运行  .\build-apk.ps1  [-BuildType release]
# ============================================================

[CmdletBinding()]
param(
    [ValidateSet("debug", "release")]
    [string]$BuildType = "release"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── 颜色函数 ──────────────────────────────────────────────
function Write-Info($msg)    { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Step($msg)    { Write-Host "`n▶ $msg" -ForegroundColor Blue -BackgroundColor Black }
function Write-Ok($msg)      { Write-Host "✓  $msg" -ForegroundColor Green }
function Write-Warn($msg)    { Write-Host "⚠  $msg" -ForegroundColor Yellow }
function Write-Error($msg)   { Write-Host "✗  $msg" -ForegroundColor Red }
function Write-Section($msg) { Write-Host "`n══ $msg ══" -ForegroundColor Cyan }

# ── 横幅 ──────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "  ║   Local AI Painter v4.0.0 TaiShen Build    ║" -ForegroundColor Blue
Write-Host "  ║   INT2/FP8 • GraphFusion • CIM • ZeroCopy  ║" -ForegroundColor Blue
Write-Host "  ╚════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host "  Project: $ProjectRoot"
Write-Host "  Date:    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# ── Step 1: 检测 JDK ────────────────────────────────────
Write-Section "Step 1/5 — JDK 检测"
try {
    $javaVersion = & java -version 2>&1 | Select-Object -First 1
    Write-Ok "JDK 已安装: $javaVersion"
} catch {
    Write-Error "未找到 Java，请先安装 JDK 17+"
    Write-Host "  下载: https://adoptium.net/temurin/releases/?version=17"
    exit 1
}

# ── Step 2: 检测 Android SDK ────────────────────────────
Write-Section "Step 2/5 — Android SDK 检测"
$androidHome = $env:ANDROID_HOME
if (-not $androidHome) { $androidHome = $env:ANDROID_SDK_ROOT }
if (-not $androidHome) {
    $commonPaths = @(
        "$env:LOCALAPPDATA\Android\Sdk",
        "C:\Android\Sdk",
        "$env:USERPROFILE\AppData\Local\Android\Sdk"
    )
    foreach ($p in $commonPaths) {
        if (Test-Path $p) { $androidHome = $p; break }
    }
}

if ($androidHome -and (Test-Path $androidHome)) {
    Write-Ok "Android SDK: $androidHome"
    $env:ANDROID_HOME = $androidHome
} else {
    Write-Error "未找到 Android SDK"
    Write-Host "  请设置: \$env:ANDROID_HOME = 'C:\path\to\android\sdk'"
    exit 1
}

# ── Step 3: 检测 NDK ────────────────────────────────────
Write-Section "Step 3/5 — NDK 检测"
$ndkPath = Get-ChildItem "$androidHome\ndk" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ndkPath) {
    Write-Ok "NDK 已安装: $($ndkPath.Name)"
} else {
    Write-Warn "未找到 NDK，请通过 SDK Manager 安装 NDK 26.1.10909125"
}

# ── Step 4: 安装 Gradle Wrapper ──────────────────────────
Write-Section "Step 4/5 — Gradle Wrapper 检查"
$wrapperJar = "$ProjectRoot\gradle\wrapper\gradle-wrapper.jar"
if (Test-Path $wrapperJar) {
    Write-Ok "gradle-wrapper.jar 已就绪"
} else {
    Write-Warn "gradle-wrapper.jar 缺失"
    Write-Host "  请将已下载的 gradle-8.7-bin.zip 放到项目根目录"
    Write-Host "  然后运行: Expand-Archive -Path .\gradle-8.7-bin.zip -DestinationPath .\temp_gradle"
    Write-Host "  再运行: Copy-Item .\temp_gradle\gradle-8.7\lib\plugins\gradle-wrapper-*.jar .\gradle\wrapper\gradle-wrapper.jar"
    Write-Host "  或者更简单: 在有网络的电脑上运行 'gradle wrapper --gradle-version 8.7'"
    
    $userChoice = Read-Host "是否尝试从网络下载? (y/n)"
    if ($userChoice -eq 'y') {
        $tempZip = "$env:TEMP\gradle-8.7-bin.zip"
        Write-Info "下载中..."
        Invoke-WebRequest -Uri "https://services.gradle.org/distributions/gradle-8.7-bin.zip" -OutFile $tempZip
        $tempDir = "$env:TEMP\gradle_extract"
        Expand-Archive -Path $tempZip -DestinationPath $tempDir -Force
        $wrapperSource = Get-ChildItem "$tempDir" -Filter "gradle-wrapper-*.jar" -Recurse | Select-Object -First 1
        if ($wrapperSource) {
            Copy-Item $wrapperSource.FullName $wrapperJar -Force
            Write-Ok "gradle-wrapper.jar 安装成功"
        }
    }
}

# ── Step 5: 编译 APK ─────────────────────────────────────
Write-Section "Step 5/5 — 编译 APK ($BuildType)"

if ($BuildType -eq "debug") {
    $task = "assembleDebug"
    $outputDir = "debug"
} else {
    $task = "assembleRelease"
    $outputDir = "release"
}

Write-Info "执行: gradlew clean $task"
$startTime = Get-Date

try {
    & "$ProjectRoot\gradlew.bat" clean $task --no-daemon --stacktrace
    $exitCode = $LASTEXITCODE
} catch {
    $exitCode = 1
}

$endTime = Get-Date
$elapsed = [int]($endTime - $startTime).TotalSeconds

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Section "构建成功!"
    Write-Ok "耗时: ${elapsed} 秒"
    
    $apkPath = Get-ChildItem "$ProjectRoot\app\build\outputs\apk\$outputDir" -Filter "*.apk" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($apkPath) {
        $apkSize = [math]::Round($apkPath.Length / 1MB, 2)
        Write-Host ""
        Write-Host "  APK 文件:" -ForegroundColor White
        Write-Host "    路径: $($apkPath.FullName)" -ForegroundColor Green
        Write-Host "    大小: ${apkSize} MB" -ForegroundColor Cyan
        
        # 复制到根目录
        Copy-Item $apkPath.FullName "$ProjectRoot\TaiShen-v4.0.0-$BuildType.apk" -Force
        Write-Ok "已复制到: $ProjectRoot\TaiShen-v4.0.0-$BuildType.apk"
    }
    
    Write-Host ""
    Write-Host "  ╔════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "  ║   Local AI Painter v4.0.0 TaiShen 构建完成!  ║" -ForegroundColor Green
    Write-Host "  ╚════════════════════════════════════════╝" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Error "构建失败! 耗时: ${elapsed} 秒"
    Write-Host ""
    Write-Host "  排查步骤:"
    Write-Host "  1. 确认 ANDROID_HOME 环境变量正确"
    Write-Host "  2. 确认 NDK 版本 >= 26"
    Write-Host "  3. 确认 JDK 版本 >= 17"
    Write-Host "  4. 检查网络能否访问 Maven 仓库"
    Write-Host "  5. 查看上方完整错误日志"
    exit 1
}
