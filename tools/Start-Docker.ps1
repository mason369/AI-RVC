param(
    [ValidateSet('cuda', 'cpu')]
    [string]$Variant = 'cuda',
    [switch]$WithAuth
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Version = (Get-Content -LiteralPath (Join-Path $ProjectRoot 'VERSION') -Raw).Trim()
$ComposeFile = if ($Variant -eq 'cpu') { 'compose.cpu.yaml' } else { 'compose.yaml' }
$ComposeArguments = @('compose', '-f', $ComposeFile)
if ($WithAuth) { $ComposeArguments += @('-f', 'compose.auth.yaml') }

function Invoke-DockerChecked {
    param([string[]]$DockerArguments)
    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker 命令失败，退出码 $LASTEXITCODE。请根据上方原始错误处理后再运行。"
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw '未找到 Docker CLI。Windows 请先启动 Docker Desktop（WSL2 后端）；手动在 WSL 安装 Engine 的用户请在 WSL 终端使用 Compose 命令。'
}

Push-Location -LiteralPath $ProjectRoot
try {
    $ContainerOs = (Invoke-DockerChecked @('info', '--format', '{{.OSType}}')).Trim()
    if ($ContainerOs -ne 'linux') { throw '本项目使用 Linux 镜像，请将 Docker Desktop 切换到 Linux 容器。' }
    Invoke-DockerChecked @('compose', 'version')
    $Archive = Join-Path $ProjectRoot "AI-RVC-$Version-$Variant.tar.gz"
    if (Test-Path -LiteralPath $Archive -PathType Leaf) {
        $Checksums = Join-Path $ProjectRoot 'SHA256SUMS'
        if (-not (Test-Path -LiteralPath $Checksums -PathType Leaf)) {
            throw '镜像归档存在，但缺少 SHA256SUMS，已停止导入。'
        }
        $ArchiveName = [IO.Path]::GetFileName($Archive)
        $MatchesForFile = @(Get-Content -LiteralPath $Checksums | Where-Object {
            $_ -match '^([0-9a-fA-F]{64})  (.+)$' -and $Matches[2] -ceq $ArchiveName
        })
        if ($MatchesForFile.Count -ne 1) {
            throw "校验清单中没有唯一的 $ArchiveName 条目。"
        }
        $ExpectedHash = $MatchesForFile[0].Substring(0, 64)
        if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash -ine $ExpectedHash) {
            throw "镜像归档校验失败：$ArchiveName。已停止导入。"
        }
        Invoke-DockerChecked @('load', '-i', $Archive)
    } else {
        Invoke-DockerChecked ($ComposeArguments + @('build'))
    }
    Invoke-DockerChecked ($ComposeArguments + @('run', '--rm', 'ai-rvc', 'prepare'))
    Invoke-DockerChecked ($ComposeArguments + @('up', '-d', '--no-build', '--wait', '--wait-timeout', '180'))
    Invoke-DockerChecked ($ComposeArguments + @('ps'))
    Write-Host 'Docker 服务已通过健康检查。默认访问 http://127.0.0.1:7860；如已设置 .env，以其中的地址和端口为准。'
} finally {
    Pop-Location
}
