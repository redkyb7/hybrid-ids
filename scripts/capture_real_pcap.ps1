param(
    [Parameter(Mandatory = $true)]
    [string]$Interface,
    [ValidateRange(5, 300)]
    [int]$DurationSeconds = 60,
    [ValidateNotNullOrEmpty()]
    [string]$CaptureFilter = 'ip and (tcp or udp or icmp)',
    [string]$OutputPath = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$dataRoot = Join-Path $repoRoot 'data/pcaps'
$dumpcap = 'C:\Program Files\Wireshark\dumpcap.exe'
$capinfos = 'C:\Program Files\Wireshark\capinfos.exe'
if (-not (Test-Path -LiteralPath $dumpcap)) {
    throw 'Wireshark dumpcap.exe is not installed at the expected path.'
}

if (-not $OutputPath) {
    $OutputPath = Join-Path $dataRoot ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '.pcapng')
}
$capture = if ([IO.Path]::IsPathRooted($OutputPath)) {
    [IO.Path]::GetFullPath($OutputPath)
} else {
    [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
}
$allowedRoot = [IO.Path]::GetFullPath($dataRoot)
if (-not $capture.StartsWith($allowedRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputPath must be inside $allowedRoot"
}
if ([IO.Path]::GetExtension($capture) -ne '.pcapng') {
    throw 'OutputPath must end in .pcapng'
}
if (Test-Path -LiteralPath $capture) { throw "Capture already exists: $capture" }

$interfaces = @(& $dumpcap -D)
if ($LASTEXITCODE -ne 0) { throw 'Could not list capture interfaces.' }
$selected = @($interfaces | Where-Object { $_ -match ('^' + [regex]::Escape($Interface) + '\. ') })
if ($selected.Count -ne 1) {
    throw "Select an interface number from dumpcap -D. Received: $Interface"
}
New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($capture)) -Force | Out-Null

# Full packet bytes are needed for flow length features. The capture is bounded
# by both time and size; do not treat capture timing as an attack label.
$startedUtc = [DateTime]::UtcNow.ToString('o')
Write-Host "Capturing $DurationSeconds seconds from $($selected[0])"
Write-Host "Output: $capture"
$dumpcapLog = [IO.Path]::ChangeExtension($capture, '.dumpcap.log')
try {
    # Windows PowerShell turns native stderr progress into ErrorRecords in jobs.
    $ErrorActionPreference = 'Continue'
    & $dumpcap -i $Interface -f $CaptureFilter -a "duration:$DurationSeconds" `
        -a 'filesize:51200' -w $capture 2> $dumpcapLog
    $result = $LASTEXITCODE
} finally {
    $ErrorActionPreference = 'Stop'
}
$endedUtc = [DateTime]::UtcNow.ToString('o')
if ($result -ne 0) { throw "dumpcap exited with code $result; inspect $dumpcapLog" }
if (-not (Test-Path -LiteralPath $capture)) { throw 'No capture file was written.' }
$file = Get-Item -LiteralPath $capture
$hash = (Get-FileHash -LiteralPath $capture -Algorithm SHA256).Hash.ToLowerInvariant()
$info = @(& $capinfos -c -a -e -S $capture)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the captured PCAP.' }
$metadata = [ordered]@{
    schema_version = 1
    pcap = $file.Name
    sha256 = $hash
    bytes = $file.Length
    interface = $selected[0]
    capture_filter = $CaptureFilter
    capture_started_utc = $startedUtc
    capture_ended_utc = $endedUtc
    capinfos = $info
}
$metadataPath = [IO.Path]::ChangeExtension($capture, '.capture.json')
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metadataPath -Encoding utf8
Write-Host "Capture metadata: $metadataPath"
$info
