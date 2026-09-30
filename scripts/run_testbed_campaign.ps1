param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'scan', 'scan_connect', 'scan_sparse',
        'botnet', 'botnet_fast', 'botnet_slow',
        'ssh_bruteforce', 'ssh_bruteforce_fast', 'ssh_bruteforce_slow',
        'web_login', 'web_sqli', 'web_xss',
        'dos_http', 'dos_slow', 'benign_slow_http', 'dos_syn', 'infiltration',
        'ddos_http_loic', 'ddos_http_hoic', 'ddos_udp',
        'benign_http', 'benign_http_api', 'benign_http_burst'
    )]
    [string]$Attack,
    [int]$Seed = 42,
    [ValidateRange(2,25)][double]$SlowDuration = 20,
    [ValidateRange(0.2,2)][double]$SlowInterval = 0.6,
    [ValidateRange(1,8)][int]$SlowConnections = 4,
    [ValidateRange(0,128)][int]$SlowPadding = 0,
    [ValidateRange(0,0.2)][double]$SlowJitter = 0,
    [switch]$PassThru
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$campaignId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$isDdos = $Attack.StartsWith('ddos_')
$isBenign = $Attack.StartsWith('benign_http')
$restoreAttacker = $false
$restoreBenign = $false
$workersStarted = $false

Push-Location -LiteralPath $repoRoot
try {
    $running = @(docker compose ps --services --status running)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect Docker Compose services.' }
    if ('victim' -notin $running -or 'monitor' -notin $running) {
        throw 'Start the victim and monitor first: docker compose up -d --build'
    }
    $restoreAttacker = 'attacker' -in $running
    $restoreBenign = 'benign_client' -in $running
    if ($restoreAttacker) {
        docker compose stop attacker
        if ($LASTEXITCODE -ne 0) { throw 'Could not pause the continuous attacker.' }
    }
    if ($restoreBenign) {
        docker compose stop benign_client
        if ($LASTEXITCODE -ne 0) { throw 'Could not pause the benign generator.' }
    }

    if ($isBenign) {
        docker compose build --quiet benign_client
        if ($LASTEXITCODE -ne 0) { throw 'Benign client image build failed.' }
    }
    else {
        docker compose build --quiet attacker
        if ($LASTEXITCODE -ne 0) { throw 'Attacker image build failed.' }
    }
    Write-Host "Campaign: $campaignId ($Attack)"

    if ($isBenign) {
        $scenario = 'mixed'
        if ($Attack -eq 'benign_http_api') { $scenario = 'api' }
        if ($Attack -eq 'benign_http_burst') { $scenario = 'burst' }
        docker compose run --rm --no-deps -T --entrypoint python benign_client /app/benign_campaign.py `
            --campaign-id $campaignId --seed $Seed --scenario $scenario
        if ($LASTEXITCODE -ne 0) { throw "Benign campaign $campaignId failed; inspect its manifest." }
    }
    elseif ($isDdos) {
        $oldId = $env:IDS_CAMPAIGN_ID
        $oldMode = $env:IDS_DDOS_MODE
        $oldStart = $env:IDS_CAMPAIGN_START_EPOCH
        $oldSeed1 = $env:IDS_DDOS_SEED_1
        $oldSeed2 = $env:IDS_DDOS_SEED_2
        try {
            $env:IDS_CAMPAIGN_ID = $campaignId
            $env:IDS_DDOS_MODE = $Attack
            $env:IDS_CAMPAIGN_START_EPOCH = [string]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + 8)
            $env:IDS_DDOS_SEED_1 = [string]$Seed
            $env:IDS_DDOS_SEED_2 = [string]($Seed + 1)
            $workersStarted = $true
            docker compose --profile ddos up -d --no-deps --no-build --force-recreate ddos_worker_1 ddos_worker_2
            if ($LASTEXITCODE -ne 0) { throw 'Could not start the two DDoS workers.' }
            $workerCodes = @(docker wait ids-ddos-worker-1 ids-ddos-worker-2)
            if ($LASTEXITCODE -ne 0 -or $workerCodes.Count -ne 2 -or
                [int]$workerCodes[0] -ne 0 -or [int]$workerCodes[1] -ne 0) {
                docker compose --profile ddos logs --tail 30 ddos_worker_1 ddos_worker_2
                throw "DDoS worker exit codes: $($workerCodes -join ', ')"
            }
        }
        finally {
            $env:IDS_CAMPAIGN_ID = $oldId
            $env:IDS_DDOS_MODE = $oldMode
            $env:IDS_CAMPAIGN_START_EPOCH = $oldStart
            $env:IDS_DDOS_SEED_1 = $oldSeed1
            $env:IDS_DDOS_SEED_2 = $oldSeed2
        }
    }
    else {
        $slowArguments = @()
        if ($Attack -in @('dos_slow', 'benign_slow_http')) {
            $culture = [Globalization.CultureInfo]::InvariantCulture
            $slowArguments = @('--slow-duration', $SlowDuration.ToString($culture),
                '--slow-interval', $SlowInterval.ToString($culture),
                '--slow-connections', "$SlowConnections", '--slow-padding', "$SlowPadding",
                '--slow-jitter', $SlowJitter.ToString($culture))
        }
        docker compose run --rm --no-deps -T --entrypoint python attacker /app/attack_campaigns.py `
            --attack $Attack --campaign-id $campaignId --seed $Seed @slowArguments
        if ($LASTEXITCODE -ne 0) { throw "Campaign $Attack failed; inspect its manifest." }
    }

    Start-Sleep -Seconds 3
    Write-Host "Manifests: data/campaigns/$campaignId-*.json"
}
finally {
    if ($workersStarted) {
        docker compose --profile ddos stop ddos_worker_1 ddos_worker_2 | Out-Null
    }
    if ($restoreAttacker) {
        docker compose up -d --no-deps --no-build --force-recreate attacker | Out-Null
    }
    if ($restoreBenign) {
        docker compose start benign_client | Out-Null
    }
    Pop-Location
}

if ($PassThru) { Write-Output $campaignId }
