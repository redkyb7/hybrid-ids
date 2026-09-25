param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'scan', 'botnet', 'ssh_bruteforce', 'web_login', 'web_sqli', 'web_xss',
        'dos_http', 'dos_slow', 'dos_syn', 'infiltration',
        'ddos_http_loic', 'ddos_http_hoic', 'ddos_udp'
    )]
    [string]$Attack,
    [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$campaignId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$isDdos = $Attack.StartsWith('ddos_')
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

    docker compose build --quiet attacker
    if ($LASTEXITCODE -ne 0) { throw 'Attacker image build failed.' }
    Write-Host "Campaign: $campaignId ($Attack)"

    if ($isDdos) {
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
        docker compose run --rm --no-deps -T --entrypoint python attacker /app/attack_campaigns.py `
            --attack $Attack --campaign-id $campaignId --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "Campaign $Attack failed; inspect its manifest." }
    }

    Start-Sleep -Seconds 3
    Write-Host "Manifests: data/campaigns/$campaignId-*.json"
    Write-Host "Evaluate: uv tool run --from duckdb python scripts/evaluate_attack_campaigns.py --campaign-id $campaignId"
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
