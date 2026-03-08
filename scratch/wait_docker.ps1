$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
Write-Host "Waiting for Docker daemon (up to 120s)..."
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 5
    $out = & $docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Ready after $(($i+1)*5)s."
        exit 0
    }
    Write-Host "  $(($i+1)*5)s - not ready yet..."
}
Write-Host "Timed out waiting for Docker."
exit 1
