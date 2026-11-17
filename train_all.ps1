$ErrorActionPreference = "Stop"

$configs = @(
    "config/train_wikitext.py",
    "config/train_wikitext_mqa.py",
    "config/train_bookcorpus.py",
    "config/train_bookcorpus_mqa.py"
)

foreach ($config in $configs) {
    Write-Host "`n====================================" -ForegroundColor Cyan
    Write-Host "Starting training: $config" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan

    python -u train.py $config

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed: $config" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`nAll four training runs completed!" -ForegroundColor Green