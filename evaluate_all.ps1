
$ErrorActionPreference = "Stop"

$configs = @(
    @("--out_dir=out-wikitext2-mha", "--dataset=wikitext2", "--kv_cache", "--save_json", "--prompt_length=128", "--gen_tokens=127"),
    @("--out_dir=out-wikitext2-mqa", "--dataset=wikitext2", "--kv_cache", "--save_json", "--prompt_length=128", "--gen_tokens=127"),
    @("--out_dir=out-bookcorpus-mha", "--dataset=bookcorpus", "--kv_cache", "--save_json", "--prompt_length=256", "--gen_tokens=255"),
    @("--out_dir=out-bookcorpus-mqa", "--dataset=bookcorpus", "--kv_cache", "--save_json", "--prompt_length=256", "--gen_tokens=255")
)

foreach ($config in $configs) {
    Write-Host "`n====================================" -ForegroundColor Cyan
    Write-Host "Starting evaluation: $($config[0])" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan

    python ./evaluate.py @config

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Evaluation failed: $($config[0])" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`nAll evaluations completed!" -ForegroundColor Green
