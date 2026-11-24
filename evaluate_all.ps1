
$ErrorActionPreference = "Stop"

$configs = @(
    @("--out_dir=out-wikitext2-mha", "--dataset=wikitext2", "--kv_cache", "--save_json", "--benchmark_inference", "--prompt_length=128"),
    @("--out_dir=out-wikitext2-mqa", "--dataset=wikitext2", "--kv_cache", "--save_json", "--benchmark_inference", "--prompt_length=128"),
    @("--out_dir=out-bookcorpus-mha", "--dataset=bookcorpus", "--kv_cache", "--save_json", "--benchmark_inference", "--prompt_length=384"),
    @("--out_dir=out-bookcorpus-mqa", "--dataset=bookcorpus", "--kv_cache", "--save_json", "--benchmark_inference", "--prompt_length=384")
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
