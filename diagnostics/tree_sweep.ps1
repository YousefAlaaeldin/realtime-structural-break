# Tree-settings sweep, screen stage (HANDOFF section 44.2). One change at a time from the baseline
# (leaves 31, min-leaf 2000, lr 0.03, ff 0.5, rounds 1200), seed 0, paired vs the current best trees.
# Run from the project folder:   .\diagnostics\tree_sweep.ps1
# Sequential (one GPU job at a time); a config whose summary file already exists is skipped, so a stopped sweep
# resumes where it left off. Each run prints its PAIRED line; the summaries are cv_out\summary_<tag>.txt.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$base = "--gpu --threads 6 --rounds 1200 --aug 2 --ev compact --vol --dist full --drop dist_z_ --bocpd --nbm " +
        "--nbm-batch 24 --nbm-epochs 25 --compare cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz"
$cfgs = [ordered]@{
    "L63"    = "--leaves 63 --min-leaf 2000 --lr 0.03 --ff 0.5"
    "L15"    = "--leaves 15 --min-leaf 2000 --lr 0.03 --ff 0.5"
    "ML500"  = "--leaves 31 --min-leaf 500 --lr 0.03 --ff 0.5"
    "ML8000" = "--leaves 31 --min-leaf 8000 --lr 0.03 --ff 0.5"
    "FF03"   = "--leaves 31 --min-leaf 2000 --lr 0.03 --ff 0.3"
    "FF08"   = "--leaves 31 --min-leaf 2000 --lr 0.03 --ff 0.8"
    "LR015"  = "--leaves 31 --min-leaf 2000 --lr 0.015 --ff 0.5"
}
foreach ($name in $cfgs.Keys) {
    $tag = "aug2_df_bo_nbm_bs24_ep25_$name"
    if (Test-Path "cv_out\summary_$tag.txt") {
        Write-Host "skip $name (cv_out\summary_$tag.txt exists)"
        continue
    }
    Write-Host "=== $name : $($cfgs[$name])"
    $a = @("cv.py") + ($cfgs[$name] -split " ") + ($base -split " ") + @("--tag", $tag)
    & python @a
    if ($LASTEXITCODE -ne 0) { Write-Host "$name failed (exit $LASTEXITCODE); stopping"; break }
}
Write-Host "`nPAIRED lines of all configs:"
foreach ($name in $cfgs.Keys) {
    $f = "cv_out\summary_aug2_df_bo_nbm_bs24_ep25_$name.txt"
    if (Test-Path $f) { Write-Host "$name : $((Select-String -Path $f -Pattern 'PAIRED').Line)" }
}
