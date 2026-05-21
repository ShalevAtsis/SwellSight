# Close GitHub issues implemented in repo (sync with docs/SYSTEM_ROADMAP.md)
$comment = @"
Closed per roadmap progress sync (2026-05-21).

Implementation verified in repo. See docs/SYSTEM_ROADMAP.md — Progress summary and Section 8 task status.
"@

$toClose = @(
    5, 7
    8..25
    26..29
    31, 32
    33..35
    37..48
    50..60
    62..63
    65..69
    72
    # 64 omitted — forgot-password still open
    79
)

foreach ($n in $toClose) {
    Write-Host "Closing #$n..."
    gh issue close $n --repo ShalevAtsis/SwellSight --comment $comment 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Warning "Failed #$n" }
    Start-Sleep -Milliseconds 300
}

Write-Host "Done. Open issues: 6,30,36,49,61,64,70-71,73-75,76-78,80-99,100"
