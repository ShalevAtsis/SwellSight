$comment = "Implemented in repo (2026-05-21). See docs/SYSTEM_ROADMAP.md progress sync."
$issues = @(
    6, 30, 39, 49, 64, 70, 71, 73, 75,
    76, 77, 78, 79, 81, 88, 90, 92, 95
)
foreach ($n in $issues) {
    gh issue close $n --repo ShalevAtsis/SwellSight --comment $comment 2>&1
    Start-Sleep -Milliseconds 250
}
