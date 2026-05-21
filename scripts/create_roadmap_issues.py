#!/usr/bin/env python3
"""Create GitHub issues from docs/SYSTEM_ROADMAP.md task IDs."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

REPO = "ShalevAtsis/SwellSight"
ROADMAP_URL = f"https://github.com/{REPO}/blob/main/docs/SYSTEM_ROADMAP.md"


@dataclass
class Task:
    task_id: str
    title: str
    details: str
    phase: str
    milestone: str
    section: str
    closed: bool = False


MILESTONES = [
    ("P0: Foundation", "Stable model path, docs, CI basics", 0),
    ("P1: ML platform", "Fully repeatable AI training and inference", 1),
    ("P2: MLOps", "Versioned datasets and model registry", 2),
    ("P3: Backend platform", "API, worker, DB, surf score v1", 3),
    ("P4: Web product", "Surfer UI and E2E staging demo", 4),
    ("P5: Production", "Deploy, monitor, scale", 5),
]


def gh(*args: str, input_json: Optional[dict] = None) -> dict:
    cmd = ["gh", *args, "--repo", REPO]
    if input_json is not None:
        cmd.extend(["--input", "-"])
    result = subprocess.run(
        cmd,
        input=json.dumps(input_json) if input_json else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh failed: {' '.join(cmd)}\n{result.stderr}")
    if result.stdout.strip():
        return json.loads(result.stdout)
    return {}


def ensure_label(name: str, color: str, description: str) -> None:
    subprocess.run(
        ["gh", "label", "create", name, "--color", color, "--description", description, "--repo", REPO],
        capture_output=True,
        text=True,
    )


def ensure_milestone(title: str, description: str) -> None:
    existing = subprocess.run(
        ["gh", "api", f"repos/{REPO}/milestones", "--jq", ".[].title"],
        capture_output=True,
        text=True,
    )
    if title in (existing.stdout or ""):
        return
    subprocess.run(
        [
            "gh",
            "api",
            f"repos/{REPO}/milestones",
            "-X",
            "POST",
            "-f",
            f"title={title}",
            "-f",
            f"description={description}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def issue_body(task: Task) -> str:
    status = "Done (pre-roadmap cleanup)" if task.closed else "Open"
    return f"""## Task ID
`{task.task_id}`

## Phase
{task.phase} — {task.section}

## Status
{status}

## Description
{task.details}

## Acceptance criteria
- [ ] Implementation complete
- [ ] Tests or manual verification documented
- [ ] Referenced in docs if user-facing

## References
- [SYSTEM_ROADMAP.md — Section 8]({ROADMAP_URL}#8-task-checklist-by-phase)
- Task ID: **{task.task_id}**

---
*Auto-created from SwellSight roadmap. Work order: P0 → P1 → P2 → P3 → P4 → P5.*
"""


def create_issue(task: Task) -> int:
    labels = [f"phase:{task.phase}"]
    if task.closed:
        labels.append("status:done")

    result = subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            REPO,
            "--title",
            f"[{task.task_id}] {task.title}",
            "--body",
            issue_body(task),
            "--milestone",
            task.milestone,
            *sum([["--label", label] for label in labels], []),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # Output: https://github.com/.../issues/123
    url = result.stdout.strip()
    issue_num = int(url.rstrip("/").split("/")[-1])
    if task.closed:
        subprocess.run(
            ["gh", "issue", "close", str(issue_num), "--repo", REPO, "--comment", "Completed in foundation cleanup."],
            check=True,
            capture_output=True,
        )
    return issue_num


def all_tasks() -> List[Task]:
    tasks: List[Task] = []

    # P0
    p0 = "P0: Foundation"
    for tid, title, details, done in [
        ("P0-T01", "Unify WaveAnalysisModel with inference analyzer", "Shared checkpoint between training and DINOv2WaveAnalyzer.", True),
        ("P0-T02", "Fix YAML ConfigManager and _base_ inheritance", "training.yaml overrides apply correctly.", True),
        ("P0-T03", "Implement scripts/inference.py and swellsight CLI", "Working inference and CLI delegation to scripts.", True),
        ("P0-T04", "Add MODEL_GUIDE.md", "Single reference for train/inference/requirements.", True),
        ("P0-T05", "Verify pip install -e . on Windows and Linux", "Document in MODEL_GUIDE; fix packaging if needed.", False),
        ("P0-T06", "Fix remaining test collection failures", "Remove sys.path hacks; rely on conftest.", False),
        ("P0-T07", "Add GitHub Actions lint and unit tests on PR", "CI workflow for pytest + lint.", False),
    ]:
        tasks.append(Task(tid, title, details, "P0", p0, "Foundation", done))

    # P1
    p1 = "P1: ML platform"
    p1_tasks = [
        ("P1-T01", "Refactor extract_depth_maps.py for local runs", "Remove Colab/Drive; CLI --input, --output, --gpu.", "P1.A Data & depth"),
        ("P1-T02", "Standardize dataset layout", "Document and validate data/raw, depth_maps, processed.", "P1.A Data & depth"),
        ("P1-T03", "Depth quality gate before training", "Reuse quality_validation / data_validator.", "P1.A Data & depth"),
        ("P1-T04", "Dataset manifest", "datasets/manifest.json with paths, labels, split, version.", "P1.A Data & depth"),
        ("P1-T05", "Integrate real depth in WaveDataset", "Load _depth.npy when present.", "P1.A Data & depth"),
        ("P1-T06", "Wire MultiTaskLoss in trainer", "Replace inline MSE/CE; config loss weights.", "P1.B Training"),
        ("P1-T07", "Wire LR scheduler in trainer", "create_lr_scheduler + warmup per config.", "P1.B Training"),
        ("P1-T08", "Sim-to-real trainer mode", "Synthetic pretrain then real finetune in one CLI.", "P1.B Training"),
        ("P1-T09", "Training callbacks", "Early stopping; TensorBoard/W&B optional.", "P1.B Training"),
        ("P1-T10", "Export best checkpoint and metrics.json", "checkpoints/best_model.pth always written.", "P1.B Training"),
        ("P1-T11", "Evaluation gate script", "Fail CI if MAE/accuracy below thresholds.", "P1.B Training"),
        ("P1-T12", "Pipeline checkpoint from env/config", "SWELLSIGHT_CHECKPOINT + inference.yaml.", "P1.C Inference"),
        ("P1-T13", "Batch inference function for worker", "List of images to results dict.", "P1.C Inference"),
        ("P1-T14", "Model warmup on worker start", "Avoid cold-start timeout.", "P1.C Inference"),
        ("P1-T15", "CPU/GPU fallback tests and docs", "Document limits in MODEL_GUIDE.", "P1.C Inference"),
        ("P1-T16", "Refactor generate_synthetic_data.py for local", "HF token via env; no google.colab.", "P1.D Synthetic"),
        ("P1-T17", "Synthetic job config YAML", "Prompts, count, controlnet scale.", "P1.D Synthetic"),
        ("P1-T18", "Auto-label from depth geometry", "Tie into synthetic_generator labels.", "P1.D Synthetic"),
    ]
    for tid, title, details, section in p1_tasks:
        tasks.append(Task(tid, title, details, "P1", p1, section))

    # P2
    p2 = "P2: MLOps"
    for tid, title, details in [
        ("P2-T01", "Define models/registry.yaml", "version, path, metrics, data_manifest_id, git_sha."),
        ("P2-T02", "Script promote_model.py", "Copy checkpoint and update registry."),
        ("P2-T03", "Pin dependency versions", "Lock requirements for training/inference."),
        ("P2-T04", "Experiment logging", "MLflow or JSON experiment log."),
        ("P2-T05", "Dataset versioning", "DVC or manifest hash in registry."),
        ("P2-T06", "Automated train smoke in CI", "Dummy data, 1 epoch; optional GPU runner."),
        ("P2-T07", "Model card per version", "docs/models/vX.md with limits and metrics."),
    ]:
        tasks.append(Task(tid, title, details, "P2", p2, "MLOps"))

    # P3
    p3 = "P3: Backend platform"
    p3_tasks = [
        ("P3-T01", "SQLAlchemy 2.0 + Alembic setup", "ORM and migrations from day one.", "P3.A Database"),
        ("P3-T02", "Schema: users", "id, email, password_hash, created_at.", "P3.A Database"),
        ("P3-T03", "Schema: analyses", "user_id, status, storage, result_json, score.", "P3.A Database"),
        ("P3-T04", "Schema: spots (optional v1)", "name, lat/lon for future beach cams.", "P3.A Database"),
        ("P3-T05", "Schema: model_versions", "Registry sync for audit.", "P3.A Database"),
        ("P3-T06", "Seed and migration scripts", "alembic upgrade head works.", "P3.A Database"),
        ("P3-T07", "Register / login / refresh endpoints", "bcrypt or argon2 passwords.", "P3.B Auth"),
        ("P3-T08", "JWT middleware on protected routes", "Secure analysis endpoints.", "P3.B Auth"),
        ("P3-T09", "Rate limiting per user/IP", "Redis sliding window.", "P3.B Auth"),
        ("P3-T10", "Upload validation", "Max size, MIME, dimensions.", "P3.B Auth"),
        ("P3-T11", "CORS for web origins", "Staging and prod URLs.", "P3.B Auth"),
        ("P3-T12", "POST /v1/analyses", "Multipart upload returns job_id.", "P3.C API"),
        ("P3-T13", "GET /v1/analyses/{id}", "Status and result when complete.", "P3.C API"),
        ("P3-T14", "GET /v1/analyses history", "Paginated user history.", "P3.C API"),
        ("P3-T15", "GET /v1/health", "API, DB, queue depth.", "P3.C API"),
        ("P3-T16", "Publish OpenAPI and TS client", "Generate frontend client.", "P3.C API"),
        ("P3-T17", "Refactor endpoints.py pipeline access", "Use app.state; remove frame hack.", "P3.C API"),
        ("P3-T18", "Idempotency key on upload", "Optional duplicate prevention.", "P3.C API"),
        ("P3-T19", "Redis queue module", "swellsight/jobs/ package.", "P3.D Worker"),
        ("P3-T20", "Worker entrypoint scripts/worker.py", "Dequeue, pipeline, save.", "P3.D Worker"),
        ("P3-T21", "Job state machine", "pending → processing → completed/failed.", "P3.D Worker"),
        ("P3-T22", "Retry and dead letter queue", "3 retries; log failure.", "P3.D Worker"),
        ("P3-T23", "Object storage for artifacts", "Pre-signed URLs for downloads.", "P3.D Worker"),
        ("P3-T24", "Wire ModelServer checkpoint from env", "Production model version.", "P3.D Worker"),
        ("P3-T25", "Define surf score spec (0-100)", "Inputs: height, direction, breaking, confidence.", "P3.E Surf score"),
        ("P3-T26", "Implement SurfScoreEngine", "src/swellsight/scoring/ weighted formula.", "P3.E Surf score"),
        ("P3-T27", "Unit tests for score monotonicity", "Sanity checks on formula.", "P3.E Surf score"),
        ("P3-T28", "Expose surf_score in API response", "surf_score and score_breakdown fields.", "P3.E Surf score"),
        ("P3-T29", "Backlog: learned surf score", "Regressor on user ratings (v2).", "P3.E Surf score"),
    ]
    for tid, title, details, section in p3_tasks:
        tasks.append(Task(tid, title, details, "P3", p3, section))

    # P4
    p4 = "P4: Web product"
    p4_tasks = [
        ("P4-T01", "Create web/ Next.js app", "TypeScript, App Router, monorepo.", "P4.A App shell"),
        ("P4-T02", "Design system", "Tailwind tokens, surf brand.", "P4.A App shell"),
        ("P4-T03", "Auth pages", "Login, register, forgot password.", "P4.A App shell"),
        ("P4-T04", "API client from OpenAPI", "Generated hooks or fetch wrapper.", "P4.A App shell"),
        ("P4-T05", "Landing page", "Value prop and CTA.", "P4.B Core flows"),
        ("P4-T06", "Upload flow", "Drag-drop and mobile camera.", "P4.B Core flows"),
        ("P4-T07", "Progress UI", "Poll or WebSocket while analyzing.", "P4.B Core flows"),
        ("P4-T08", "Results page with surf score gauge", "Height, direction, breaking, score.", "P4.B Core flows"),
        ("P4-T09", "Analysis history list", "Thumbnails and past scores.", "P4.B Core flows"),
        ("P4-T10", "Error states UI", "Low quality, timeout, warnings.", "P4.B Core flows"),
        ("P4-T11", "Score breakdown tooltips", "Explain what affects score.", "P4.C UX"),
        ("P4-T12", "Share result (optional)", "Link or image export.", "P4.C UX"),
        ("P4-T13", "i18n backlog", "English first.", "P4.C UX"),
        ("P4-T14", "Accessibility basics", "WCAG, keyboard nav.", "P4.C UX"),
    ]
    for tid, title, details, section in p4_tasks:
        tasks.append(Task(tid, title, details, "P4", p4, section))

    # P5
    p5 = "P5: Production"
    p5_tasks = [
        ("P5-T01", "Dockerfile.api", "Slim Python, no GPU.", "P5.A Containers"),
        ("P5-T02", "Dockerfile.worker", "CUDA base and model cache volume.", "P5.A Containers"),
        ("P5-T03", "docker-compose.yml", "api, worker, postgres, redis, minio.", "P5.A Containers"),
        ("P5-T04", "Env config and .env.example", "Secrets via platform.", "P5.A Containers"),
        ("P5-T05", "Staging environment", "Parity with prod, smaller GPU.", "P5.A Containers"),
        ("P5-T06", "CI test and lint on PR", "pytest and linters.", "P5.B CI/CD"),
        ("P5-T07", "CI build images on main", "Tag sha and semver.", "P5.B CI/CD"),
        ("P5-T08", "CD deploy staging auto", "On merge to main.", "P5.B CI/CD"),
        ("P5-T09", "CD deploy prod manual approve", "Release workflow.", "P5.B CI/CD"),
        ("P5-T10", "DB migrations in deploy", "Alembic job in pipeline.", "P5.B CI/CD"),
        ("P5-T11", "TLS everywhere", "LB terminates HTTPS.", "P5.C Security"),
        ("P5-T12", "Secrets rotation doc", "HF, DB, JWT secrets.", "P5.C Security"),
        ("P5-T13", "Image retention policy", "Delete raw after N days (7).", "P5.C Security"),
        ("P5-T14", "Privacy policy and ToS", "User beach photo uploads.", "P5.C Security"),
        ("P5-T15", "Dependency scanning", "Dependabot or Snyk.", "P5.C Security"),
        ("P5-T16", "Structured JSON logging", "Correlation id per job.", "P5.D Observability"),
        ("P5-T17", "Prometheus metrics", "Latency, queue, GPU util.", "P5.D Observability"),
        ("P5-T18", "Grafana dashboards", "API and worker panels.", "P5.D Observability"),
        ("P5-T19", "Alerts", "Failed jobs, backlog, error rate.", "P5.D Observability"),
        ("P5-T20", "Ops runbook", "docs/ops/RUNBOOK.md deploy and rollback.", "P5.D Observability"),
        ("P5-T21", "Horizontal API replicas", "Stateless scaling.", "P5.E Scale"),
        ("P5-T22", "Multiple GPU workers", "Queue consumer pool.", "P5.E Scale"),
        ("P5-T23", "CDN for static web", "Cache Next.js assets.", "P5.E Scale"),
        ("P5-T24", "Cost monitoring", "GPU hours per analysis.", "P5.E Scale"),
    ]
    for tid, title, details, section in p5_tasks:
        tasks.append(Task(tid, title, details, "P5", p5, section))

    return tasks


def main() -> int:
    print(f"Creating milestones and issues for {REPO}...")
    colors = {"P0": "0E8A16", "P1": "1D76DB", "P2": "5319E7", "P3": "D93F0B", "P4": "FBCA04", "P5": "B60205"}
    for phase, color in colors.items():
        ensure_label(f"phase:{phase}", color, f"Roadmap phase {phase}")

    ensure_label("status:done", "EDEDED", "Completed in foundation work")

    for title, desc, _ in MILESTONES:
        ensure_milestone(title, desc)
        time.sleep(0.3)

    created = []
    errors = []
    for task in all_tasks():
        try:
            num = create_issue(task)
            created.append((task.task_id, num))
            print(f"  OK {task.task_id} -> #{num}")
            time.sleep(0.5)
        except Exception as exc:
            errors.append((task.task_id, str(exc)))
            print(f"  FAIL {task.task_id}: {exc}")

    print(f"\nCreated {len(created)} issues, {len(errors)} failures.")
    if errors:
        for tid, err in errors:
            print(f"  {tid}: {err}")
        return 1

    # Epic tracking issue
    subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            REPO,
            "--title",
            "[EPIC] SwellSight product roadmap P0→P5",
            "--body",
            f"""## Overview
Track the full SwellSight delivery plan: ML platform → MLOps → backend → web UI → production.

**Work in order:** P0 → P1 → P2 → P3 → P4 → P5

## Documentation
- [SYSTEM_ROADMAP.md]({ROADMAP_URL})

## Phase milestones
- [ ] P0: Foundation
- [ ] P1: ML platform ({sum(1 for t in all_tasks() if t.phase == 'P1')} tasks)
- [ ] P2: MLOps
- [ ] P3: Backend platform
- [ ] P4: Web product
- [ ] P5: Production

Filter issues by label: `phase:P1`, `phase:P2`, etc.
""",
            "--label",
            "phase:P0",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
