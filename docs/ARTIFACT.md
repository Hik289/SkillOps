# Artifact Guide

Operational notes for reproducing `SkillOps` from the public `SkillOps` repository.

## Review Path

- `skillops/`: Project-specific implementation subtree.
- `examples/`: Small runnable examples and smoke-test entry points.
- `tests/`: Local tests or smoke checks for fresh checkouts.
- Root-level entry points: `run_skillops.py`.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `pyproject.toml`: Package metadata and optional extras when available.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
python -m pytest tests -q
python tests/test_smoke.py
```

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `python run_skillops.py`

## Figure Assets

- `1.jpg`

## Data And Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
