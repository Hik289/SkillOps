"""End-to-end demo of SkillOps without any external API calls.

Run::

    python examples/demo.py

The demo loads the bundled skill library (12 skills, 5 domain types), builds
the External Graph-of-Graphs, plans three tasks, and runs one Library-Time
maintenance sweep. No network calls or API keys are required.
"""
from __future__ import annotations

import json
from pathlib import Path

from skillops import (
    GraphOfGraphsPlanner,
    MaintenanceEngine,
    SkillLibrary,
)


HERE = Path(__file__).resolve().parent


def _print_plan(label: str, result) -> None:
    print(f"\n--- {label} ---")
    print(f"  match level   : {result.match_level}")
    print(f"  chosen skill  : {result.chosen_skill_id}")
    print(f"  maintenance   : {result.maintenance_actions_applied}")
    print(f"  validator     : {result.validator_issues}")
    print(f"  plan ({len(result.plan)}):")
    for i, a in enumerate(result.plan):
        print(f"    {i+1}. {a.to_str()}")


def main() -> int:
    library = SkillLibrary.load_directory(HERE / "library")
    edge_counts = library.build_edges()
    print("=" * 64)
    print("SkillOps demo")
    print("=" * 64)
    print(f"library size  : {len(library)} skills")
    print(f"edge counts   : {edge_counts}")
    print(f"domain types  : {sorted({s.domain_type for s in library.all()})}")

    planner = GraphOfGraphsPlanner(library)

    # --- Task A: exact-signature match in the library ---
    target = library.skills["sk_004_clean_then_place_apple"]
    task_a = {
        "task_id": "demoA",
        "description": "place a clean apple in the fridge",
        "domain_type": "clean_then_place",
        "object": "apple",
        "parent": "fridge",
        "signature": target.signature(),
    }
    _print_plan("Task A (exact-signature match)", planner.plan(task_a))

    # --- Task B: domain neighbour (object/parent unknown to library) ---
    task_b = {
        "task_id": "demoB",
        "description": "place a candle on the desk",
        "domain_type": "place_in_container",
        "object": "candle",
        "parent": "desk",
    }
    _print_plan("Task B (domain neighbour + template instantiate)", planner.plan(task_b))

    # --- Task C: no library coverage at all (planner returns empty plan) ---
    task_c = {
        "task_id": "demoC",
        "description": "compute something abstract",
        "domain_type": "compute",
        "object": "vector",
        "parent": "memory",
    }
    _print_plan("Task C (no coverage; LLM fallback disabled in demo)", planner.plan(task_c))

    # --- Library-Time maintenance sweep ---
    print("\n" + "=" * 64)
    print("Library-Time maintenance sweep")
    print("=" * 64)
    engine = MaintenanceEngine(library)
    report = engine.sweep()
    print(json.dumps(report.to_dict(), indent=2))
    print(f"library size after sweep: {len(library)} skills")
    print(f"edge counts after sweep : "
          f"{ {t: sum(1 for e in library.edges if e.edge_type == t) for t in ('dependency','compatibility','redundancy','alternative','lineage')} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
