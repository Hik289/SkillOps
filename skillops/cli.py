"""Tiny CLI: load a library, plan one task, print the result.

Example
-------
::

    skillops --task "place a clean apple in the fridge" --library examples/library/

The CLI parses the natural-language task into a minimal task dict and runs the
planner. For richer integrations, import :class:`GraphOfGraphsPlanner` directly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .planner import GraphOfGraphsPlanner, PlannerConfig
from .skill_graph import SkillLibrary


# ---------------------------------------------------------------------------
# A very small task parser. Real users should construct task dicts directly.
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "place_in_container": ("place", "put", "store", "fridge", "shelf"),
    "heat_then_place": ("heat", "warm", "microwave"),
    "cool_then_place": ("cool", "chill", "fridge"),
    "clean_then_place": ("clean", "wash", "rinse"),
    "fetch_object": ("fetch", "get", "bring"),
    "look_at_object": ("look", "inspect", "examine"),
}


def _parse_task(text: str) -> Dict[str, Any]:
    """Best-effort heuristic parser: identify domain_type + key entities."""
    lower = text.lower()
    domain_type = "fetch_object"
    for dt, kws in DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            domain_type = dt
            break

    nouns = re.findall(r"\b([a-z]+)\b", lower)
    stop = {
        "a", "an", "the", "in", "into", "on", "to", "from", "of", "and", "or",
        "with", "place", "put", "fetch", "get", "bring", "store", "look", "at",
        "heat", "warm", "cool", "chill", "clean", "wash", "rinse", "inspect",
        "examine", "then",
    }
    content_nouns = [n for n in nouns if n not in stop and len(n) > 2]
    obj = content_nouns[0] if content_nouns else ""
    parent = content_nouns[1] if len(content_nouns) > 1 else ""

    task: Dict[str, Any] = {
        "task_id": f"cli_task_{abs(hash(text)) % 10**6}",
        "description": text,
        "domain_type": domain_type,
        "object": obj,
        "parent": parent,
    }
    return task


def _emit_plan(plan_actions: List[Any]) -> List[Dict[str, Any]]:
    return [{"name": a.name, "args": list(a.args)} for a in plan_actions]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="skillops", description="Run the SkillOps planner on one task.")
    ap.add_argument("--task", required=True, help="natural-language task description")
    ap.add_argument("--library", required=True, help="path to a directory or *.json library file")
    ap.add_argument("--task-json", default=None,
                    help="optional path to a JSON file with a richer task dict; overrides --task parsing")
    ap.add_argument("--no-llm-fallback", action="store_true",
                    help="disable the LLM fallback even if an OPENAI_API_KEY is present")
    ap.add_argument("--json-out", action="store_true",
                    help="emit JSON instead of human-readable text")
    args = ap.parse_args(argv)

    lib_path = Path(args.library)
    if lib_path.is_dir():
        library = SkillLibrary.load_directory(lib_path)
    elif lib_path.is_file():
        library = SkillLibrary.load(lib_path)
    else:
        print(f"library not found: {lib_path}", file=sys.stderr)
        return 2
    library.build_edges()

    if args.task_json:
        task = json.loads(Path(args.task_json).read_text())
    else:
        task = _parse_task(args.task)

    config = PlannerConfig(use_llm_fallback=(not args.no_llm_fallback))
    planner = GraphOfGraphsPlanner(library, config=config)
    result = planner.plan(task)

    payload = {
        "task": task,
        "plan": _emit_plan(result.plan),
        "chosen_skill_id": result.chosen_skill_id,
        "match_level": result.match_level,
        "maintenance_actions_applied": result.maintenance_actions_applied,
        "validator_issues": result.validator_issues,
        "n_llm_calls": result.n_llm_calls,
        "cost_usd": result.cost_usd,
        "latency_ms": round(result.latency_ms, 2),
        "library_size": len(library),
        "library_edges": len(library.edges),
    }

    if args.json_out:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("=" * 64)
        print("SkillOps - one-task plan")
        print("=" * 64)
        print(f"task description : {task['description']!r}")
        print(f"parsed signature : domain_type={task.get('domain_type')!r}  "
              f"object={task.get('object')!r}  parent={task.get('parent')!r}")
        print(f"library          : {len(library)} skills, {len(library.edges)} edges")
        print(f"match level      : {result.match_level}")
        print(f"chosen skill     : {result.chosen_skill_id}")
        print(f"maintenance      : {result.maintenance_actions_applied}")
        print(f"validator issues : {result.validator_issues}")
        print(f"latency          : {result.latency_ms:.2f} ms")
        print(f"plan ({len(result.plan)} actions):")
        for i, a in enumerate(result.plan):
            print(f"  {i+1}. {a.to_str()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
