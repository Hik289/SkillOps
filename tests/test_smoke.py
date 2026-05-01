"""End-to-end smoke tests. None of these tests touch the network."""
from __future__ import annotations

from pathlib import Path

import pytest

from skillops import (
    Action,
    GraphOfGraphsPlanner,
    MaintenanceEngine,
    Skill,
    SkillContract,
    SkillLibrary,
)
from skillops import maintenance as M


EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "library"


def _make_library() -> SkillLibrary:
    library = SkillLibrary.load_directory(EXAMPLES)
    library.build_edges()
    return library


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #


def test_library_loads_all_examples():
    lib = _make_library()
    assert len(lib) == 12
    domains = {s.domain_type for s in lib.all()}
    assert domains >= {
        "place_in_container", "clean_then_place", "heat_then_place",
        "cool_then_place", "fetch_object", "look_at_object",
    }


def test_edges_built():
    lib = _make_library()
    cnt = {t: sum(1 for e in lib.edges if e.edge_type == t)
           for t in ("dependency", "compatibility", "redundancy", "alternative", "lineage")}
    # at least one redundancy (synthetic clone) and lineage edge
    assert cnt["redundancy"] >= 1
    assert cnt["lineage"] >= 1


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


def test_planner_exact_match_returns_plan():
    lib = _make_library()
    p = GraphOfGraphsPlanner(lib)
    # Use the canonical signature so the planner takes the exact-match branch.
    target = lib.skills["sk_001_place_apple_fridge"]
    task = {
        "task_id": "t1",
        "domain_type": "place_in_container",
        "object": "apple",
        "parent": "fridge",
        "signature": target.signature(),
    }
    res = p.plan(task)
    assert res.match_level == "exact"
    assert any(a.name == "PutObject" for a in res.plan)
    assert "instantiate_template" in res.maintenance_actions_applied


def test_planner_hierarchical_partial_match():
    """Without a precomputed signature, the planner falls back to partial."""
    lib = _make_library()
    p = GraphOfGraphsPlanner(lib)
    task = {
        "task_id": "t1b",
        "domain_type": "place_in_container",
        "object": "apple",
        "parent": "fridge",
    }
    res = p.plan(task)
    assert res.match_level in ("partial", "exact", "domain_neighbor")
    assert any(a.name == "PutObject" for a in res.plan)


def test_planner_partial_match_with_instantiation():
    lib = _make_library()
    p = GraphOfGraphsPlanner(lib)
    # candle/desk are not in the library, but the domain matches
    task = {
        "task_id": "t2",
        "domain_type": "place_in_container",
        "object": "candle",
        "parent": "desk",
    }
    res = p.plan(task)
    assert res.match_level in ("partial", "domain_neighbor")
    flat_args = [a for act in res.plan for a in act.args]
    assert "candle" in flat_args
    assert "desk" in flat_args


def test_planner_returns_empty_plan_with_no_coverage_and_no_llm():
    lib = _make_library()
    from skillops.planner import PlannerConfig
    p = GraphOfGraphsPlanner(lib, config=PlannerConfig(use_llm_fallback=False))
    task = {
        "task_id": "t3",
        "domain_type": "compute",
        "object": "vector",
        "parent": "memory",
    }
    res = p.plan(task)
    # nothing to match in this domain
    assert res.match_level in ("none",)
    assert res.plan == []


# --------------------------------------------------------------------------- #
# Maintenance
# --------------------------------------------------------------------------- #


def test_merge_redundant_collapses_clone():
    lib = _make_library()
    sids = ["sk_001_place_apple_fridge", "sk_011_synthetic_redundant_apple"]
    survivor = M.merge_redundant(lib, sids)
    assert survivor in sids
    # only one of the two should remain
    remaining = sum(1 for sid in sids if sid in lib.skills)
    assert remaining == 1


def test_add_validator_fills_gap():
    lib = _make_library()
    sid = "sk_012_synthetic_missing_validator_book"
    assert lib.skills[sid].validator == []
    ok = M.add_validator(lib, sid, "lineage_inherited_validator")
    assert ok and lib.skills[sid].validator


def test_retire_skill_removes_it():
    lib = _make_library()
    n0 = len(lib)
    M.retire_skill(lib, "sk_010_look_at_painting")
    assert len(lib) == n0 - 1
    assert "sk_010_look_at_painting" not in lib.skills


def test_repair_skill_replaces_operation():
    lib = _make_library()
    new_op = [Action(name="DummyOp", args=["x"])]
    ok = M.repair_skill(lib, "sk_008_fetch_keys", new_op)
    assert ok
    s = lib.skills["sk_008_fetch_keys"]
    assert [a.name for a in s.operation] == ["DummyOp"]


def test_add_adapter_inserts_skill():
    lib = _make_library()
    aid = M.add_adapter(
        lib,
        src_id="sk_008_fetch_keys",
        dst_id="sk_001_place_apple_fridge",
        adapter_action=Action(name="CastType", args=["holding", "object"]),
    )
    assert aid is not None
    assert aid in lib.skills
    assert lib.skills[aid].is_synthetic is True


def test_maintenance_engine_sweep_runs():
    lib = _make_library()
    n0 = len(lib)
    eng = MaintenanceEngine(lib)
    report = eng.sweep()
    # at least one merge or add_validator should fire on the bundled library
    assert report.merged + report.validators_added >= 1
    assert len(lib) <= n0


# --------------------------------------------------------------------------- #
# Round-trip serialization
# --------------------------------------------------------------------------- #


def test_skill_roundtrip(tmp_path):
    lib = _make_library()
    p = tmp_path / "lib.json"
    lib.save(p)
    lib2 = SkillLibrary.load(p)
    assert len(lib) == len(lib2)
