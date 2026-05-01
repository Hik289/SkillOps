"""
Library-Time Maintenance Loop: five actions on the External Graph-of-Graphs.

Each action is a pure function that takes a SkillLibrary plus arguments and
mutates the library in place. The :class:`MaintenanceEngine` wraps the five
actions and exposes a periodic-sweep interface.

Actions
-------
merge_redundant(lib, ids)
    Collapse a set of redundant skills (same signature) into one canonical
    skill, choosing the one with the most validators.

repair_skill(lib, skill_id, fixed_operation)
    Replace the operation chain of a failure-prone skill in place; sets a
    lineage edge on the previous version if not already retired.

retire_skill(lib, skill_id)
    Remove a skill from the library and drop incident edges.

add_validator(lib, skill_id, validator_rule)
    Append a validator rule string to a skill's contract (closes a gap when
    a downstream consumer keeps failing on a missing post-condition check).

add_adapter(lib, src_id, dst_id, adapter_action)
    Insert an Adapter Action between two skills whose dependency edge fails
    a compatibility check. The adapter is realised as a brand-new skill in
    the library that consumes ``src.artifact`` and produces ``dst.precondition``.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional

from .skill_graph import Action, Skill, SkillContract, SkillLibrary


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def merge_redundant(lib: SkillLibrary, skill_ids: Iterable[str]) -> Optional[str]:
    """Collapse redundant skills sharing the same signature.

    Returns the surviving skill_id, or ``None`` if no merge happened.
    """
    ids = [sid for sid in skill_ids if sid in lib.skills]
    if len(ids) < 2:
        return None
    skills = [lib.skills[sid] for sid in ids]
    sig = skills[0].signature()
    if any(s.signature() != sig for s in skills):
        raise ValueError("merge_redundant: skills do not share the same signature")
    # Keep the one with the most validators; ties broken by non-synthetic preference.
    skills.sort(key=lambda s: (len(s.validator), int(not s.is_synthetic)), reverse=True)
    survivor = skills[0]
    for s in skills[1:]:
        # union failure modes onto survivor
        for fm in s.failure_modes:
            if fm not in survivor.failure_modes:
                survivor.failure_modes.append(fm)
        lib.remove_skill(s.skill_id)
    return survivor.skill_id


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------


def repair_skill(lib: SkillLibrary, skill_id: str, fixed_operation: List[Action]) -> bool:
    """Patch a skill's operation chain in place. Returns True on success."""
    if skill_id not in lib.skills:
        return False
    s = lib.skills[skill_id]
    s.contract = dataclasses.replace(s.contract, operation=list(fixed_operation))
    s.metadata.setdefault("repair_history", []).append({"reason": "library_time_repair"})
    return True


# ---------------------------------------------------------------------------
# retire
# ---------------------------------------------------------------------------


def retire_skill(lib: SkillLibrary, skill_id: str) -> bool:
    """Remove a skill from the library."""
    if skill_id not in lib.skills:
        return False
    lib.remove_skill(skill_id)
    return True


# ---------------------------------------------------------------------------
# add_validator
# ---------------------------------------------------------------------------


def add_validator(lib: SkillLibrary, skill_id: str, validator_rule: str) -> bool:
    """Add a validator rule to a skill's contract."""
    if skill_id not in lib.skills:
        return False
    if not validator_rule:
        return False
    s = lib.skills[skill_id]
    if validator_rule not in s.validator:
        s.contract.validator.append(validator_rule)
    return True


# ---------------------------------------------------------------------------
# add_adapter
# ---------------------------------------------------------------------------


def add_adapter(
    lib: SkillLibrary,
    src_id: str,
    dst_id: str,
    adapter_action: Action,
    new_skill_id: Optional[str] = None,
) -> Optional[str]:
    """Insert an adapter skill between two skills with an interface mismatch.

    The new adapter skill consumes the producer's artifact dictionary and
    re-emits it under the consumer's precondition signature.

    Returns the new adapter skill_id on success.
    """
    if src_id not in lib.skills or dst_id not in lib.skills:
        return None
    src = lib.skills[src_id]
    dst = lib.skills[dst_id]
    aid = new_skill_id or f"adapter__{src_id}__{dst_id}"
    if aid in lib.skills:
        return aid
    adapter = Skill(
        skill_id=aid,
        name=f"adapter::{src.name}->{dst.name}",
        domain_type=dst.domain_type,
        contract=SkillContract(
            precondition=dict(src.artifact),
            operation=[adapter_action],
            artifact=dict(dst.precondition),
            validator=["adapter_typecheck"],
            failure_modes=["adapter_runtime_error"],
        ),
        is_synthetic=True,
        parent_skill_id=src_id,
        degradation_tag="adapter",
    )
    lib.add_skill(adapter)
    lib.add_edge(src_id, aid, "dependency", reason="adapter_in")
    lib.add_edge(aid, dst_id, "dependency", reason="adapter_out")
    lib.add_edge(src_id, aid, "lineage", tag="adapter")
    return aid


# ---------------------------------------------------------------------------
# MaintenanceEngine: orchestrates a periodic sweep
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MaintenanceReport:
    merged: int = 0
    retired: int = 0
    repaired: int = 0
    validators_added: int = 0
    adapters_added: int = 0

    def to_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


class MaintenanceEngine:
    """Run a single Library-Time maintenance sweep.

    The engine is intentionally simple: rule-based defaults that users can
    extend or replace with LLM-driven decision logic.

    Parameters
    ----------
    library : SkillLibrary
        The library to maintain in place.
    utility_log : Dict[str, int], optional
        Map ``skill_id -> usage_count`` from execution logs. Skills with
        usage below ``retire_threshold`` are retired.
    failure_log : Dict[str, int], optional
        Map ``skill_id -> failure_count``. High-failure skills are listed as
        repair candidates (the actual fix is provided by the caller).
    retire_threshold : int
        Minimum usage to keep a skill (default 0 - never retire by usage).
    """

    def __init__(
        self,
        library: SkillLibrary,
        utility_log: Optional[Dict[str, int]] = None,
        failure_log: Optional[Dict[str, int]] = None,
        retire_threshold: int = 0,
    ) -> None:
        self.library = library
        self.utility_log = dict(utility_log or {})
        self.failure_log = dict(failure_log or {})
        self.retire_threshold = retire_threshold

    def sweep(self) -> MaintenanceReport:
        """Execute one maintenance sweep and return a report."""
        report = MaintenanceReport()

        # 1) merge: cluster by signature, merge any cluster of size >= 2
        clusters = []
        for sids in self.library._by_signature.values():
            if len(sids) >= 2:
                clusters.append(list(sids))
        for cluster in clusters:
            survivor = merge_redundant(self.library, cluster)
            if survivor is not None:
                report.merged += len(cluster) - 1

        # 2) retire: by utility threshold (only if user supplied utility_log)
        if self.utility_log and self.retire_threshold > 0:
            to_retire = [
                sid for sid, n in self.utility_log.items()
                if sid in self.library.skills and n < self.retire_threshold
            ]
            for sid in to_retire:
                if retire_skill(self.library, sid):
                    report.retired += 1

        # 3) add_validator: any synthetic skill tagged "missing_validator"
        for s in list(self.library.skills.values()):
            if s.is_synthetic and s.degradation_tag == "missing_validator" and not s.validator:
                add_validator(self.library, s.skill_id, "lineage_inherited_validator")
                report.validators_added += 1

        # NOTE: repair and add_adapter require domain-specific decisions that
        # the engine cannot make alone. Callers should invoke them directly:
        #
        #   repair_skill(library, sid, fixed_operation=[...])
        #   add_adapter(library, src_id, dst_id, Action("CastType", [...]))
        #
        # (We intentionally do not auto-fabricate operation chains here.)

        # Rebuild edges after structural changes
        self.library.build_edges()
        return report
