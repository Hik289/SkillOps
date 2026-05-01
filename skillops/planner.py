"""
Graph-of-Graphs Planner.

Pipeline (per task)
-------------------
1. Task-Skill Matching
   Hierarchical signature lookup against the External Graph-of-Graphs.

2. Inter-Skill Stitching
   When no exact-signature match exists, walk ``alternative`` and
   ``dependency`` edges to assemble a candidate plan.

3. Validator / Adapter Insertion
   Run the candidate skill's validator rules against the task. Insert an
   Adapter or trigger ``add_validator`` when issues are detected.

4. Local Repair
   Apply user-provided repair rules (for example argument-order normalisation)
   to the assembled action sequence.

The planner is domain-agnostic. To use it, supply a ``Task`` dict whose
``signature`` field can be matched against ``Skill.signature()`` keys.

The planner falls back to a user-provided ``llm_planner`` callable (typically
backed by :class:`skillops.llm_client.LLMClient`) when graph traversal cannot
construct a full plan.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .skill_graph import Action, Skill, SkillLibrary


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PlanResult:
    """The output of one planning call."""

    plan: List[Action]
    task_id: str
    chosen_skill_id: Optional[str]
    match_level: str  # "exact" | "partial" | "stitch" | "llm_fallback" | "none"
    maintenance_actions_applied: List[str]
    validator_issues: List[str]
    n_llm_calls: int
    cost_usd: float
    latency_ms: float
    extras: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def plan_as_dicts(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.plan]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PlannerConfig:
    """Switches for planner stages (handy for ablation studies).

    Default ``True`` for everything yields the full Graph-of-Graphs Planner.
    """

    use_signature_matching: bool = True
    use_signature_hierarchy: bool = True   # if False, only exact match counts
    use_lineage_redirect: bool = True      # synthetic -> non-synthetic parent
    enable_validator: bool = True
    enable_add_adapter: bool = True
    enable_repair: bool = True
    enable_template_instantiate: bool = True
    enable_stitch: bool = True
    use_llm_fallback: bool = True


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class GraphOfGraphsPlanner:
    """4-stage planner over a hierarchical skill graph.

    Parameters
    ----------
    library : SkillLibrary
        The library to plan over.
    config : PlannerConfig, optional
        Stage switches (defaults to a fully enabled planner).
    repair_rules : list of Callable, optional
        A list of functions ``f(plan, task) -> plan`` applied at the local
        repair stage. The default is empty.
    llm_planner : Callable, optional
        Fallback callable ``f(task, library) -> List[Action]`` used when graph
        traversal cannot produce a plan. If ``None``, ``use_llm_fallback`` has
        no effect.
    """

    def __init__(
        self,
        library: SkillLibrary,
        config: Optional[PlannerConfig] = None,
        repair_rules: Optional[List[Callable[[List[Action], Dict[str, Any]], List[Action]]]] = None,
        llm_planner: Optional[Callable[[Dict[str, Any], SkillLibrary], List[Action]]] = None,
    ) -> None:
        self.library = library
        self.config = config or PlannerConfig()
        self.repair_rules = list(repair_rules or [])
        self.llm_planner = llm_planner

    # -------- public --------
    def plan(self, task: Dict[str, Any]) -> PlanResult:
        """Run the full pipeline on a single task."""
        t0 = time.time()
        applied: List[str] = []

        matched, level = self._stage1_match(task)
        chosen_id = matched.skill_id if matched else None
        op: Optional[List[Action]] = None

        if matched is not None:
            # lineage redirect: prefer non-synthetic parent
            if self.config.use_lineage_redirect and matched.is_synthetic and matched.parent_skill_id \
                    in self.library.skills and not self.library.skills[matched.parent_skill_id].is_synthetic:
                matched = self.library.skills[matched.parent_skill_id]
                applied.append("lineage_redirect")
                chosen_id = matched.skill_id
            if self.config.enable_template_instantiate:
                op = self._instantiate(matched.operation, matched, task)
                applied.append("instantiate_template")
            else:
                op = list(matched.operation)
        elif self.config.enable_stitch:
            op = self._stage2_stitch(task)
            level = "stitch" if op else "none"

        validator_issues: List[str] = []
        if op is not None and self.config.enable_validator:
            validator_issues = self._stage3_validate(op, matched, task)
            if validator_issues and self.config.enable_add_adapter:
                op = self._stage3_apply_adapter(op, validator_issues, task)
                applied.append("add_adapter")

        if op is not None and self.config.enable_repair and self.repair_rules:
            for rule in self.repair_rules:
                op = list(rule(op, task))
            applied.append("local_repair")

        n_calls = 0
        cost = 0.0
        if (op is None or len(op) == 0) and self.config.use_llm_fallback and self.llm_planner is not None:
            llm_out = self.llm_planner(task, self.library)
            if llm_out:
                op = list(llm_out)
                level = "llm_fallback"
                applied.append("llm_fallback")
                # callers can attach n_calls / cost via the planner's last_extras
                # (left at zero here to keep the interface simple)

        if op is None:
            op = []

        return PlanResult(
            plan=op,
            task_id=str(task.get("task_id", "")),
            chosen_skill_id=chosen_id,
            match_level=level,
            maintenance_actions_applied=applied,
            validator_issues=validator_issues,
            n_llm_calls=n_calls,
            cost_usd=cost,
            latency_ms=(time.time() - t0) * 1000.0,
            extras={
                "task_signature": task.get("signature"),
            },
        )

    # -------- stages --------
    def _stage1_match(self, task: Dict[str, Any]) -> Tuple[Optional[Skill], str]:
        """Return the best matching skill and a description of the match level."""
        if not self.config.use_signature_matching:
            return None, "none"

        sig = task.get("signature")
        if sig is not None:
            cands = self.library.find_by_signature(tuple(sig) if isinstance(sig, list) else sig)
            if cands:
                return self._pick_best(cands), "exact"

        if not self.config.use_signature_hierarchy:
            return None, "none"

        # Hierarchical fallback by domain_type + partial precondition match
        domain = task.get("domain_type")
        if domain:
            cands = self.library.by_domain_type(domain)
            partial = self._partial_match(cands, task)
            if partial:
                return partial, "partial"
            if cands:
                return self._pick_best(cands), "domain_neighbor"
        return None, "none"

    def _stage2_stitch(self, task: Dict[str, Any]) -> Optional[List[Action]]:
        """Walk alternative / dependency edges to assemble a chain."""
        domain = task.get("domain_type")
        if not domain:
            return None
        cands = self.library.by_domain_type(domain)
        if not cands:
            return None
        # Trivial stitch: take the closest non-synthetic candidate.
        cands.sort(key=lambda s: (int(s.is_synthetic), -len(s.validator)))
        best = cands[0]
        if self.config.use_lineage_redirect and best.is_synthetic and best.parent_skill_id in self.library.skills:
            parent = self.library.skills[best.parent_skill_id]
            if not parent.is_synthetic:
                best = parent
        return self._instantiate(best.operation, best, task) if self.config.enable_template_instantiate else list(best.operation)

    def _stage3_validate(self, op: List[Action], skill: Optional[Skill], task: Dict[str, Any]) -> List[str]:
        """Run a skill's validator rules against the proposed plan + task.

        Each validator rule is a string parsed with the convention
        ``"<rule_name>:<argument>"``. The two built-in rules below cover the
        common cases. Callers can extend by overriding this method.

        Built-in rules
        --------------
        ``contains_action:<NAME>``
            The plan must contain at least one action with ``name == NAME``.
        ``arg_in_plan:<KEY>``
            ``task[KEY]`` (lowercased) must appear as some action argument.
        """
        if not skill:
            return []
        issues: List[str] = []
        flat_args = [str(a).lower() for act in op for a in act.args]
        action_names = {act.name for act in op}
        for rule in skill.validator:
            if not rule or ":" not in rule:
                continue
            kind, _, payload = rule.partition(":")
            kind = kind.strip()
            payload = payload.strip()
            if kind == "contains_action":
                if payload and payload not in action_names:
                    issues.append(f"missing_action:{payload}")
            elif kind == "arg_in_plan":
                want = str(task.get(payload, "")).lower()
                if want and want not in flat_args:
                    issues.append(f"missing_arg:{payload}={want}")
        return issues

    def _stage3_apply_adapter(self, op: List[Action], issues: List[str], task: Dict[str, Any]) -> List[Action]:
        """Insert a synthetic adapter Action that fills the first missing arg.

        This is intentionally minimal: more sophisticated adapter logic should
        be implemented in :func:`skillops.maintenance.add_adapter` and called
        from a Library-Time sweep.
        """
        for issue in issues:
            if issue.startswith("missing_arg:"):
                _, payload = issue.split(":", 1)
                key, _, val = payload.partition("=")
                if val:
                    return [Action(name="Acquire", args=[val])] + op
        return op

    # -------- helpers --------
    def _pick_best(self, cands: List[Skill]) -> Skill:
        return sorted(
            cands,
            key=lambda s: (int(s.is_synthetic), -len(s.validator), s.degradation_tag or ""),
        )[0]

    def _partial_match(self, cands: List[Skill], task: Dict[str, Any]) -> Optional[Skill]:
        """Best skill in ``cands`` whose precondition shares the most keys with ``task``."""
        task_keys = {k for k in task.keys() if k != "task_id"}
        best, best_score = None, -1
        for s in cands:
            if not s.precondition:
                continue
            shared = sum(
                1 for k, v in s.precondition.items()
                if k in task and str(task.get(k)).lower() == str(v).lower() and v not in (None, "")
            )
            if shared > best_score:
                best, best_score = s, shared
        return best if best_score > 0 else None

    def _instantiate(self, op: List[Action], src: Skill, task: Dict[str, Any]) -> List[Action]:
        """Substitute placeholders or known precondition values with task values.

        For each precondition key ``K`` in ``src.precondition``, replace the
        corresponding string in arguments with ``task[K]`` (case-insensitive
        match on the source value, lower-cased emitted). Also expands ``{K}``
        placeholders found anywhere in the operation arguments.
        """
        substitutions: Dict[str, str] = {}
        for k, v in src.precondition.items():
            if k in task and v not in (None, "") and isinstance(v, str):
                substitutions[v.lower()] = str(task[k]).lower()
        out: List[Action] = []
        for act in op:
            new_args: List[str] = []
            for arg in act.args:
                a = str(arg)
                low = a.lower()
                if low in substitutions:
                    new_args.append(substitutions[low])
                    continue
                # placeholder pattern: {key}
                if a.startswith("{") and a.endswith("}"):
                    key = a[1:-1]
                    if key in task:
                        new_args.append(str(task[key]).lower())
                        continue
                new_args.append(a.lower())
            out.append(Action(name=act.name, args=new_args))
        return out
