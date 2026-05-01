"""
Hierarchical Skill Ecosystem Graph (HSEG) data structures.

A skill is represented as an Internal Skill Graph:
    s = (Precondition, Operation, Artifact, Validator, Failure-modes)

A skill library is the External Graph-of-Graphs:
    G = (S, R), where R is a set of typed edges among skills.

Edge types (R):
    dependency     - artifact of A is required as precondition of B
    compatibility  - artifact type of A matches precondition type of B
    redundancy     - A and B share signature (same precondition + artifact spec)
    alternative    - same goal (domain_type), different operation paths
    lineage        - A is derived from B by a maintenance action

This module is domain-agnostic. Skills carry user-defined ``domain_type`` and
``precondition``/``artifact`` dictionaries; downstream components match by
signature without assumptions on a particular environment.
"""
from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Action: a single operator invocation in a skill operation chain
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Action:
    """A single high-level operator step.

    Attributes
    ----------
    name : str
        Operator name (free-form; decided by the user's domain).
    args : List[str]
        Positional arguments. Use placeholders (e.g. ``"{object}"``) to make
        the action reusable across tasks; the planner will substitute these
        when instantiating a skill template.
    """

    name: str
    args: List[str] = dataclasses.field(default_factory=list)

    def to_str(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": list(self.args)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        return cls(name=d["name"], args=list(d.get("args", [])))


# ---------------------------------------------------------------------------
# Skill Contract (the explicit five-tuple per the paper)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SkillContract:
    """The explicit, queryable contract of a skill.

    This is the structural counterpart to a free-text "skill description" in
    flat skill libraries. Making the contract explicit lets the Graph-of-Graphs
    planner reason about artifact-level compatibility and validator coverage.
    """

    precondition: Dict[str, Any]
    operation: List[Action]
    artifact: Dict[str, Any]
    validator: List[str] = dataclasses.field(default_factory=list)
    failure_modes: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precondition": self.precondition,
            "operation": [a.to_dict() for a in self.operation],
            "artifact": self.artifact,
            "validator": list(self.validator),
            "failure_modes": list(self.failure_modes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillContract":
        return cls(
            precondition=dict(d.get("precondition", {})),
            operation=[Action.from_dict(x) for x in d.get("operation", [])],
            artifact=dict(d.get("artifact", {})),
            validator=list(d.get("validator", [])),
            failure_modes=list(d.get("failure_modes", [])),
        )


# ---------------------------------------------------------------------------
# Skill: an internal skill graph with metadata
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Skill:
    """An internal skill graph plus library-level metadata.

    Attributes
    ----------
    skill_id : str
        Unique identifier in a library.
    name : str
        Human-readable label.
    domain_type : str
        A coarse grouping label (for example a goal family). Used by the
        planner's first-level filter and the alternative-edge builder.
    contract : SkillContract
        The explicit five-tuple (P, O, A, V, F).
    is_synthetic : bool
        True if the skill was automatically generated or degraded; useful for
        the lineage edge and the ``add_validator``/``retire`` actions.
    parent_skill_id : Optional[str]
        If synthetic, points to the lineage parent.
    degradation_tag : Optional[str]
        Free-form tag describing the type of degradation, for example
        ``"adapter_broken"`` or ``"missing_validator"``.
    metadata : Dict[str, Any]
        Free-form annotations.
    """

    skill_id: str
    name: str
    domain_type: str
    contract: SkillContract
    is_synthetic: bool = False
    parent_skill_id: Optional[str] = None
    degradation_tag: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # ----- convenience accessors -----
    @property
    def precondition(self) -> Dict[str, Any]:
        return self.contract.precondition

    @property
    def operation(self) -> List[Action]:
        return self.contract.operation

    @property
    def artifact(self) -> Dict[str, Any]:
        return self.contract.artifact

    @property
    def validator(self) -> List[str]:
        return self.contract.validator

    @property
    def failure_modes(self) -> List[str]:
        return self.contract.failure_modes

    def signature(self) -> Tuple:
        """A canonical signature for redundancy / alternative-edge building.

        Defaults to ``(domain_type, precondition_items_sorted, artifact_items_sorted)``.
        Users may subclass and override for richer signatures.
        """
        pre_items = tuple(sorted((k, _hashable(v)) for k, v in self.precondition.items()))
        art_items = tuple(sorted((k, _hashable(v)) for k, v in self.artifact.items()))
        return (self.domain_type, pre_items, art_items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "domain_type": self.domain_type,
            "contract": self.contract.to_dict(),
            "is_synthetic": self.is_synthetic,
            "parent_skill_id": self.parent_skill_id,
            "degradation_tag": self.degradation_tag,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(
            skill_id=d["skill_id"],
            name=d.get("name", d["skill_id"]),
            domain_type=d.get("domain_type", ""),
            contract=SkillContract.from_dict(d.get("contract", {})),
            is_synthetic=bool(d.get("is_synthetic", False)),
            parent_skill_id=d.get("parent_skill_id"),
            degradation_tag=d.get("degradation_tag"),
            metadata=dict(d.get("metadata", {})),
        )


def _hashable(v: Any):
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

EDGE_TYPES = ("dependency", "compatibility", "redundancy", "alternative", "lineage")


@dataclasses.dataclass
class Edge:
    src: str
    dst: str
    edge_type: str
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "edge_type": self.edge_type, "meta": dict(self.meta)}


# ---------------------------------------------------------------------------
# SkillLibrary: the External Graph-of-Graphs
# ---------------------------------------------------------------------------


class SkillLibrary:
    """A skill library plus typed edges among skills.

    Parameters
    ----------
    skills : iterable of Skill, optional
        Initial skill set. Edges may be added afterwards via
        :meth:`add_edge` or :meth:`build_edges`.
    """

    def __init__(self, skills: Optional[Iterable[Skill]] = None) -> None:
        self.skills: Dict[str, Skill] = {}
        self.edges: List[Edge] = []
        self._by_domain_type: Dict[str, List[str]] = defaultdict(list)
        self._by_signature: Dict[Tuple, List[str]] = defaultdict(list)
        if skills:
            for s in skills:
                self.add_skill(s)

    # ----- mutation -----
    def add_skill(self, skill: Skill) -> None:
        if skill.skill_id in self.skills:
            return
        self.skills[skill.skill_id] = skill
        self._by_domain_type[skill.domain_type].append(skill.skill_id)
        self._by_signature[skill.signature()].append(skill.skill_id)

    def remove_skill(self, skill_id: str) -> None:
        if skill_id not in self.skills:
            return
        s = self.skills.pop(skill_id)
        try:
            self._by_domain_type[s.domain_type].remove(skill_id)
        except ValueError:
            pass
        try:
            self._by_signature[s.signature()].remove(skill_id)
        except ValueError:
            pass
        self.edges = [e for e in self.edges if e.src != skill_id and e.dst != skill_id]

    def add_edge(self, src: str, dst: str, edge_type: str, **meta: Any) -> None:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {edge_type}; expected one of {EDGE_TYPES}")
        if src == dst or src not in self.skills or dst not in self.skills:
            return
        self.edges.append(Edge(src=src, dst=dst, edge_type=edge_type, meta=dict(meta)))

    # ----- querying -----
    def by_domain_type(self, domain_type: str) -> List[Skill]:
        return [self.skills[i] for i in self._by_domain_type.get(domain_type, [])]

    def all(self) -> List[Skill]:
        return list(self.skills.values())

    def find_by_signature(self, signature: Tuple) -> List[Skill]:
        return [self.skills[i] for i in self._by_signature.get(signature, [])]

    def edges_of(self, skill_id: str, edge_type: Optional[str] = None) -> List[Edge]:
        return [
            e for e in self.edges
            if (e.src == skill_id or e.dst == skill_id)
            and (edge_type is None or e.edge_type == edge_type)
        ]

    def __len__(self) -> int:
        return len(self.skills)

    # ----- automatic edge construction -----
    def build_edges(self) -> Dict[str, int]:
        """Recompute typed edges based on signatures and contract overlap.

        Heuristics
        ----------
        - redundancy: same signature
        - alternative: same domain_type, different signature, capped degree
        - dependency: artifact[k] of A equals precondition[k] of B for some
          shared key (commonly ``"holding"`` or ``"output_type"``)
        - compatibility: dependency edges where artifact[k] type matches
          precondition[k] type
        - lineage: synthetic skill -> its non-synthetic parent

        Returns a count of edges per type.
        """
        self.edges = []
        # 1) redundancy
        for sids in self._by_signature.values():
            for i, a in enumerate(sids):
                for b in sids[i + 1:]:
                    self.add_edge(a, b, "redundancy", reason="same_signature")
        # 2) alternative (degree-capped to avoid quadratic blowup on big libs)
        for dt, sids in self._by_domain_type.items():
            for i, a in enumerate(sids[:200]):
                for b in sids[i + 1: i + 6]:
                    if self.skills[a].signature() != self.skills[b].signature():
                        self.add_edge(a, b, "alternative", reason="same_domain_type")
        # 3) dependency: matching artifact key with precondition key
        for a in self.skills.values():
            for k, v in a.artifact.items():
                for b in self.skills.values():
                    if a.skill_id == b.skill_id:
                        continue
                    if k in b.precondition and b.precondition[k] == v and v not in (None, ""):
                        self.add_edge(a.skill_id, b.skill_id, "dependency", via_key=k, value=v)
        # 4) compatibility: dependency where types annotated in metadata match
        for e in [e for e in self.edges if e.edge_type == "dependency"]:
            a = self.skills[e.src]
            b = self.skills[e.dst]
            ka = e.meta.get("via_key")
            type_a = a.metadata.get("artifact_types", {}).get(ka)
            type_b = b.metadata.get("precondition_types", {}).get(ka)
            if type_a and type_b and type_a == type_b:
                self.add_edge(e.src, e.dst, "compatibility", via_key=ka, type=type_a)
        # 5) lineage
        for s in self.skills.values():
            if s.is_synthetic and s.parent_skill_id and s.parent_skill_id in self.skills:
                self.add_edge(s.parent_skill_id, s.skill_id, "lineage", tag=s.degradation_tag)
        cnt: Dict[str, int] = defaultdict(int)
        for e in self.edges:
            cnt[e.edge_type] += 1
        return dict(cnt)

    # ----- serialization -----
    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "skills": [s.to_dict() for s in self.skills.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_jsonable(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "SkillLibrary":
        d = json.loads(Path(path).read_text())
        lib = cls()
        for s in d.get("skills", []):
            lib.add_skill(Skill.from_dict(s))
        for e in d.get("edges", []):
            lib.add_edge(e["src"], e["dst"], e["edge_type"], **e.get("meta", {}))
        return lib

    @classmethod
    def load_directory(cls, path) -> "SkillLibrary":
        """Load every ``*.json`` and ``*.yaml`` file in a directory as one skill.

        This is the friendliest format for hand-curated example libraries.
        """
        lib = cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        for fp in sorted(p.iterdir()):
            if fp.suffix.lower() == ".json":
                d = json.loads(fp.read_text())
                lib.add_skill(Skill.from_dict(d))
            elif fp.suffix.lower() in (".yaml", ".yml"):
                try:
                    import yaml  # type: ignore
                except ImportError as exc:
                    raise RuntimeError("install pyyaml to load YAML skills") from exc
                d = yaml.safe_load(fp.read_text())
                lib.add_skill(Skill.from_dict(d))
        lib.build_edges()
        return lib
