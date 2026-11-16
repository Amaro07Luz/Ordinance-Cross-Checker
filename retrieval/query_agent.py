"""
Part C (1/3): Query Understanding
Turns a natural-language question into a structured ParsedQuery:
jurisdiction(s), state, ZIP codes, regulation type(s), and search keywords.

Design: deterministic rules first (fast, free, testable), with an optional LLM
hook that can fill in what the rules missed. If the LLM call fails, the rules
result is still returned, so retrieval never blocks on an API outage.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# ------------------------------------------------------------------
# Gazetteer. Canonical names MUST match GovernmentSource.jurisdiction_name
# in Part A's SourceRegistry so the jurisdiction filter lines up.
# "parents" are higher-level jurisdictions whose rules also apply.
# ------------------------------------------------------------------
GAZETTEER: Dict[str, Dict] = {
    "Fulshear": {
        "state": "TX", "zips": ["77441", "77423"],
        "aliases": ["fulshear", "cross creek ranch", "jordan ranch"],
        "parents": ["Fort Bend County"],
    },
    "Harris County": {
        "state": "TX", "zips": ["77433", "77429"],
        "aliases": ["harris county", "cypress", "bridgeland", "towne lake"],
        "parents": [],
    },
    "Montgomery County": {
        "state": "TX", "zips": ["77304", "77316"],
        "aliases": ["montgomery county", "conroe", "montgomery", "grand central park",
                    "woodforest", "artavia"],
        "parents": [],
    },
    "Washington Township (Sewell)": {
        "state": "NJ", "zips": ["08080"],
        "aliases": ["washington township", "sewell", "gloucester county"],
        "parents": [],
    },
    "Evesham Township (Marlton)": {
        "state": "NJ", "zips": ["08053"],
        "aliases": ["evesham", "marlton", "burlington county"],
        "parents": [],
    },
}

STATE_NAMES = {"texas": "TX", "new jersey": "NJ"}
STATE_DOT = {"TX": "TxDOT", "NJ": "NJDOT"}

# regulation type -> (trigger terms, expansion terms added to the search text)
REGULATION_TYPES: Dict[str, Dict[str, List[str]]] = {
    "setback": {
        "triggers": ["setback", "yard", "buffer", "front yard", "rear yard", "side yard"],
        "expand": ["setback", "yard", "minimum distance", "property line"],
    },
    "height": {
        "triggers": ["height", "stories", "story", "tall"],
        "expand": ["building height", "maximum height", "stories"],
    },
    "lot_density": {
        "triggers": ["lot size", "lot area", "minimum lot", "density", "lot coverage",
                     "frontage", "lot width", "impervious", "units per acre"],
        "expand": ["lot area", "lot width", "density", "lot coverage", "square feet"],
    },
    "access_dot": {
        "triggers": ["driveway", "access", "curb cut", "entrance", "dot", "txdot", "njdot",
                     "highway", "right-of-way", "right of way", "road"],
        "expand": ["access", "driveway", "right-of-way", "highway", "department of transportation"],
    },
    "subdivision": {
        "triggers": ["subdivision", "subdivide", "plat", "replat", "preliminary plat",
                     "final plat", "land subdivision", "site plan"],
        "expand": ["subdivision", "plat", "preliminary plat", "final plat", "site plan"],
    },
    "environmental": {
        "triggers": ["environmental", "floodplain", "flood", "drainage", "detention",
                     "wetland", "stormwater", "tree", "erosion"],
        "expand": ["floodplain", "drainage", "detention", "wetlands", "stormwater"],
    },
    "land_use": {
        "triggers": ["zoning", "zone", "district", "permitted use", "residential", "townhome",
                     "townhouse", "multifamily", "commercial", "allowed"],
        "expand": ["zoning district", "permitted uses", "residential"],
    },
    "parking": {
        "triggers": ["parking", "parking space"],
        "expand": ["parking", "off-street parking", "spaces per unit"],
    },
    "permitting": {
        "triggers": ["permit", "application", "fee", "approval", "variance"],
        "expand": ["permit", "application", "approval", "variance"],
    },
}

STOPWORDS = set("""a an and are as at be by can do does for from how i if in is it of on or
that the their there to what when where which who will with would should about need needs
rules rule requirements requirement regulations regulation required allowed tell me my our
we us any also than then this these those into over per""".split())


@dataclass
class ParsedQuery:
    raw_query: str
    jurisdictions: List[str] = field(default_factory=list)        # canonical names
    parent_jurisdictions: List[str] = field(default_factory=list)  # county-level overlays
    states: List[str] = field(default_factory=list)
    zip_codes: List[str] = field(default_factory=list)
    regulation_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    needs_state_dot: bool = False
    expanded_terms: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def search_text(self) -> str:
        """Text sent to BM25 / the embedder: the question plus synonym expansion."""
        return " ".join([self.raw_query] + self.expanded_terms)


class QueryParser:
    def __init__(self, llm_parser: Optional[Callable[[str], dict]] = None):
        """
        llm_parser: optional fn(query) -> dict with any of the keys
        'jurisdictions', 'regulation_types', 'keywords'. Its output is merged in;
        any exception is swallowed and noted as a warning.
        """
        self.llm_parser = llm_parser

    def parse(self, query: str) -> ParsedQuery:
        q = query.lower()
        pq = ParsedQuery(raw_query=query)

        # 1. ZIP codes -> jurisdictions
        pq.zip_codes = re.findall(r"\b\d{5}\b", query)
        for name, info in GAZETTEER.items():
            if any(z in info["zips"] for z in pq.zip_codes):
                self._add_jurisdiction(pq, name)

        # 2. Place names / aliases (word-boundary match)
        for name, info in GAZETTEER.items():
            if any(re.search(rf"\b{re.escape(a)}\b", q) for a in info["aliases"]):
                self._add_jurisdiction(pq, name)

        # 3. Explicit state mentions (only used if no jurisdiction matched)
        for sname, abbr in STATE_NAMES.items():
            if re.search(rf"\b{sname}\b", q) and abbr not in pq.states:
                pq.states.append(abbr)
        for abbr in ("TX", "NJ"):
            if re.search(rf"\b{abbr}\b", query) and abbr not in pq.states:
                pq.states.append(abbr)

        # 4. Regulation types
        for rtype, spec in REGULATION_TYPES.items():
            if any(re.search(rf"\b{re.escape(t)}", q) for t in spec["triggers"]):
                pq.regulation_types.append(rtype)
                pq.expanded_terms.extend(spec["expand"])
        pq.needs_state_dot = "access_dot" in pq.regulation_types

        # 5. Keywords (keeps section numbers like "220-4")
        tokens = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", q)
        pq.keywords = [t for t in dict.fromkeys(tokens) if t not in STOPWORDS and len(t) > 1]

        # 6. Optional LLM enrichment
        if self.llm_parser:
            try:
                extra = self.llm_parser(query) or {}
                for name in extra.get("jurisdictions", []):
                    if name in GAZETTEER:
                        self._add_jurisdiction(pq, name)
                for rt in extra.get("regulation_types", []):
                    if rt in REGULATION_TYPES and rt not in pq.regulation_types:
                        pq.regulation_types.append(rt)
                        pq.expanded_terms.extend(REGULATION_TYPES[rt]["expand"])
                pq.keywords += [k for k in extra.get("keywords", []) if k not in pq.keywords]
            except Exception as e:  # never let an LLM failure block retrieval
                pq.warnings.append(f"LLM parser failed, used rules only: {e}")

        # 7. Warnings the analysis agent (workstream D) can surface to the user
        if not pq.jurisdictions and not pq.states:
            pq.warnings.append("No jurisdiction detected; results span all indexed sources.")
        if not pq.regulation_types:
            pq.warnings.append("No regulation type detected; using keyword/semantic match only.")
        return pq

    @staticmethod
    def _add_jurisdiction(pq: ParsedQuery, name: str):
        if name in pq.jurisdictions:
            return
        info = GAZETTEER[name]
        pq.jurisdictions.append(name)
        if info["state"] not in pq.states:
            pq.states.append(info["state"])
        for p in info["parents"]:
            if p not in pq.parent_jurisdictions:
                pq.parent_jurisdictions.append(p)
