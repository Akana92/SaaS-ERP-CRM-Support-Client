"""Read-only fictional ERP objects, scoped by server-owned demo identity.

This is not authentication. The local demo assigns one fictional principal per
audience; an actual ERP connector must receive an authenticated principal.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

from support.contracts import ERPContext


REFERENCE = re.compile(r"(?<![\w-])(?:INV|SIM|BP)-[0-9]+(?![\w-])", re.IGNORECASE)
PRINCIPALS = {"customer": "demo-customer-01", "employee": "demo-employee-01"}


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def _references(message):
    return list(dict.fromkeys(match.group().upper() for match in REFERENCE.finditer(message)))


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate demo ERP JSON field")
        result[key] = value
    return result


class DemoERP:
    def __init__(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"version", "objects"}:
            raise ValueError("Invalid demo ERP fixture")
        if not _nonblank(payload["version"]) or not isinstance(payload["objects"], list):
            raise ValueError("Invalid demo ERP version or objects")
        self.version = payload["version"]
        self._objects = {}
        fields = {"reference", "org_id", "audience", "principal_id", "source_status", "facts", "label", "message"}
        for obj in payload["objects"]:
            if not isinstance(obj, dict) or set(obj) != fields:
                raise ValueError("Invalid demo ERP object fields")
            if any(not _nonblank(obj[k]) for k in fields - {"facts"}):
                raise ValueError("Empty demo ERP object field")
            reference = obj["reference"]
            if not REFERENCE.fullmatch(reference) or reference != reference.upper():
                raise ValueError("Invalid demo ERP reference")
            if reference in self._objects:
                raise ValueError("Duplicate demo ERP reference")
            if obj["audience"] not in PRINCIPALS:
                raise ValueError("Invalid demo ERP audience")
            ERPContext.model_validate({"source_status": obj["source_status"], "facts": obj["facts"]})
            if obj["facts"].get("erp.requester.audience", obj["audience"]) != obj["audience"]:
                raise ValueError("Conflicting demo ERP audience fact")
            if _references(obj["message"]) != [reference]:
                raise ValueError("Demo example must identify exactly its own object")
            self._objects[reference] = copy.deepcopy(obj)

    @classmethod
    def from_path(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_fields))

    def resolve(self, context: dict, message: str, history: list):
        result = copy.deepcopy(context)
        refs = _references(message)
        if not refs:
            for turn in reversed(history):
                if not isinstance(turn, dict) or turn.get("role", "user") != "user":
                    continue
                user_message = turn.get("message")
                if isinstance(user_message, str):
                    refs = _references(user_message)
                    if refs:
                        break
        metadata = dict(source="fictional_demo_erp", version=self.version,
                        reference=refs[0] if len(refs) == 1 else None,
                        status=result["erp_context"]["source_status"], resolution="general_help")
        if not refs:
            return result, metadata
        # Unknown and outside-scope identifiers have identical responses. Never
        # merge objects, reveal a foreign object's existence, or guess its facts.
        result["erp_context"] = {"source_status": "not_found", "facts": {}}
        metadata.update(status="not_found", resolution="ambiguous" if len(refs) > 1 else "not_found")
        if len(refs) != 1:
            return result, metadata
        audience = context.get("id")
        principal = context.get("principal_id", PRINCIPALS.get(audience))
        obj = self._objects.get(refs[0])
        if (obj is None or audience not in PRINCIPALS or obj["org_id"] != context.get("org_id")
                or obj["audience"] != audience or obj["principal_id"] != principal):
            return result, metadata
        erp = {"source_status": obj["source_status"], "facts": copy.deepcopy(obj["facts"])}
        if erp["source_status"] == "ok":
            erp["facts"]["erp.requester.audience"] = audience
            erp["facts"]["erp.help.context_scope"] = (
                f"Проверен только вымышленный объект {obj['reference']} в учебной ERP. "
                "Известны только перечисленные факты; действия с объектом не выполнялись."
            )
        result["erp_context"] = erp
        metadata.update(status=erp["source_status"], resolution="resolved")
        return result, metadata

    def examples(self, audience: str, *, context: dict | None = None):
        if context is None:
            context = {"id": audience, "org_id": "org-demo-01"}
        if audience not in PRINCIPALS or context.get("id") != audience:
            return []
        principal = context.get("principal_id", PRINCIPALS[audience])
        return [{"label": obj["label"], "message": obj["message"]}
                for obj in self._objects.values()
                if obj["audience"] == audience and obj["org_id"] == context.get("org_id")
                and obj["principal_id"] == principal]
