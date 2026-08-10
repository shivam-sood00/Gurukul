"""Keep the public task-stage catalog synchronized with Gym registrations."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "source/Gurukul/Gurukul/tasks"
STATUS_DOC = REPO_ROOT / "website/docs/reference/task-status.md"
TASK_ID_RE = re.compile(r"`(Gurukul-[^`\s]+)`")


def _registered_task_ids() -> set[str]:
    """Extract literal Gurukul IDs without importing Isaac Sim."""

    task_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for source_path in sorted(TASKS_ROOT.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "gym"
            ):
                continue

            id_keyword = next((keyword for keyword in node.keywords if keyword.arg == "id"), None)
            if id_keyword is None:
                continue
            try:
                task_id = ast.literal_eval(id_keyword.value)
            except (ValueError, TypeError) as exc:
                raise AssertionError(
                    f"Gym registration ID must be a literal string for documentation coverage: "
                    f"{source_path}:{node.lineno}"
                ) from exc

            if not isinstance(task_id, str) or not task_id.startswith("Gurukul-"):
                continue
            if task_id in task_ids:
                duplicate_ids.add(task_id)
            task_ids.add(task_id)

    assert not duplicate_ids, f"Duplicate Gym task IDs: {sorted(duplicate_ids)}"
    return task_ids


def _catalogued_task_ids() -> list[str]:
    assert STATUS_DOC.is_file(), f"Missing public task-stage catalog: {STATUS_DOC}"
    return TASK_ID_RE.findall(STATUS_DOC.read_text(encoding="utf-8"))


def test_task_stage_catalog_matches_gym_registrations() -> None:
    """Every registered task appears once, and no removed task remains documented."""

    registered = _registered_task_ids()
    catalogued_list = _catalogued_task_ids()
    catalogued = set(catalogued_list)

    missing = sorted(registered - catalogued)
    stale = sorted(catalogued - registered)
    duplicates = sorted(task_id for task_id, count in Counter(catalogued_list).items() if count != 1)

    assert not missing, f"Registered task IDs missing from task-status.md: {missing}"
    assert not stale, f"Stale task IDs in task-status.md: {stale}"
    assert not duplicates, f"Task IDs must appear exactly once in task-status.md: {duplicates}"
