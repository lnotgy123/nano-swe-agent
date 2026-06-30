from __future__ import annotations

import re
import shlex
from pathlib import Path


def test_command(tests: list[str], repo_root: str | Path | None = None) -> str:
    if not tests:
        return "python -m pytest -q"
    if repo_root is not None and (Path(repo_root) / "tests" / "runtests.py").is_file():
        labels = [label for test in tests if (label := _django_label(test)) is not None]
        if not labels:
            return "python tests/runtests.py --verbosity 1"
        quoted = " ".join(shlex.quote(label) for label in labels)
        return f"python tests/runtests.py --verbosity 1 {quoted}"
    quoted = " ".join(shlex.quote(test) for test in tests)
    return f"python -m pytest -q {quoted}"


def command_argv(test: str, repo_root: str | Path) -> list[str] | None:
    if (Path(repo_root) / "tests" / "runtests.py").is_file():
        label = _django_label(test)
        if label is None:
            return None
        return ["python", "tests/runtests.py", "--verbosity", "1", label]
    return ["python", "-m", "pytest", "-q", test]


def _django_label(test: str) -> str | None:
    match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s+\(([^)]+)\)\s*", test)
    if match:
        method, dotted_class = match.groups()
        return f"{dotted_class}.{method}"
    return None
