from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvInstallResult:
    env_path: str
    ok: bool
    commands: list[str]
    log_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def instance_env_path(env_root: str | Path, instance_id: str) -> Path:
    safe_id = instance_id.replace("/", "__")
    return Path(env_root).resolve() / safe_id


def prepare_instance_env(
    repo_root: str | Path,
    env_root: str | Path,
    instance_id: str,
    recreate: bool = False,
    reuse_existing: bool = False,
    timeout: int = 1200,
    python_executable: str | Path | None = None,
) -> EnvInstallResult:
    repo_path = Path(repo_root).resolve()
    env_path = instance_env_path(env_root, instance_id)
    log_path = env_path.with_suffix(".install.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if recreate and env_path.exists():
        shutil.rmtree(env_path)

    if reuse_existing and env_path.exists():
        return EnvInstallResult(
            env_path=str(env_path),
            ok=True,
            commands=[],
            log_path=str(log_path),
        )

    commands: list[list[str]] = []
    python = str(Path(python_executable).resolve()) if python_executable else sys.executable
    if not env_path.exists():
        commands.append([python, "-m", "venv", str(env_path)])

    pip = str(env_path / "bin" / "pip")
    commands.append([pip, "install", "-U", "pip", "setuptools", "wheel"])
    commands.extend(_install_commands(repo_path, pip))

    ok = True
    completed: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        for command in commands:
            command_text = " ".join(command)
            completed.append(command_text)
            log.write(f"$ {command_text}\n")
            log.flush()
            proc = subprocess.run(
                command,
                cwd=repo_path,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            log.write(f"exit_code: {proc.returncode}\n")
            log.write("stdout:\n")
            log.write(proc.stdout)
            log.write("\nstderr:\n")
            log.write(proc.stderr)
            log.write("\n" + "=" * 80 + "\n")
            log.flush()
            if proc.returncode != 0:
                fallback_commands = _fallback_commands(repo_path, pip)
                if fallback_commands:
                    log.write("Entering fallback install strategy.\n")
                    for fallback in fallback_commands:
                        fallback_text = " ".join(fallback)
                        completed.append(fallback_text)
                        log.write(f"$ {fallback_text}\n")
                        log.flush()
                        fallback_proc = subprocess.run(
                            fallback,
                            cwd=repo_path,
                            text=True,
                            capture_output=True,
                            timeout=timeout,
                        )
                        log.write(f"exit_code: {fallback_proc.returncode}\n")
                        log.write("stdout:\n")
                        log.write(fallback_proc.stdout)
                        log.write("\nstderr:\n")
                        log.write(fallback_proc.stderr)
                        log.write("\n" + "=" * 80 + "\n")
                        log.flush()
                        if fallback_proc.returncode != 0:
                            ok = False
                            break
                    else:
                        ok = True
                    break

                ok = False
                break

    return EnvInstallResult(
        env_path=str(env_path),
        ok=ok,
        commands=completed,
        log_path=str(log_path),
    )


def _install_commands(repo_path: Path, pip: str) -> list[list[str]]:
    commands: list[list[str]] = []
    requirements = sorted(repo_path.glob("requirements*.txt"))
    for req in requirements:
        commands.append([pip, "install", "-r", str(req)])

    if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists() or (repo_path / "setup.cfg").exists():
        commands.append([pip, "install", "-e", ".[test]"])
    elif not commands:
        commands.append([pip, "install", "-e", "."])

    return commands


def _fallback_commands(repo_path: Path, pip: str) -> list[list[str]]:
    if not (repo_path / "setup.py").exists():
        return []

    python = str(Path(pip).parent / "python")
    commands = [
        [
            pip,
            "install",
            "pip<24",
            "setuptools<60",
            "wheel<0.38",
            "setuptools_scm>=6.2,<8",
            "cython==0.29.22",
            "oldest-supported-numpy",
            "extension-helpers",
        ],
        [python, "setup.py", "develop"],
        [pip, "install", "pytest==7.4.4", "hypothesis"],
    ]

    if "pytest-astropy" in _read_optional_config(repo_path):
        commands.append(
            [
                pip,
                "install",
                "pytest-astropy>=0.9",
                "pytest-astropy-header>=0.2.2",
                "pytest-xdist",
            ]
        )

    return commands


def _read_optional_config(repo_path: Path) -> str:
    parts = []
    for name in ("setup.cfg", "pyproject.toml", "tox.ini"):
        path = repo_path / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)
