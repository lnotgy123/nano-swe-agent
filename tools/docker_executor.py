from __future__ import annotations

import os
import shlex
import uuid
from pathlib import Path

import docker

from tools.executor import ToolExecutor, ToolResult


class DockerMountedToolExecutor(ToolExecutor):
    """Use host files for editing and an official SWE-bench image for shell commands."""

    command_repo_root = "/testbed"

    def __init__(
        self,
        repo_root: str | Path,
        image: str,
        instance_id: str,
        max_output_chars: int = 16000,
        repo_cache_root: str | Path | None = None,
    ) -> None:
        super().__init__(repo_root, max_output_chars=max_output_chars)
        self.image = image
        self.instance_id = instance_id
        self.client = docker.from_env()
        safe_id = instance_id.lower().replace("__", "-").replace("_", "-")
        volumes = {str(self.repo_root): {"bind": self.command_repo_root, "mode": "rw"}}
        if repo_cache_root is not None:
            cache_root = str(Path(repo_cache_root).resolve())
            volumes[cache_root] = {"bind": cache_root, "mode": "ro"}
        self.container = self.client.containers.create(
            image=image,
            name=f"sweagent.rollout.{safe_id}.{uuid.uuid4().hex[:8]}",
            command="tail -f /dev/null",
            detach=True,
            working_dir=self.command_repo_root,
            volumes=volumes,
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            user=f"{os.getuid()}:{os.getgid()}",
        )
        self.container.start()

    def run_shell(self, command: str, timeout: int = 30) -> ToolResult:
        timeout = max(1, int(timeout))
        shell = (
            f"cd {shlex.quote(self.command_repo_root)} && "
            f"export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && "
            f"export PYTHONDONTWRITEBYTECODE=1 && "
            f"timeout --signal=KILL {timeout}s /bin/bash --noprofile --norc -lc {shlex.quote(command)}"
        )
        try:
            result = self.container.exec_run(
                ["/bin/bash", "-lc", shell],
                workdir=self.command_repo_root,
                user=f"{os.getuid()}:{os.getgid()}",
                demux=True,
            )
            stdout_bytes, stderr_bytes = result.output
            stdout = (stdout_bytes or b"").decode(errors="replace")
            stderr = (stderr_bytes or b"").decode(errors="replace")
            output = "\n".join(
                [f"exit_code: {result.exit_code}", "stdout:", stdout, "stderr:", stderr]
            )
            timed_out = result.exit_code == 124 or result.exit_code == 137
            error_type = "timeout" if timed_out else (None if result.exit_code == 0 else "command_failed")
            return ToolResult("run_shell", result.exit_code == 0, self._truncate(output), error_type)
        except Exception as exc:
            return ToolResult("run_shell", False, self._truncate(f"{type(exc).__name__}: {exc}"), "container_error")

    def close(self) -> None:
        try:
            self.container.remove(force=True)
        finally:
            self.client.close()

    def __enter__(self) -> "DockerMountedToolExecutor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
