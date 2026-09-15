from __future__ import annotations

import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from .utils import utc_now_iso

ProgressCallback = Callable[[float, str], None]
Task = Callable[[ProgressCallback], dict[str, Any] | None]

LOGGER = logging.getLogger("tm_local.jobs")


@dataclass
class JobRecord:
    id: str
    job_type: str
    project_id: str | None
    state: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    result: dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None
    log: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "job_type": self.job_type,
            "project_id": self.project_id,
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": deepcopy(self.result),
            "error": self.error,
            "log": self.log[-100:],
        }
        return payload


class JobManager:
    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tm-local-job")
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._active_by_project: dict[str, str] = {}

    def submit(self, job_type: str, project_id: str | None, task: Task) -> dict[str, Any]:
        with self._lock:
            if project_id and project_id in self._active_by_project:
                active = self._jobs.get(self._active_by_project[project_id])
                if active and active.state in {"queued", "running"}:
                    raise RuntimeError("This project already has a running job.")
            job_id = uuid.uuid4().hex
            record = JobRecord(id=job_id, job_type=job_type, project_id=project_id)
            self._jobs[job_id] = record
            if project_id:
                self._active_by_project[project_id] = job_id
            self._trim_locked()
            LOGGER.info(
                "Job queued: id=%s type=%s project=%s", job_id, job_type, project_id
            )
            self._executor.submit(self._run, job_id, task)
            return record.public()

    def _update(self, job_id: str, progress: float, message: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            next_progress = min(1.0, max(0.0, float(progress)))
            message_text = str(message)
            if "falling back" in message_text.lower() or "改用 cpu" in message_text.lower():
                record.progress = next_progress
            else:
                record.progress = max(record.progress, next_progress)
            record.message = message_text
            record.updated_at = utc_now_iso()
            if message and (not record.log or record.log[-1] != message):
                record.log.append(str(message))
                LOGGER.info(
                    "Job progress: id=%s type=%s project=%s progress=%.1f%% message=%s",
                    record.id,
                    record.job_type,
                    record.project_id,
                    record.progress * 100.0,
                    message,
                )

    def _run(self, job_id: str, task: Task) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.state = "running"
            record.message = "Starting…"
            record.updated_at = utc_now_iso()
        try:
            result = task(lambda progress, message: self._update(job_id, progress, message))
            with self._lock:
                record = self._jobs[job_id]
                record.state = "completed"
                record.progress = 1.0
                record.message = "Completed"
                record.result = result or {}
                record.updated_at = utc_now_iso()
                LOGGER.info(
                    "Job completed: id=%s type=%s project=%s",
                    record.id,
                    record.job_type,
                    record.project_id,
                )
        except Exception as exc:
            with self._lock:
                record = self._jobs[job_id]
                record.state = "failed"
                record.message = str(exc)
                record.error = f"{type(exc).__name__}: {exc}"
                record.traceback = traceback.format_exc()
                record.log.append(record.error)
                record.updated_at = utc_now_iso()
                LOGGER.error(
                    "Job failed: id=%s type=%s project=%s error=%s\n%s",
                    record.id,
                    record.job_type,
                    record.project_id,
                    record.error,
                    record.traceback,
                )
        finally:
            with self._lock:
                record = self._jobs.get(job_id)
                if record and record.project_id:
                    self._active_by_project.pop(record.project_id, None)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id].public()

    def active_for_project(self, project_id: str) -> dict[str, Any] | None:
        with self._lock:
            job_id = self._active_by_project.get(project_id)
            if not job_id:
                return None
            record = self._jobs.get(job_id)
            return record.public() if record else None


    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(
                self._jobs.values(), key=lambda item: item.updated_at, reverse=True
            )
            return [record.public() for record in records[: max(1, int(limit))]]

    def _trim_locked(self) -> None:
        if len(self._jobs) <= 100:
            return
        completed = [
            record for record in self._jobs.values() if record.state in {"completed", "failed"}
        ]
        completed.sort(key=lambda item: item.updated_at)
        for record in completed[: max(0, len(self._jobs) - 80)]:
            self._jobs.pop(record.id, None)
