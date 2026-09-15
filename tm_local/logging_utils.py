from __future__ import annotations

import atexit
import io
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO


class TeeTextIO(io.TextIOBase):
    """Write text to the original console and one or more UTF-8 log files."""

    def __init__(self, primary: TextIO, *mirrors: TextIO):
        self.primary = primary
        self.mirrors = mirrors
        self._lock = threading.RLock()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self.primary, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str | None:  # type: ignore[override]
        return getattr(self.primary, "errors", None)

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return bool(self.primary.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self.primary.fileno()

    def write(self, value: str) -> int:
        text = str(value)
        with self._lock:
            try:
                self.primary.write(text)
            except Exception:
                pass
            for mirror in self.mirrors:
                try:
                    mirror.write(text)
                except Exception:
                    pass
        return len(text)

    def flush(self) -> None:
        with self._lock:
            try:
                self.primary.flush()
            except Exception:
                pass
            for mirror in self.mirrors:
                try:
                    mirror.flush()
                except Exception:
                    pass


class SessionLog:
    def __init__(self, log_root: Path):
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = self.log_root / f"studio_{stamp}.log"
        self.latest_path = self.log_root / "LATEST.log"
        self._session_handle = self.session_path.open("a", encoding="utf-8", buffering=1)
        self._latest_handle = self.latest_path.open("w", encoding="utf-8", buffering=1)
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = TeeTextIO(self._old_stdout, self._session_handle, self._latest_handle)
        sys.stderr = TeeTextIO(self._old_stderr, self._session_handle, self._latest_handle)
        os.environ["TM_LOCAL_CURRENT_LOG"] = str(self.latest_path)
        atexit.register(self.close)

    def close(self) -> None:
        if sys.stdout is not self._old_stdout and isinstance(sys.stdout, TeeTextIO):
            try:
                sys.stdout.flush()
            except Exception:
                pass
            sys.stdout = self._old_stdout
        if sys.stderr is not self._old_stderr and isinstance(sys.stderr, TeeTextIO):
            try:
                sys.stderr.flush()
            except Exception:
                pass
            sys.stderr = self._old_stderr
        for handle in (self._session_handle, self._latest_handle):
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass


def setup_session_logging(log_root: Path) -> SessionLog:
    return SessionLog(log_root)
