from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .config import MAX_PROJECT_ARCHIVE_BYTES


class ArchiveError(ValueError):
    pass


def _safe_member_path(destination: Path, name: str) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    drive_like = bool(pure.parts) and len(pure.parts[0]) == 2 and pure.parts[0][1] == ":"
    if pure.is_absolute() or drive_like or ".." in pure.parts:
        raise ArchiveError(f"Unsafe ZIP member: {name}")
    target = (destination / Path(*pure.parts)).resolve()
    target.relative_to(destination.resolve())
    return target


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        if len(infos) > 100_000:
            raise ArchiveError("Archive contains too many entries.")
        total = sum(max(0, int(info.file_size)) for info in infos)
        if total > MAX_PROJECT_ARCHIVE_BYTES:
            raise ArchiveError("Archive expands beyond the 2 GiB safety limit.")
        for info in infos:
            target = _safe_member_path(destination, info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ArchiveError(f"Symbolic links are not accepted: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def zip_directory(source: Path, output: Path, *, include_root: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            arcname = Path(source.name) / relative if include_root else relative
            zf.write(path, arcname.as_posix())
    return output
