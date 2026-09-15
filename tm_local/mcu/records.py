from __future__ import annotations

import ntpath
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from .errors import DeployError

SourceKind = Literal["c", "cxx", "asm"]
_EXTENSIONS: dict[str, SourceKind] = {".c": "c", ".cpp": "cxx", ".cc": "cxx", ".s": "asm", ".S": "asm"}
_LIB_EXTENSIONS = {".a"}
_TOOL_ALIASES = {"make_gcc_arm": ("make_gcc_arm", "gcc_arm")}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    kind: SourceKind
    progen_relpath: str


@dataclass
class BuildManifest:
    project_name: str
    app_dir: Path
    sources: list[SourceFile]
    includes: list[Path]
    defines: list[str]
    lib_dirs: list[Path]
    libs: list[str]
    common_flags: list[str]
    c_flags: list[str]
    cxx_flags: list[str]
    asm_flags: list[str]
    ld_flags: list[str]
    standard_libraries: list[str]
    linker_script: Path
    extra_ld_flags: list[str] = field(default_factory=list)
    file_prefix_map: list[tuple[str, str]] = field(default_factory=list)

    def _sorted(self, kind: SourceKind) -> list[SourceFile]:
        return sorted((s for s in self.sources if s.kind == kind), key=lambda s: s.progen_relpath)

    def c_sources(self) -> list[SourceFile]:
        return self._sorted("c")

    def cxx_sources(self) -> list[SourceFile]:
        return self._sorted("cxx")

    def asm_sources(self) -> list[SourceFile]:
        return self._sorted("asm")

    @staticmethod
    def object_name(src: SourceFile) -> str:
        return src.path.stem + ".o"

    def check_object_collisions(self) -> None:
        seen: dict[str, Path] = {}
        for src in self.sources:
            name = self.object_name(src)
            if name in seen:
                raise DeployError(f"兩個來源檔會產生同名目的檔 {name}：{seen[name]} 與 {src.path}")
            seen[name] = src.path


def _norm(entry: str) -> str:
    text = str(entry).strip().replace("/", "\\")
    return ntpath.normpath(text)


def progen_relpath(entry: str) -> str:
    return ntpath.normpath(ntpath.join("..\\..\\", _norm(entry)))


def resolve_record_path(entry: str, project_name: str, app_dir: Path, bsp_root: Path) -> Path:
    text = _norm(entry)
    if ntpath.isabs(text) or ntpath.splitdrive(text)[0]:
        raise DeployError(f"progen record 不允許絕對路徑：{entry}")
    parts = text.split("\\")
    if parts[:2] == ["..", ".."]:
        rest = parts[2:]
        if ".." in rest:
            raise DeployError(f"progen record 路徑越界：{entry}")
        return (Path(bsp_root).joinpath(*rest)).resolve()
    if parts[0] == project_name:
        rest = parts[1:]
        if ".." in rest:
            raise DeployError(f"progen record 路徑越界：{entry}")
        return (Path(app_dir).joinpath(*rest)).resolve()
    raise DeployError(f"無法解析 progen record 路徑 {entry!r}（需以 ..\\..\\ 或 {project_name}\\ 開頭）")


def _load_yaml(path: Path) -> dict:
    import yaml  # lazy: only the MCU path needs PyYAML

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}


def _tool_sections(record: dict, tool: str) -> list[dict]:
    specific = record.get("tool_specific") or {}
    return [specific[name] for name in _TOOL_ALIASES.get(tool, (tool,)) if isinstance(specific.get(name), dict)]


def _expand_source(entry: str, project_name: str, app_dir: Path, bsp_root: Path) -> list[SourceFile]:
    resolved = resolve_record_path(entry, project_name, app_dir, bsp_root)
    rel = progen_relpath(entry)
    if resolved.is_dir():
        out: list[SourceFile] = []
        for child in sorted(resolved.iterdir()):
            if child.is_file() and child.suffix in _EXTENSIONS:
                out.append(SourceFile(child.resolve(), _EXTENSIONS[child.suffix], ntpath.join(rel, child.name)))
        return out
    if resolved.is_file():
        if resolved.suffix in _EXTENSIONS:
            return [SourceFile(resolved, _EXTENSIONS[resolved.suffix], rel)]
        return []
    if resolved.suffix == ".cpp":
        alt = resolved.with_suffix(".cc")
        if alt.is_file():
            return [SourceFile(alt.resolve(), "cxx", rel)]
    raise DeployError(f"progen record 指向不存在的來源 {entry}（解析為 {resolved}）")


def load_records(template_dir: Path, app_dir: Path, bsp_root: Path, tool: str = "make_gcc_arm") -> BuildManifest:
    template_dir = Path(template_dir)
    project_yaml = template_dir / "progen" / "project.yaml"
    if not project_yaml.is_file():
        raise DeployError(f"找不到 progen project.yaml：{project_yaml}")
    project = _load_yaml(project_yaml)
    projects = project.get("projects") or {}
    if len(projects) != 1:
        raise DeployError(f"project.yaml 必須恰好定義一個專案：{project_yaml}")
    project_name, module_lists = next(iter(projects.items()))
    record_files: list[Path] = []
    for module in module_lists:
        for rel in module:
            record_files.append((template_dir / "progen" / str(rel).replace("\\", "/")).resolve())

    includes: list[Path] = []
    include_keys: set[str] = set()
    sources: list[SourceFile] = []
    lib_entries: list[str] = []
    defines: list[str] = []
    misc: dict[str, list[str]] = {}
    linker_file: str | None = None

    for record_path in record_files:
        record = _load_yaml(record_path)
        common = record.get("common") or {}
        sections = [common] + _tool_sections(record, tool)
        for section in sections:
            for entry in section.get("includes") or []:
                key = progen_relpath(entry)
                if key in include_keys:
                    continue
                include_keys.add(key)
                includes.append(resolve_record_path(entry, project_name, app_dir, bsp_root))
            for entries in (section.get("sources") or {}).values():
                for entry in entries or []:
                    if ntpath.splitext(_norm(entry))[1] in _LIB_EXTENSIONS:
                        lib_entries.append(entry)
                    else:
                        sources.extend(_expand_source(entry, project_name, app_dir, bsp_root))
            for macro in section.get("macros") or []:
                defines.append(str(macro).strip())
            for key, value in (section.get("misc") or {}).items():
                misc.setdefault(key, []).extend(str(v).strip() for v in value or [])
            if section.get("linker_file"):
                linker_file = str(section["linker_file"][0])

    if linker_file is None:
        raise DeployError(f"records 沒有 {tool} 的 linker_file：{template_dir}")
    bsp_resolved = Path(bsp_root).resolve()
    app_resolved = Path(app_dir).resolve()
    lib_entries.sort(key=progen_relpath)
    lib_dirs: list[Path] = []
    libs: list[str] = []
    for entry in lib_entries:
        resolved = resolve_record_path(entry, project_name, app_dir, bsp_root)
        if not resolved.is_file():
            raise DeployError(f"找不到靜態程式庫 {resolved}")
        name = resolved.name[:-2]
        libs.append(name.removeprefix("lib"))
        lib_dirs.append(resolved.parent)

    return BuildManifest(
        project_name=project_name,
        app_dir=app_resolved,
        sources=sources,
        includes=includes,
        defines=defines,
        lib_dirs=lib_dirs,
        libs=libs,
        common_flags=misc.get("common_flags", []),
        c_flags=misc.get("c_flags", []),
        cxx_flags=misc.get("cxx_flags", []),
        asm_flags=misc.get("asm_flags", []),
        ld_flags=misc.get("ld_flags", []),
        standard_libraries=misc.get("standard_libraries", []),
        linker_script=resolve_record_path(linker_file, project_name, app_dir, bsp_root),
        # Deterministic __FILE__/debug-info prefixes for gcc_build's -ffile-prefix-map: the
        # firmware .bin must not depend on where this Studio checkout or its mcu_toolkit/
        # happen to be installed. Callers (e.g. the spike parity test) may replace this list
        # with the exact relative prefixes a specific reference build used.
        file_prefix_map=[(str(bsp_resolved), "BSP"), (str(app_resolved), "APP")],
    )


def apply_overlay(
    manifest: BuildManifest,
    *,
    add_sources: Iterable[Path] = (),
    remove_source_basenames: Iterable[str] = (),
    add_includes: Iterable[Path] = (),
    add_defines: Iterable[str] = (),
    remove_defines: Iterable[str] = (),
) -> BuildManifest:
    removed = {name.lower() for name in remove_source_basenames}
    sources = [s for s in manifest.sources if s.path.name.lower() not in removed]
    for extra in add_sources:
        path = Path(extra).resolve()
        if not path.is_file():
            raise DeployError(f"overlay 來源不存在：{path}")
        if path.suffix not in _EXTENSIONS:
            raise DeployError(f"overlay 來源副檔名不支援：{path}")
        rel = "zz_overlay\\" + path.name  # sorts after every progen path; link order stays deterministic
        sources.append(SourceFile(path, _EXTENSIONS[path.suffix], rel))
    includes = list(manifest.includes)
    for inc in add_includes:
        p = Path(inc).resolve()
        if p not in includes:
            includes.append(p)
    dropped = {d.split("=", 1)[0] for d in remove_defines}
    defines = [d for d in manifest.defines if d.split("=", 1)[0] not in dropped]
    for d in add_defines:
        defines = [x for x in defines if x.split("=", 1)[0] != d.split("=", 1)[0]]
        defines.append(d)
    return replace(manifest, sources=sources, includes=includes, defines=defines)
