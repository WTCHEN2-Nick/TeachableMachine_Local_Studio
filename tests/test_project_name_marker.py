from __future__ import annotations

from pathlib import Path

from tm_local.project_store import ProjectStore, _marker_filename, _refresh_name_marker


def _markers(project_dir: Path) -> list[str]:
    return sorted(p.name for p in project_dir.glob("【*】.txt"))


def test_a_new_project_gets_a_marker_named_after_it(tmp_path: Path) -> None:
    """The folder is a uuid; this file is how a human tells the folders apart in Explorer."""
    store = ProjectStore(tmp_path)
    project = store.create_project("image", "貓狗分類")
    assert _markers(store.project_dir(project["id"])) == ["【貓狗分類】.txt"]


def test_renaming_replaces_the_marker_rather_than_adding_one(tmp_path: Path) -> None:
    """Two markers would be worse than none -- the reader could not tell which name is current."""
    store = ProjectStore(tmp_path)
    project = store.create_project("image", "舊名字")
    store.update_project(project["id"], {"name": "新名字"})
    assert _markers(store.project_dir(project["id"])) == ["【新名字】.txt"]


def test_the_marker_is_empty_and_never_shadows_project_json(tmp_path: Path) -> None:
    """It carries no data. project.json stays the only source of truth."""
    store = ProjectStore(tmp_path)
    project = store.create_project("audio", "測試")
    project_dir = store.project_dir(project["id"])
    assert (project_dir / "【測試】.txt").stat().st_size == 0
    assert (project_dir / "project.json").is_file()
    assert store.get_project(project["id"])["name"] == "測試"


def test_characters_windows_refuses_are_replaced_not_dropped(tmp_path: Path) -> None:
    """A name with a slash must still produce a usable file, not a crash or a nested path."""
    assert _marker_filename('a/b:c*d?') == "【a_b_c_d_】.txt"
    assert "/" not in _marker_filename("cats/dogs")


def test_a_name_that_sanitises_to_nothing_still_gets_a_marker(tmp_path: Path) -> None:
    """Trailing dots and spaces are dropped by the filesystem; an empty result needs a fallback."""
    assert _marker_filename("...") == "【未命名專案】.txt"
    assert _marker_filename("   ") == "【未命名專案】.txt"


def test_a_marker_that_cannot_be_written_does_not_fail_the_save(tmp_path: Path) -> None:
    """The marker is a convenience; losing it must never cost the student their project."""
    missing = tmp_path / "not-a-directory"
    _refresh_name_marker(missing, "whatever")  # must not raise
    assert not missing.exists()
