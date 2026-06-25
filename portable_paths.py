"""Drive-flexible path storage and resolution for portable data folders."""

from __future__ import annotations

import os
from pathlib import Path


def path_for_storage(path: str | Path, base: str | Path) -> str:
    """Convert an absolute path to a portable form relative to a base directory."""
    abs_path = Path(path).resolve()
    abs_base = Path(base).resolve()
    try:
        return abs_path.relative_to(abs_base).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(str(abs_path), str(abs_base))).as_posix()
        except ValueError:
            return abs_path.as_posix()


def path_from_storage(
    stored: str,
    current_base: str | Path,
    *,
    stored_base: str | Path | None = None,
) -> Path:
    """Resolve a stored path against the current base directory."""
    if not stored:
        return Path(stored)
    stored_path = Path(stored)
    current_base_path = Path(current_base).resolve()

    if stored_path.is_absolute():
        resolved = stored_path.resolve()
        if resolved.exists():
            return resolved
        if stored_base is not None:
            stored_base_path = Path(stored_base).resolve()
            for relative_maker in (
                lambda: resolved.relative_to(stored_base_path),
                lambda: Path(os.path.relpath(str(resolved), str(stored_base_path))),
            ):
                try:
                    relative = relative_maker()
                except ValueError:
                    continue
                candidate = (current_base_path / relative).resolve()
                if candidate.exists():
                    return candidate
                return candidate
        return resolved

    return (current_base_path / stored_path).resolve()


def resolve_directory(
    stored_directory: str | Path | None,
    current_directory: str | Path,
) -> str:
    """Resolve a stored row directory against the currently loaded directory."""
    if stored_directory is None:
        return str(Path(current_directory).resolve())
    text = str(stored_directory).strip()
    if not text:
        return str(Path(current_directory).resolve())
    stored_path = Path(text)
    current = Path(current_directory).resolve()
    if stored_path.is_absolute():
        return str(path_from_storage(text, current, stored_base=text))
    return str((current / stored_path).resolve())


def directory_matches(
    stored_directory: str | Path | None,
    current_directory: str | Path,
) -> bool:
    """Return True when a stored row directory refers to the current directory."""
    if stored_directory is None:
        return False
    text = str(stored_directory).strip()
    if not text:
        return False
    current = Path(current_directory).resolve()
    stored_path = Path(text)
    try:
        if stored_path.resolve() == current:
            return True
    except OSError:
        pass
    return Path(resolve_directory(text, current)).resolve() == current
