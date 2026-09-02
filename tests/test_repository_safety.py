from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 1_000_000

ALLOWED_DATA_PLACEHOLDERS = {
    Path("data/raw/.gitkeep"),
    Path("data/synthetic/.gitkeep"),
}
ALLOWED_AGGREGATE_JSON = {
    Path("reports/decision-optimization-benchmark.json"),
    Path("reports/point-in-time-pipeline.json"),
    Path("reports/source-benchmark.json"),
}
ALLOWED_REQUIREMENT_TEXT = {Path("requirements-benchmark.txt")}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".arrow",
    ".avro",
    ".avi",
    ".bin",
    ".bmp",
    ".bz2",
    ".ckpt",
    ".csv",
    ".db",
    ".docx",
    ".duckdb",
    ".dll",
    ".exe",
    ".feather",
    ".gz",
    ".gif",
    ".h5",
    ".ipynb",
    ".jar",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".keras",
    ".mkv",
    ".mov",
    ".ndjson",
    ".npy",
    ".npz",
    ".onnx",
    ".orc",
    ".parquet",
    ".pdf",
    ".png",
    ".pickle",
    ".pkl",
    ".pptx",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".svg",
    ".tar",
    ".tgz",
    ".tsv",
    ".webm",
    ".whl",
    ".xls",
    ".xlsx",
    ".zip",
    ".rar",
    ".xz",
    ".mp4",
}
FORBIDDEN_PATH_PARTS = {
    ".aws",
    ".local",
    ".sparkstaging",
    "artifacts",
    "checkpoints",
    "metastore_db",
    "mlartifacts",
    "mlruns",
    "models",
    "projects",
    "spark-warehouse",
}
FALLBACK_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "venv",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".bat",
    ".cmd",
    ".htm",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".ps1",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".xml",
}

WINDOWS_LOCAL_PATH = re.compile(
    r"(?i)(?<![a-z])[a-z]:[\\/]"
)
POSIX_LOCAL_PATH = re.compile(
    r"(?i)(?<![a-z0-9])/(?:users|home|tmp|var/tmp)/[^\s'\"]+"
)
FILE_URI = re.compile(r"(?i)\bfile:" + r"//")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)['\"]?(?:api[_-]?key|client[_-]?secret|access[_-]?token|"
    r"aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key|"
    r"password|passwd)['\"]?\s*[=:]\s*"
    r"(?:['\"][^'\"]{12,}['\"]|[a-z0-9_./+=-]{16,})"
)
PRIVATE_INSTITUTION = "\u76f4\u901a" + "\u7845\u8c37"
PRIVATE_DRIVE_LINK = "drive." + "google.com"


def _fallback_public_files() -> list[Path]:
    """Fail closed over the visible tree when Git metadata is unavailable."""
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or FALLBACK_EXCLUDED_PARTS.intersection(path.parts):
            continue
        if any(part.endswith(".egg-info") for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _public_files() -> list[Path]:
    """Return the tracked tree, falling back to a conservative filesystem scan."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=False,
            capture_output=True,
        )
    except (FileNotFoundError, OSError):
        return _fallback_public_files()

    if result.returncode != 0:
        return _fallback_public_files()

    files: list[Path] = []
    for raw_relative in result.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = Path(os.fsdecode(raw_relative))
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise AssertionError(f"Invalid tracked path: {relative}")
        files.append(path)

    # The command includes non-ignored untracked candidates so local checks see
    # the same files that could enter the next release. Keep this fallback for
    # environments whose older Git ignores the combined flags.
    if Path(__file__).resolve() not in files:
        files.append(Path(__file__).resolve())
    return sorted(files)


def _relative(path: Path) -> Path:
    return path.relative_to(ROOT)


def test_gitless_fallback_still_scans_visible_public_tree(monkeypatch) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", unavailable)
    files = _public_files()
    assert ROOT / "README.md" in files
    assert Path(__file__).resolve() in files


def test_public_tree_has_no_oversized_or_sensitive_artifact() -> None:
    findings: list[str] = []
    for path in _public_files():
        relative = _relative(path)
        suffix = path.suffix.lower()
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            findings.append(f"oversized:{relative}")
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden-suffix:{relative}")
    assert findings == []


def test_public_tree_has_no_restricted_or_generated_path() -> None:
    findings: list[str] = []
    for path in _public_files():
        relative = _relative(path)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts.intersection(FORBIDDEN_PATH_PARTS):
            findings.append(str(relative))
        if relative.parts[:2] in {("data", "raw"), ("data", "synthetic")}:
            if relative not in ALLOWED_DATA_PLACEHOLDERS:
                findings.append(str(relative))
        if path.suffix.lower() == ".json" and relative not in ALLOWED_AGGREGATE_JSON:
            findings.append(str(relative))
        if path.suffix.lower() == ".txt" and relative not in ALLOWED_REQUIREMENT_TEXT:
            findings.append(str(relative))
    assert sorted(set(findings)) == []


def test_public_text_has_no_local_path_institution_or_secret_assignment() -> None:
    findings: list[str] = []
    for path in _public_files():
        if path.suffix.lower() not in TEXT_SUFFIXES and not path.name.startswith(
            ".env"
        ) and path.name not in {
            ".gitattributes",
            ".gitignore",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            WINDOWS_LOCAL_PATH.search(text)
            or POSIX_LOCAL_PATH.search(text)
            or FILE_URI.search(text)
            or PRIVATE_INSTITUTION in text
            or PRIVATE_DRIVE_LINK in text
            or SECRET_ASSIGNMENT.search(text)
        ):
            findings.append(str(_relative(path)))
    assert findings == []
