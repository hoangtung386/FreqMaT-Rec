"""Resumable downloads of the raw datasets into ``data/raw/<dataset>/``."""

import logging
import shutil
import urllib.request
from pathlib import Path

from freqmat import paths
from freqmat.data.sources import SOURCES

LOGGER = logging.getLogger(__name__)
CHUNK = 1 << 20


def fetch(url: str, dest: Path, retries: int = 5) -> Path:
    """Download ``url`` to ``dest``, resuming a partial ``.part`` file; no-op if ``dest`` exists."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        try:
            with urllib.request.urlopen(request) as resp, open(part, "ab" if start else "wb") as out:
                if start and resp.status != 206:  # server ignored Range: restart from scratch
                    out.seek(0)
                    out.truncate()
                shutil.copyfileobj(resp, out, CHUNK)
            part.rename(dest)
            return dest
        except OSError as err:  # includes HTTP/URL errors
            if "416" in str(err):  # already fully downloaded
                part.rename(dest)
                return dest
            LOGGER.warning("download attempt %d/%d failed: %s", attempt, retries, err)
    raise RuntimeError(f"could not download {url}")


def download(names: list[str] | None = None) -> list[Path]:
    """Download the named sources (default: all)."""
    selected = names or list(SOURCES)
    unknown = [n for n in selected if n not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; choose from {sorted(SOURCES)}")
    results = []
    for name in selected:
        subdir, url = SOURCES[name]
        LOGGER.info("== %s", name)
        results.append(fetch(url, paths.DATA_RAW / subdir / url.rsplit("/", 1)[1]))
    return results
