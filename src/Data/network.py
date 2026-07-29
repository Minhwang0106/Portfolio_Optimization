"""Keep the CA bundle on a path Windows' ANSI codepage can represent.

yfinance 1.5+ sends every request through curl_cffi, which on Windows encodes
libcurl file-path options with `locale.getpreferredencoding(False)` -- the
process ANSI codepage -- because that is what the CRT's `fopen` expects. On
this machine that is cp1252, and the project (hence the venv, hence certifi's
`cacert.pem`) lives under 'Học tập và làm việc'. cp1252 has no 'ọ', so
`setopt(CAINFO, ...)` raises UnicodeEncodeError and *every* HTTPS request dies
before it reaches the network -- `Ticker.history()` returns empty and
`Ticker.splits` returns None, which `take_price` reports as "possibly
delisted". `src.Data.run` then skips every ticker and raises "No ticker
succeeded".

Widening the encoding would not help: libcurl really cannot open the file,
so the bundle has to sit somewhere ANSI-representable. Copying it to an
ASCII-only path and pointing CURL_CA_BUNDLE there is enough.
"""
import locale
import os
import shutil
import sys
import tempfile
from pathlib import Path

_CACERT_NAME: str = 'cacert.pem'
_checked: bool = False


def _ansi_encodable(path: str | os.PathLike) -> bool:
    """Whether libcurl could hand this path to `fopen` on this platform."""
    if not sys.platform.startswith('win'):
        return True
    try:
        os.fspath(path).encode(locale.getpreferredencoding(False),
                               errors='strict')
    except UnicodeEncodeError:
        return False
    return True


def _candidate_dirs() -> list[Path]:
    """Places to park the bundle, best first."""
    return [Path.home() / '.certs',
            Path(tempfile.gettempdir()) / 'certs',
            Path(os.environ.get('SystemDrive', 'C:')) / os.sep / 'certs']


def ensure_encodable_cacert() -> str | None:
    """Point CURL_CA_BUNDLE at a CA bundle curl can actually open.

    A no-op on non-Windows, and on Windows whenever certifi's own path is
    already representable in the ANSI codepage -- so this costs nothing once
    the project is moved somewhere ASCII. Idempotent; safe to call per fetch.

    Returns:
        str | None: Path to the bundle now in use, or None if no relocation
            was needed.

    Raises:
        RuntimeError: If every candidate location is itself unrepresentable,
            which leaves no path libcurl could open.
    """
    global _checked
    if _checked:
        return os.environ.get('CURL_CA_BUNDLE')

    existing: str | None = os.environ.get('CURL_CA_BUNDLE')
    if existing and Path(existing).is_file() and _ansi_encodable(existing):
        _checked = True
        return existing

    try:
        import certifi
    except ImportError:      # nothing to relocate; let curl use its own store
        _checked = True
        return None

    source: str = certifi.where()
    if _ansi_encodable(source):
        _checked = True
        return None

    for directory in _candidate_dirs():
        if not _ansi_encodable(directory):
            continue
        destination: Path = directory / _CACERT_NAME
        directory.mkdir(parents=True, exist_ok=True)
        # Re-copy when stale: certifi ships new roots on upgrade, and a bundle
        # that silently lags would fail verification rather than encoding.
        if (not destination.is_file()
                or destination.stat().st_mtime < Path(source).stat().st_mtime):
            shutil.copyfile(source, destination)
        os.environ['CURL_CA_BUNDLE'] = str(destination)
        os.environ.setdefault('REQUESTS_CA_BUNDLE', str(destination))
        _checked = True
        return str(destination)

    raise RuntimeError(
        f'No ANSI-representable directory available for the CA bundle; '
        f'tried {[str(d) for d in _candidate_dirs()]}. Move the project to a '
        f'path without non-cp1252 characters.'
    )