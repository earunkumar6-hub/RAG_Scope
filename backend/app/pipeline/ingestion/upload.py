"""S1 upload checks: extension whitelist, size limit, magic-byte content sniffing, SHA-256."""

import hashlib
import io
import zipfile
from dataclasses import dataclass

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES = 10
ALLOWED_EXTENSIONS = (".pdf", ".docx", ".md", ".txt")


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    extension: str
    content: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def extension_of(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot >= 0 else ""


def sniff_error(extension: str, content: bytes) -> str | None:
    """Return why ``content`` does not match ``extension`` (magic bytes), or None if it does."""
    if not content:
        return "file is empty"
    if extension == ".pdf":
        return None if content[:1024].lstrip().startswith(b"%PDF-") else "not a PDF (bad magic)"
    if extension == ".docx":
        if not content.startswith(b"PK\x03\x04"):
            return "not a DOCX (not a ZIP container)"
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                if "word/document.xml" not in zf.namelist():
                    return "not a DOCX (word/document.xml missing)"
        except zipfile.BadZipFile:
            return "not a DOCX (corrupt ZIP)"
        return None
    if extension in (".md", ".txt"):
        if b"\x00" in content:
            return "binary content in a text file"
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError:
            return "text file is not valid UTF-8"
        return None
    return f"unsupported extension '{extension}'"
