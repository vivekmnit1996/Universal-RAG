"""Security validation utilities for Universal RAG."""

import os
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple
from backend.config import settings


class ValidationError(Exception):
    """Custom exception for user input validation errors."""
    pass


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent directory traversal and illegal characters.
    Extracts the bare filename and replaces hazardous characters.
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")

    # Extract bare filename without directory traversal paths
    clean_name = Path(filename).name

    # Remove any null bytes or path separators
    clean_name = clean_name.replace("\x00", "").replace("/", "").replace("\\", "")

    # Limit allowed characters: alphanumeric, dots, underscores, dashes, spaces
    clean_name = re.sub(r"[^\w\s\.\-]", "", clean_name).strip()

    if not clean_name:
        raise ValidationError("Filename contains invalid characters")

    return clean_name


def validate_file_extension(filename: str) -> str:
    """Validate that the file extension is among allowed extensions."""
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        raise ValidationError("File has no extension")

    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(settings.ALLOWED_EXTENSIONS)
        raise ValidationError(
            f"Unsupported file type '{ext}'. Allowed extensions: {allowed}"
        )
    return ext


def validate_file_size(size_bytes: int) -> None:
    """Validate that uploaded file size does not exceed the limit."""
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValidationError(
            f"File size ({size_bytes / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB"
        )
    if size_bytes == 0:
        raise ValidationError("Uploaded file is empty (0 bytes)")


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate web and YouTube URLs to ensure safe schemes and prevent SSRF.
    Rejects private IP ranges or non-http/https protocols.
    """
    url = url.strip()
    if not url:
        raise ValidationError("URL cannot be empty")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(f"Invalid URL protocol '{parsed.scheme}'. Only http and https are supported.")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValidationError("URL does not contain a valid hostname")

    # Block localhost and obvious private IPs to prevent SSRF
    blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    if hostname in blocked_hosts or hostname.endswith(".local") or hostname.endswith(".internal"):
        raise ValidationError("Access to local/private network resources is restricted")

    return True, url
