"""Document parsing and text extraction layer for multiple data sources."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader
import requests
from backend.security import validate_url, ValidationError


@dataclass
class ParsedDocument:
    """Standardized parsed document representation prior to chunking."""
    content: str
    metadata: Dict[str, Any]


class DocumentParseError(Exception):
    """Custom exception raised when parsing a document fails."""
    pass


# -------------------------------------------------------------
# SOURCE DETECTION
# -------------------------------------------------------------
def detect_source_type(source_input: str) -> str:
    """Detect source type from input string (URL, file extension, or text)."""
    source_clean = source_input.strip()

    if "youtube.com" in source_clean or "youtu.be" in source_clean:
        return "youtube"
    if source_clean.startswith("http://") or source_clean.startswith("https://"):
        return "web"

    ext = os.path.splitext(source_clean)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".csv":
        return "csv"
    if ext in [".png", ".jpg", ".jpeg"]:
        return "image"
    if ext in [".txt", ".md", ".py", ".json", ".js", ".html"]:
        return "text"

    # Default fallback
    return "text"


# -------------------------------------------------------------
# PARSERS
# -------------------------------------------------------------
def parse_pdf(file_path: str, source_name: Optional[str] = None) -> List[ParsedDocument]:
    """Extract text from PDF file page by page with 1-indexed page metadata."""
    if not os.path.exists(file_path):
        raise DocumentParseError(f"PDF file not found: {file_path}")

    name = source_name or Path(file_path).name
    parsed_docs: List[ParsedDocument] = []

    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise DocumentParseError(f"PDF is password encrypted: {name}")

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise DocumentParseError(f"PDF contains no pages: {name}")

        for idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                parsed_docs.append(
                    ParsedDocument(
                        content=text,
                        metadata={
                            "source_name": name,
                            "source_type": "pdf",
                            "source_uri": file_path,
                            "page_number": idx + 1,  # 1-indexed for citation clarity
                            "total_pages": num_pages,
                        },
                    )
                )

        if not parsed_docs:
            raise DocumentParseError(f"No extractable text found in PDF: {name} (may be scanned images)")

        return parsed_docs

    except DocumentParseError:
        raise
    except Exception as e:
        raise DocumentParseError(f"Failed to parse PDF '{name}': {str(e)}")


def parse_csv(file_path: str, source_name: Optional[str] = None) -> List[ParsedDocument]:
    """Parse CSV into structured row-level or batch representations with column headers."""
    if not os.path.exists(file_path):
        raise DocumentParseError(f"CSV file not found: {file_path}")

    name = source_name or Path(file_path).name

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise DocumentParseError(f"CSV file is empty: {name}")

        total_rows = len(df)
        columns = [str(c) for c in df.columns]

        # Group rows into manageable batches (e.g. 20 rows per section) for meaningful retrieval
        batch_size = 20
        parsed_docs: List[ParsedDocument] = []

        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            rows_formatted = []
            for row_idx, row in batch_df.iterrows():
                row_str = ", ".join(f"{col}: {val}" for col, val in zip(columns, row.values))
                rows_formatted.append(f"Row {row_idx + 1}: {row_str}")

            batch_text = (
                f"CSV Dataset: {name}\n"
                f"Columns: {', '.join(columns)}\n"
                f"Rows {start_idx + 1} to {end_idx}:\n"
                + "\n".join(rows_formatted)
            )

            parsed_docs.append(
                ParsedDocument(
                    content=batch_text,
                    metadata={
                        "source_name": name,
                        "source_type": "csv",
                        "source_uri": file_path,
                        "row_start": start_idx + 1,
                        "row_end": end_idx,
                        "total_rows": total_rows,
                        "columns": columns,
                    },
                )
            )

        return parsed_docs

    except DocumentParseError:
        raise
    except Exception as e:
        raise DocumentParseError(f"Failed to parse CSV '{name}': {str(e)}")


def parse_web(url: str) -> List[ParsedDocument]:
    """Extract clean content and title from a web URL."""
    validate_url(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip uninformative elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        page_title = (soup.title.string or "").strip() if soup.title else url

        # Extract text from paragraphs and headings
        blocks = []
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = elem.get_text(separator=" ", strip=True)
            if text and len(text) > 15:
                blocks.append(text)

        full_text = "\n\n".join(blocks).strip()
        if not full_text:
            # Fallback to general get_text
            full_text = soup.get_text(separator=" ", strip=True)

        if not full_text or len(full_text) < 20:
            raise DocumentParseError(f"No meaningful text content extracted from web page: {url}")

        return [
            ParsedDocument(
                content=full_text,
                metadata={
                    "source_name": page_title or url,
                    "source_type": "web",
                    "source_uri": url,
                    "title": page_title,
                },
            )
        ]

    except requests.exceptions.RequestException as e:
        raise DocumentParseError(f"Failed to fetch webpage at {url}: {str(e)}")
    except DocumentParseError:
        raise
    except Exception as e:
        raise DocumentParseError(f"Error parsing webpage {url}: {str(e)}")


def extract_youtube_video_id(url: str) -> str:
    """Robust extraction of YouTube video ID from various URL patterns."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValidationError(f"Could not extract a valid 11-character YouTube video ID from '{url}'")


def parse_youtube(url: str) -> List[ParsedDocument]:
    """Extract transcript from YouTube video with multilingual and API fallbacks."""
    validate_url(url)
    video_id = extract_youtube_video_id(url)

    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
    except ImportError:
        raise DocumentParseError("youtube-transcript-api library is not installed")

    try:
        transcript_list = None
        # Support both modern fetch() and legacy get_transcript()
        if hasattr(YouTubeTranscriptApi, "fetch"):
            try:
                transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=["en", "en-US", "en-GB", "hi"])
            except Exception:
                try:
                    transcript_list = YouTubeTranscriptApi.fetch(video_id, languages=["en", "en-US", "en-GB", "hi"])
                except Exception:
                    pass
        elif hasattr(YouTubeTranscriptApi, "get_transcript"):
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=["en", "en-US", "en-GB", "hi"]
                )
            except Exception:
                pass
            try:
                t = transcript_objs.find_transcript(["en", "en-US", "hi"])
                transcript_list = t.fetch()
            except Exception:
                # Try generated transcript
                t = transcript_objs.find_generated_transcript(["en", "hi"])
                transcript_list = t.fetch()

        if not transcript_list:
            raise DocumentParseError(f"No transcripts found for YouTube video {video_id}")

        # Combine snippets
        parts = []
        for entry in transcript_list:
            text = entry.get("text", "").strip() if isinstance(entry, dict) else getattr(entry, "text", "").strip()
            if text:
                parts.append(text)

        full_transcript = " ".join(parts).strip()
        if not full_transcript:
            raise DocumentParseError(f"Empty transcript for YouTube video {video_id}")

        return [
            ParsedDocument(
                content=full_transcript,
                metadata={
                    "source_name": f"YouTube Video ({video_id})",
                    "source_type": "youtube",
                    "source_uri": url,
                    "video_id": video_id,
                },
            )
        ]

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        raise DocumentParseError(f"YouTube transcript unavailable for video {video_id}: {str(e)}")
    except DocumentParseError:
        raise
    except Exception as e:
        raise DocumentParseError(f"Error fetching YouTube transcript for {video_id}: {str(e)}")


def parse_image(file_path: str, source_name: Optional[str] = None) -> List[ParsedDocument]:
    """Extract text from image using OCR with graceful error handling."""
    if not os.path.exists(file_path):
        raise DocumentParseError(f"Image file not found: {file_path}")

    name = source_name or Path(file_path).name

    try:
        from PIL import Image
    except ImportError:
        raise DocumentParseError("PIL/Pillow library is required for image processing")

    try:
        import pytesseract
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image).strip()
        if not text:
            raise DocumentParseError(f"No readable text detected in image '{name}'")

        return [
            ParsedDocument(
                content=text,
                metadata={
                    "source_name": name,
                    "source_type": "image",
                    "source_uri": file_path,
                },
            )
        ]
    except ImportError:
        raise DocumentParseError(
            "pytesseract library is not installed. To parse images, install pytesseract and system tesseract-ocr."
        )
    except Exception as e:
        err_msg = str(e)
        if "tesseract is not installed" in err_msg.lower() or "not found" in err_msg.lower():
            raise DocumentParseError(
                f"Tesseract OCR binary is not installed on this system. Cannot extract text from '{name}'."
            )
        raise DocumentParseError(f"OCR failed for image '{name}': {err_msg}")


def parse_text(content_or_path: str, source_name: Optional[str] = None) -> List[ParsedDocument]:
    """Parse raw text string or text file."""
    if os.path.isfile(content_or_path):
        name = source_name or Path(content_or_path).name
        with open(content_or_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        uri = content_or_path
    else:
        text = content_or_path.strip()
        name = source_name or (text[:30] + "..." if len(text) > 30 else text)
        uri = "raw_text"

    if not text:
        raise DocumentParseError("Text content is empty")

    return [
        ParsedDocument(
            content=text,
            metadata={
                "source_name": name,
                "source_type": "text",
                "source_uri": uri,
            },
        )
    ]


def parse_source(source_input: str, source_name: Optional[str] = None) -> List[ParsedDocument]:
    """
    Master ingestion router that routes input to the appropriate parser
    based on detected source type.
    """
    source_type = detect_source_type(source_input)

    if source_type == "youtube":
        return parse_youtube(source_input)
    elif source_type == "web":
        return parse_web(source_input)
    elif source_type == "pdf":
        return parse_pdf(source_input, source_name=source_name)
    elif source_type == "csv":
        return parse_csv(source_input, source_name=source_name)
    elif source_type == "image":
        return parse_image(source_input, source_name=source_name)
    else:
        return parse_text(source_input, source_name=source_name)
