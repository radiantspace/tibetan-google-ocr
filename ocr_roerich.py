#!/usr/bin/env python3
"""OCR Roerich dictionary PDF volumes into structured JSON.

Idempotent - skips pages that already have JSON output unless --force is used.
Uses compact line-oriented output format to minimize token usage, then converts to JSON.
Processes pages in parallel for speed.

Usage:
    # Process a single volume (20 workers by default)
    python ocr_roerich.py roerich/1Ka.pdf

    # Process all volumes
    python ocr_roerich.py roerich/*.pdf

    # Force re-OCR of already processed pages
    python ocr_roerich.py roerich/1Ka.pdf --force

    # Custom worker count and DPI
    python ocr_roerich.py roerich/1Ka.pdf --workers 10 --dpi 400

    # Test mode - output to dedicated test directory, don't touch production files
    python ocr_roerich.py roerich/1Ka.pdf --test
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # pymupdf

MAX_RETRIES = 2
RETRY_DELAY = 10  # seconds

DEFAULT_OUTPUT_DIR = "roerich_output"
TEST_OUTPUT_DIR = "roerich_test_output"

# Compact line-oriented format - ~52% fewer output tokens than JSON
OCR_PROMPT = """OCR this dictionary page and extract structured entries.
This is from an old Tibetan-English-Russian-Sanskrit dictionary.

Output in this EXACT compact format, one entry after another, separated by ===
Use these field prefixes (omit a line if the field is empty):
T: Tibetan headword (in Tibetan script)
W: Wylie transliteration
E: English definition
R: Russian translation
S: Sanskrit equivalent

For entries that span page boundaries:
- First line > means this entry continues from the previous page (T: may be empty)
- Last line < means this entry is cut off and continues on the next page

IMPORTANT:
- Keep each field on a SINGLE line, no matter how long
- Do NOT add blank lines within an entry
- Output ONLY the entries in this format, no commentary or markdown fences

Example output:
>
E:of the monk Katyayana.
R:монаха Катьяяна.
===
T:ཐ་སྐར
W:tha skar
E:stars beta and gamma Aries
R:звезды бета и гамма созвездия Овен
S:ashvini
===
T:ཐ་ཆེན
W:tha chen
E:1) building with columns; 2) supporter (one of the four
R:1) здание с колоннами; 2) приверженец (один из четырёх
<
==="""

FIELD_MAP = {
    "T": "tibetan",
    "W": "wylie",
    "E": "english",
    "R": "russian",
    "S": "sanskrit",
}


def parse_compact_entries(text):
    """Parse compact line-oriented format into list of dicts."""
    if not text:
        return []
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1:]
    if text.endswith("```"):
        text = text[:-3].rstrip()

    entries = []
    blocks = [b.strip() for b in text.split("===") if b.strip()]

    for block in blocks:
        entry = {}
        lines = block.split("\n")

        # Check for continuation flags
        if lines and lines[0].strip() == ">":
            entry["continued_from_prev_page"] = True
            lines = lines[1:]
        has_continues = False
        if lines and lines[-1].strip() == "<":
            has_continues = True
            lines = lines[:-1]

        # Parse field lines - handle multi-line values by appending
        current_key = None
        for line in lines:
            if not line.strip():
                continue
            matched = False
            for prefix, key in FIELD_MAP.items():
                tag = f"{prefix}:"
                if line.startswith(tag):
                    current_key = key
                    entry[current_key] = line[len(tag):]
                    matched = True
                    break
            if not matched and current_key:
                # Continuation of previous field value
                entry[current_key] = entry.get(current_key, "") + " " + line.strip()

        if has_continues:
            entry["continues_next_page"] = True

        # Only add entries that have at least one content field
        if any(entry.get(k) for k in FIELD_MAP.values()):
            entries.append(entry)

    return entries


def _is_server_error(exc):
    """Check if an exception indicates a 5xx server error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and 500 <= status < 600:
        return True
    msg = str(exc)
    return any(code in msg for code in ("500", "502", "503", "504"))


def ocr_page_gemini(image_path, client):
    """OCR a single page image using Gemini 3.1 Pro with compact format.

    Retries once on 5xx server errors only.
    """
    from google.genai import types

    with open(image_path, "rb") as f:
        image_data = f.read()

    contents = [
        types.Part.from_text(text=OCR_PROMPT),
        types.Part.from_bytes(data=image_data, mime_type="image/png"),
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=contents,
            )
            text = response.text
            if text is None or not text.strip():
                reason = ""
                if hasattr(response, "prompt_feedback"):
                    reason = f" (feedback: {response.prompt_feedback})"
                elif hasattr(response, "candidates") and response.candidates:
                    c = response.candidates[0]
                    if hasattr(c, "finish_reason"):
                        reason = f" (finish_reason: {c.finish_reason})"
                raise ValueError(
                    f"Gemini returned empty response{reason}"
                )
            return text
        except ValueError:
            raise  # empty response - no retry, page has no text
        except Exception as e:
            if attempt < MAX_RETRIES - 1 and _is_server_error(e):
                time.sleep(RETRY_DELAY)
                continue
            raise


def extract_page(doc, page_num, output_path, dpi=300):
    """Extract a single page from a PDF as PNG."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)


class ProgressTracker:
    """Thread-safe progress tracker with velocity and ETA."""

    def __init__(self, total_pages):
        self.total = total_pages
        self.processed = 0
        self.skipped = 0
        self.errors = 0
        self.error_pages = []
        self.start_time = time.time()
        self.lock = threading.Lock()

    def record_success(self, page_num, elapsed, entries, chars):
        with self.lock:
            self.processed += 1
            self._print_status(page_num, f"OK {elapsed:.0f}s {entries}ent {chars}ch")

    def record_skip(self):
        with self.lock:
            self.skipped += 1

    def record_error(self, page_num, elapsed, msg):
        with self.lock:
            self.errors += 1
            self.error_pages.append(page_num)
            self._print_status(page_num, f"ERR {elapsed:.0f}s {msg}")

    def _print_status(self, page_num, detail):
        done = self.processed + self.errors
        remaining = self.total - done - self.skipped
        elapsed_total = time.time() - self.start_time

        if done > 0:
            velocity = done / (elapsed_total / 60)
            eta_min = remaining / velocity if velocity > 0 else 0
            eta_str = f"ETA {eta_min:.0f}m" if eta_min > 1 else "ETA <1m"
        else:
            velocity = 0
            eta_str = "ETA --"

        bar_width = 20
        pct = (done + self.skipped) / self.total if self.total > 0 else 0
        filled = int(bar_width * pct)
        bar = "=" * filled + "-" * (bar_width - filled)

        err_str = f" err:{self.errors}" if self.errors else ""
        print(
            f"  [{bar}] {done + self.skipped}/{self.total} "
            f"p:{page_num} {detail} "
            f"| {velocity:.1f}pg/min {eta_str}{err_str}",
            flush=True,
        )

    def summary(self):
        elapsed = time.time() - self.start_time
        mins = elapsed / 60
        velocity = self.processed / mins if mins > 0 else 0
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "error_pages": self.error_pages,
            "elapsed_min": mins,
            "velocity": velocity,
        }


def process_single_page(page_num, png_path, json_path, client, tracker, force):
    """Process a single page - called by worker threads."""
    # Idempotent check
    if os.path.isfile(json_path) and os.path.getsize(json_path) > 0 and not force:
        tracker.record_skip()
        return

    raw = None
    start = time.time()
    try:
        raw = ocr_page_gemini(png_path, client)
        entries = parse_compact_entries(raw)

        if not entries:
            raise ValueError("no entries parsed from response")

        json_str = json.dumps(entries, ensure_ascii=False, indent=2)

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        # Clean up stale error file if a previous run left one
        err_path = json_path + ".error"
        if os.path.isfile(err_path):
            os.remove(err_path)

        elapsed = time.time() - start
        tracker.record_success(page_num, elapsed, len(entries), len(raw))

    except Exception as e:
        elapsed = time.time() - start
        err_path = json_path + ".error"
        try:
            content = raw if raw is not None else str(e)
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n---\n{content}")
        except Exception:
            pass
        tracker.record_error(page_num, elapsed, str(e)[:60])


def process_volume(pdf_path, force=False, dpi=300, skip_pages=0, workers=20,
                   output_dir=DEFAULT_OUTPUT_DIR):
    """Process all pages of a single PDF volume with parallel workers."""
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    volume = os.path.splitext(os.path.basename(pdf_path))[0]

    pages_dir = os.path.join(output_dir, "pages")
    json_dir = os.path.join(output_dir, "json")
    vol_pages_dir = os.path.join(pages_dir, volume)
    vol_json_dir = os.path.join(json_dir, volume)
    os.makedirs(vol_pages_dir, exist_ok=True)
    os.makedirs(vol_json_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total = len(doc)
    start_page = skip_pages + 1
    page_count = total - skip_pages

    print(f"\n{'='*60}")
    print(f"Volume: {volume} ({total} pages, starting at {start_page})")
    print(f"Workers: {workers} | DPI: {dpi} | Force: {force}")
    print(f"Output: {vol_json_dir}/")
    print(f"{'='*60}")

    # Phase 1: extract all page images (sequential, fast)
    print(f"\nExtracting page images...")
    page_tasks = []
    for page_num in range(start_page, total + 1):
        page_name = f"{volume}_page_{page_num:04d}"
        png_path = os.path.join(vol_pages_dir, f"{page_name}.png")
        json_path = os.path.join(vol_json_dir, f"{page_name}.json")

        if not os.path.isfile(png_path) or force:
            extract_page(doc, page_num, png_path, dpi)

        page_tasks.append((page_num, png_path, json_path))

    doc.close()
    print(f"  {len(page_tasks)} page images ready")

    # Phase 2: parallel OCR
    print(f"\nStarting OCR with {workers} workers...")
    tracker = ProgressTracker(page_count)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for page_num, png_path, json_path in page_tasks:
            f = pool.submit(
                process_single_page,
                page_num, png_path, json_path, client, tracker, force,
            )
            futures.append(f)

        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"  UNEXPECTED: {exc}")

    # Summary
    s = tracker.summary()
    print(f"\n--- {volume} ---")
    print(f"  Processed: {s['processed']} ({s['velocity']:.1f} pg/min)")
    print(f"  Skipped:   {s['skipped']} (already done)")
    print(f"  Errors:    {s['errors']}")
    print(f"  Time:      {s['elapsed_min']:.1f} min")
    if s["error_pages"]:
        print(f"  Error pages: {s['error_pages']}")

    return s["processed"], s["skipped"], s["errors"]


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="OCR Roerich dictionary PDFs into structured JSON"
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="PDF file(s) to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-OCR pages that already have output",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering DPI (default: 300)",
    )
    parser.add_argument(
        "--skip-pages",
        type=int,
        default=0,
        help="Skip first N pages of each PDF (e.g. title pages)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of parallel OCR workers (default: 20)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - write output to roerich_test_output/ instead of production dir",
    )

    args = parser.parse_args()
    output_dir = TEST_OUTPUT_DIR if args.test else DEFAULT_OUTPUT_DIR

    if args.test:
        print(f"*** TEST MODE - output goes to {TEST_OUTPUT_DIR}/ ***")

    total_processed = 0
    total_skipped = 0
    total_errors = 0
    overall_start = time.time()

    for pdf_path in sorted(args.pdfs):
        if not os.path.isfile(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping")
            continue
        p, s, e = process_volume(
            pdf_path,
            force=args.force,
            dpi=args.dpi,
            skip_pages=args.skip_pages,
            workers=args.workers,
            output_dir=output_dir,
        )
        total_processed += p
        total_skipped += s
        total_errors += e

    if len(args.pdfs) > 1:
        overall_min = (time.time() - overall_start) / 60
        print(f"\n{'='*60}")
        print(f"TOTAL: {total_processed} processed, {total_skipped} skipped, {total_errors} errors")
        print(f"TIME:  {overall_min:.1f} min")


if __name__ == "__main__":
    main()
