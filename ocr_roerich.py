#!/usr/bin/env python3
"""OCR Roerich dictionary PDF volumes into structured JSON.

Two modes:
  Real-time (default): parallel workers, immediate results
  Batch (--batch): uses Gemini Batch API for 50% cost savings, async processing

Real-time usage:
    python ocr_roerich.py roerich/1Ka.pdf
    python ocr_roerich.py roerich/*.pdf --workers 10 --dpi 400
    python ocr_roerich.py roerich/1Ka.pdf --test

Batch usage:
    python ocr_roerich.py --batch submit roerich/*.pdf
    python ocr_roerich.py --batch status
    python ocr_roerich.py --batch collect
    python ocr_roerich.py --batch retry
"""

import argparse
import json
import os
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import fitz  # pymupdf

MAX_RETRIES = 2
RETRY_DELAY = 10  # seconds
GEMINI_MODEL = "gemini-3.1-pro-preview"

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


def extract_page(doc, page_num, output_path, dpi=300):
    """Extract a single page from a PDF as PNG."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)


def _get_client():
    """Create and return a Gemini API client."""
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Batch state management
# ---------------------------------------------------------------------------

class BatchState:
    """Persistent state for batch API workflow.

    Tracks uploaded files, batch jobs, and completed pages.
    Saves atomically via temp file + rename.
    Thread-safe - all reads and writes are protected by a lock.
    """

    def __init__(self, output_dir):
        self.path = os.path.join(output_dir, "batch_state.json")
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self):
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"uploaded_files": {}, "batches": {}, "completed_pages": []}

    def save(self):
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self):
        """Save without acquiring lock (caller must hold lock)."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(self.path) or ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            os.unlink(tmp)
            raise

    # -- uploaded files --

    def is_uploaded(self, page_key):
        with self._lock:
            return page_key in self.data["uploaded_files"]

    def record_upload(self, page_key, file_name, uri):
        with self._lock:
            self.data["uploaded_files"][page_key] = {
                "file_name": file_name,
                "uri": uri,
                "uploaded_at": _now_iso(),
            }

    def record_upload_and_save(self, page_key, file_name, uri,
                               save_every=20, counter=None):
        """Record an upload and periodically save (thread-safe).

        Args:
            counter: a list with one int element used as an atomic counter.
                     Incremented under the lock to decide when to flush.
        """
        with self._lock:
            self.data["uploaded_files"][page_key] = {
                "file_name": file_name,
                "uri": uri,
                "uploaded_at": _now_iso(),
            }
            if counter is not None:
                counter[0] += 1
                if counter[0] % save_every == 0:
                    self._save_unlocked()

    def get_upload(self, page_key):
        with self._lock:
            return self.data["uploaded_files"].get(page_key)

    # -- batches --

    def record_batch(self, batch_name, volume, page_keys, display_name):
        with self._lock:
            self.data["batches"][batch_name] = {
                "display_name": display_name,
                "volume": volume,
                "page_keys": page_keys,
                "state": "JOB_STATE_PENDING",
                "created_at": _now_iso(),
                "last_checked": _now_iso(),
            }

    def update_batch_state(self, batch_name, state):
        with self._lock:
            if batch_name in self.data["batches"]:
                self.data["batches"][batch_name]["state"] = state
                self.data["batches"][batch_name]["last_checked"] = _now_iso()

    def get_active_batches(self):
        """Return batches that are still pending or running."""
        with self._lock:
            active = {}
            for name, info in self.data["batches"].items():
                if info["state"] in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING"):
                    active[name] = info
            return active

    def get_succeeded_batches(self):
        """Return batches that succeeded but haven't been fully collected."""
        with self._lock:
            result = {}
            for name, info in self.data["batches"].items():
                if info["state"] == "JOB_STATE_SUCCEEDED":
                    uncollected = [
                        pk for pk in info["page_keys"]
                        if pk not in self.data["completed_pages"]
                    ]
                    if uncollected:
                        result[name] = info
            return result

    def get_failed_batches(self):
        with self._lock:
            result = {}
            for name, info in self.data["batches"].items():
                if info["state"] in ("JOB_STATE_FAILED", "JOB_STATE_EXPIRED"):
                    result[name] = info
            return result

    # -- completed pages --

    def is_completed(self, page_key):
        with self._lock:
            return page_key in self.data["completed_pages"]

    def mark_completed(self, page_key):
        with self._lock:
            if page_key not in self.data["completed_pages"]:
                self.data["completed_pages"].append(page_key)

    # -- queries --

    def is_page_pending(self, page_key):
        """Check if a page is in an active (pending/running) batch."""
        with self._lock:
            for info in self.data["batches"].values():
                if info["state"] in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING"):
                    if page_key in info["page_keys"]:
                        return True
            return False

    def remove_batch(self, batch_name):
        with self._lock:
            self.data["batches"].pop(batch_name, None)

    def clear_uploads_for_pages(self, page_keys):
        with self._lock:
            for pk in page_keys:
                self.data["uploaded_files"].pop(pk, None)


# ---------------------------------------------------------------------------
# Batch commands
# ---------------------------------------------------------------------------

def _extract_volume_pages(pdf_path, output_dir, dpi=300, skip_pages=0):
    """Extract all pages from a PDF and return list of (page_key, png_path)."""
    volume = os.path.splitext(os.path.basename(pdf_path))[0]
    vol_pages_dir = os.path.join(output_dir, "pages", volume)
    os.makedirs(vol_pages_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total = len(doc)
    start_page = skip_pages + 1

    pages = []
    for page_num in range(start_page, total + 1):
        page_key = f"{volume}_page_{page_num:04d}"
        png_path = os.path.join(vol_pages_dir, f"{page_key}.png")
        if not os.path.isfile(png_path):
            extract_page(doc, page_num, png_path, dpi)
        pages.append((page_key, png_path))

    doc.close()
    return volume, pages


def batch_submit(pdf_paths, output_dir, dpi=300, skip_pages=0, force=False,
                 workers=20):
    """Upload page images and submit batch jobs for each volume."""
    from google.genai import types

    client = _get_client()
    state = BatchState(output_dir)
    json_dir = os.path.join(output_dir, "json")

    for pdf_path in sorted(pdf_paths):
        if not os.path.isfile(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping")
            continue

        volume, all_pages = _extract_volume_pages(
            pdf_path, output_dir, dpi, skip_pages
        )
        vol_json_dir = os.path.join(json_dir, volume)
        os.makedirs(vol_json_dir, exist_ok=True)

        # Filter out pages that already have results or are in active batches
        pages_to_submit = []
        for page_key, png_path in all_pages:
            json_path = os.path.join(vol_json_dir, f"{page_key}.json")
            has_json = (
                os.path.isfile(json_path)
                and os.path.getsize(json_path) > 0
            )
            if has_json and not force:
                continue
            if state.is_page_pending(page_key) and not force:
                continue
            pages_to_submit.append((page_key, png_path))

        if not pages_to_submit:
            print(f"{volume}: all {len(all_pages)} pages already done or queued")
            continue

        # Filter to only pages needing upload
        pages_needing_upload = [
            (pk, pp) for pk, pp in pages_to_submit
            if force or not state.is_uploaded(pk)
        ]

        print(f"\n{volume}: {len(pages_to_submit)} pages to submit "
              f"({len(all_pages)} total), "
              f"{len(pages_needing_upload)} need upload")

        # Parallel upload images to File API
        if pages_needing_upload:
            upload_workers = min(workers, len(pages_needing_upload))
            counter = [0]  # mutable counter shared across threads
            errors = []
            start_t = time.time()

            def _upload_one(item):
                page_key, png_path = item
                f = client.files.upload(
                    file=png_path,
                    config=types.UploadFileConfig(
                        display_name=page_key,
                        mime_type="image/png",
                    ),
                )
                state.record_upload_and_save(
                    page_key, f.name, f.uri,
                    save_every=20, counter=counter,
                )
                return page_key

            print(f"  Uploading {len(pages_needing_upload)} images "
                  f"({upload_workers} workers)...", end="", flush=True)

            with ThreadPoolExecutor(max_workers=upload_workers) as pool:
                futures = {
                    pool.submit(_upload_one, item): item[0]
                    for item in pages_needing_upload
                }
                done_count = 0
                for future in as_completed(futures):
                    page_key = futures[future]
                    try:
                        future.result()
                        done_count += 1
                        if done_count % 20 == 0:
                            elapsed = time.time() - start_t
                            rate = done_count / elapsed if elapsed > 0 else 0
                            print(f" {done_count}/{len(pages_needing_upload)}"
                                  f" ({rate:.1f}/s)", end="", flush=True)
                    except Exception as e:
                        errors.append((page_key, str(e)))

            elapsed = time.time() - start_t
            print(f" done ({done_count} in {elapsed:.1f}s)")
            if errors:
                print(f"  WARNING: {len(errors)} upload errors:")
                for pk, err in errors[:5]:
                    print(f"    {pk}: {err}")
            state.save()

        # Build JSONL request file
        print(f"  Building JSONL request...", end="", flush=True)
        page_keys = []
        jsonl_path = os.path.join(output_dir, f"_batch_{volume}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for page_key, _ in pages_to_submit:
                upload_info = state.get_upload(page_key)
                if not upload_info:
                    print(f"\n  WARNING: no upload for {page_key}, skipping")
                    continue
                req = {
                    "key": page_key,
                    "request": {
                        "contents": [{
                            "parts": [
                                {"text": OCR_PROMPT},
                                {"file_data": {
                                    "file_uri": upload_info["uri"],
                                    "mime_type": "image/png",
                                }},
                            ],
                            "role": "user",
                        }],
                    },
                }
                jf.write(json.dumps(req) + "\n")
                page_keys.append(page_key)
        print(f" {len(page_keys)} requests")

        # Upload JSONL to File API
        print(f"  Uploading JSONL...", end="", flush=True)
        jsonl_file = client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(
                display_name=f"batch-{volume}",
                mime_type="jsonl",
            ),
        )
        print(f" {jsonl_file.name}")

        # Create batch job
        display_name = f"roerich-{volume}"
        batch = client.batches.create(
            model=GEMINI_MODEL,
            src=jsonl_file.name,
            config={"display_name": display_name},
        )
        state.record_batch(batch.name, volume, page_keys, display_name)
        state.save()

        # Clean up local JSONL
        os.remove(jsonl_path)

        print(f"  Batch created: {batch.name} ({len(page_keys)} pages)")

    print(f"\nDone. Use '--batch status' to check progress.")


def batch_status(output_dir):
    """Poll and display status of all batch jobs."""
    client = _get_client()
    state = BatchState(output_dir)

    all_batches = state.data["batches"]
    if not all_batches:
        print("No batch jobs found.")
        return

    # Poll active batches
    active = state.get_active_batches()
    for batch_name in active:
        try:
            job = client.batches.get(name=batch_name)
            state.update_batch_state(batch_name, job.state.name)
        except Exception as e:
            print(f"  Warning: could not poll {batch_name}: {e}")
    state.save()

    # Display table
    print(f"\n{'Volume':<12} {'State':<24} {'Pages':>6} {'Created':<20} Batch")
    print("-" * 90)

    for batch_name, info in sorted(
        all_batches.items(), key=lambda x: x[1].get("created_at", "")
    ):
        vol = info.get("volume", "?")
        st = info.get("state", "?")
        pages = len(info.get("page_keys", []))
        created = info.get("created_at", "?")[:19].replace("T", " ")
        print(f"{vol:<12} {st:<24} {pages:>6} {created:<20} {batch_name}")

    # Summary
    states = {}
    for info in all_batches.values():
        s = info.get("state", "?")
        states[s] = states.get(s, 0) + len(info.get("page_keys", []))

    print(f"\nSummary: {len(all_batches)} batches, "
          f"{len(state.data['completed_pages'])} pages collected")
    for s, count in sorted(states.items()):
        print(f"  {s}: {count} pages")


def batch_collect(output_dir):
    """Download results from completed batch jobs and save as JSON."""
    client = _get_client()
    state = BatchState(output_dir)
    json_dir = os.path.join(output_dir, "json")

    # First refresh state of active batches
    for batch_name in list(state.get_active_batches()):
        try:
            job = client.batches.get(name=batch_name)
            state.update_batch_state(batch_name, job.state.name)
        except Exception as e:
            print(f"  Warning: could not poll {batch_name}: {e}")
    state.save()

    succeeded = state.get_succeeded_batches()
    if not succeeded:
        print("No completed batches with uncollected results.")
        active = state.get_active_batches()
        if active:
            total_pages = sum(
                len(info["page_keys"]) for info in active.values()
            )
            print(f"  ({len(active)} batches still running, {total_pages} pages)")
        return

    total_collected = 0
    total_errors = 0

    for batch_name, info in succeeded.items():
        volume = info["volume"]
        vol_json_dir = os.path.join(json_dir, volume)
        os.makedirs(vol_json_dir, exist_ok=True)

        print(f"\n{volume}: collecting from {batch_name}...")

        try:
            job = client.batches.get(name=batch_name)
        except Exception as e:
            print(f"  ERROR getting batch: {e}")
            continue

        # Download result file
        if job.dest and job.dest.file_name:
            print(f"  Downloading results...", end="", flush=True)
            try:
                result_bytes = client.files.download(file=job.dest.file_name)
                result_text = result_bytes.decode("utf-8")
            except Exception as e:
                print(f" ERROR: {e}")
                continue
            print(f" OK")

            # Parse JSONL responses
            collected = 0
            errors = 0
            for line in result_text.splitlines():
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                page_key = parsed.get("key", "")
                json_path = os.path.join(vol_json_dir, f"{page_key}.json")

                if "response" in parsed and parsed["response"]:
                    resp = parsed["response"]
                    # Extract text from response
                    text = None
                    if "candidates" in resp and resp["candidates"]:
                        parts = resp["candidates"][0].get(
                            "content", {}
                        ).get("parts", [])
                        for part in parts:
                            if "text" in part:
                                text = part["text"]
                                break

                    if text:
                        entries = parse_compact_entries(text)
                        if entries:
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    entries, f, ensure_ascii=False, indent=2
                                )
                            # Clean up stale error file
                            err_path = json_path + ".error"
                            if os.path.isfile(err_path):
                                os.remove(err_path)
                            state.mark_completed(page_key)
                            collected += 1
                        else:
                            err_path = json_path + ".error"
                            with open(err_path, "w", encoding="utf-8") as f:
                                f.write(f"ERROR: no entries parsed\n---\n{text}")
                            errors += 1
                    else:
                        err_path = json_path + ".error"
                        with open(err_path, "w", encoding="utf-8") as f:
                            f.write(f"ERROR: empty response text\n---\n"
                                    f"{json.dumps(resp, indent=2)}")
                        state.mark_completed(page_key)  # don't retry blanks
                        errors += 1

                elif "error" in parsed:
                    err_path = json_path + ".error"
                    with open(err_path, "w", encoding="utf-8") as f:
                        f.write(f"ERROR: {parsed['error']}")
                    errors += 1
                else:
                    errors += 1

            total_collected += collected
            total_errors += errors
            print(f"  {collected} pages collected, {errors} errors")

        elif job.dest and job.dest.inlined_responses:
            # Inline responses (small batches)
            collected = 0
            page_keys = info["page_keys"]
            for i, inline_resp in enumerate(job.dest.inlined_responses):
                if i >= len(page_keys):
                    break
                page_key = page_keys[i]
                json_path = os.path.join(vol_json_dir, f"{page_key}.json")

                if inline_resp.response:
                    text = None
                    try:
                        text = inline_resp.response.text
                    except AttributeError:
                        pass

                    if text:
                        entries = parse_compact_entries(text)
                        if entries:
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    entries, f, ensure_ascii=False, indent=2
                                )
                            state.mark_completed(page_key)
                            collected += 1
                            continue

                    state.mark_completed(page_key)
                    total_errors += 1
                elif inline_resp.error:
                    total_errors += 1

            total_collected += collected
            print(f"  {collected} pages collected")
        else:
            print(f"  No results found in batch response")

    state.save()
    print(f"\nTotal: {total_collected} collected, {total_errors} errors")


def batch_retry(output_dir, dpi=300, skip_pages=0):
    """Re-submit failed or expired batch jobs."""
    from google.genai import types

    client = _get_client()
    state = BatchState(output_dir)

    failed = state.get_failed_batches()
    if not failed:
        print("No failed or expired batches to retry.")
        return

    for batch_name, info in failed.items():
        volume = info["volume"]
        page_keys = info["page_keys"]
        print(f"\n{volume}: retrying {len(page_keys)} pages "
              f"(was {info['state']})")

        # Clear old upload refs - files may have expired
        state.clear_uploads_for_pages(page_keys)
        state.remove_batch(batch_name)
        state.save()

    # Now re-submit by finding the PDFs
    # The volumes correspond to PDF filenames in roerich/
    volumes_to_retry = set()
    for info in failed.values():
        volumes_to_retry.add(info["volume"])

    pdf_paths = []
    for vol in volumes_to_retry:
        candidates = [
            f"roerich/{vol}.pdf",
            os.path.join(os.path.dirname(output_dir), "roerich", f"{vol}.pdf"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                pdf_paths.append(c)
                break
        else:
            print(f"  WARNING: could not find PDF for volume {vol}")

    if pdf_paths:
        batch_submit(pdf_paths, output_dir, dpi=dpi, skip_pages=skip_pages)


# ---------------------------------------------------------------------------
# Real-time mode (existing)
# ---------------------------------------------------------------------------

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
                model=GEMINI_MODEL,
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
    client = _get_client()
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        "--batch",
        metavar="CMD",
        choices=["submit", "status", "collect", "retry"],
        help="Batch API mode: submit, status, collect, or retry",
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="PDF file(s) to process (required for real-time and batch submit)",
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
        help="Number of parallel workers (default: 20)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - use roerich_test_output/ instead of production dir",
    )

    args = parser.parse_args()
    output_dir = TEST_OUTPUT_DIR if args.test else DEFAULT_OUTPUT_DIR

    if args.test:
        print(f"*** TEST MODE - output goes to {output_dir}/ ***")

    # Batch mode
    if args.batch:
        if args.batch == "submit":
            if not args.pdfs:
                parser.error("--batch submit requires PDF file(s)")
            batch_submit(
                args.pdfs, output_dir,
                dpi=args.dpi, skip_pages=args.skip_pages, force=args.force,
                workers=args.workers,
            )
        elif args.batch == "status":
            batch_status(output_dir)
        elif args.batch == "collect":
            batch_collect(output_dir)
        elif args.batch == "retry":
            batch_retry(output_dir, dpi=args.dpi, skip_pages=args.skip_pages)
        return

    # Real-time mode
    if not args.pdfs:
        parser.error("PDF file(s) required (or use --batch CMD)")

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
