#!/usr/bin/env python3
"""OCR a Roerich dictionary PDF volume into structured JSON.

Idempotent - skips pages that already have JSON output unless --force is used.
Organizes output into roerich_output/pages/ (PNG) and roerich_output/json/ (JSON).

Usage:
    # Process a single volume
    python ocr_roerich.py roerich/1Ka.pdf

    # Process all volumes
    python ocr_roerich.py roerich/*.pdf

    # Force re-OCR of already processed pages
    python ocr_roerich.py roerich/1Ka.pdf --force

    # Skip first N pages (e.g. blank/title pages)
    python ocr_roerich.py roerich/1Ka.pdf --skip-pages 0

    # Custom DPI
    python ocr_roerich.py roerich/1Ka.pdf --dpi 400
"""

import argparse
import json
import os
import re
import sys
import time

import fitz  # pymupdf

OUTPUT_DIR = "roerich_output"
PAGES_DIR = os.path.join(OUTPUT_DIR, "pages")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

OCR_PROMPT = """OCR this dictionary page and extract structured entries as a JSON array.
This is from an old Tibetan-English-Russian-Sanskrit dictionary.

For each dictionary entry on the page, output a JSON object with these fields:
- "tibetan": the headword in Tibetan script
- "wylie": Wylie transliteration if you can determine it
- "english": English definition/translation
- "russian": Russian translation (omit field if not present on this page)
- "sanskrit": Sanskrit/Devanagari equivalent (omit field if not present)

IMPORTANT - handle entries that span page boundaries:
- If the FIRST entry on the page has NO headword and starts mid-definition
  (it is a continuation from a previous page), set "continued_from_prev_page": true
  and put the partial text in the appropriate language fields. The "tibetan" field
  should be empty string "" if no headword is visible.
- If the LAST entry on the page appears CUT OFF (definition ends mid-sentence,
  has unclosed parentheses, or clearly continues), set "continues_next_page": true

Preserve all diacritical marks. Output ONLY the JSON array, no markdown fences or commentary.

Example:
[{
    "continued_from_prev_page": true,
    "tibetan": "",
    "wylie": "",
    "english": "of the monk Katyayana.",
    "russian": "монаха Катьяяна."
  },
  {
    "tibetan": "ཐ་སྐར",
    "wylie": "tha skar",
    "english": "stars beta and gamma Aries",
    "russian": "звезды бета и гамма созвездия Овен",
    "sanskrit": "ashvini"
  },
  {
    "tibetan": "ཐ་ཆེན",
    "wylie": "tha chen",
    "english": "1) building with columns; 2) supporter (one of the four",
    "russian": "1) здание с колоннами; 2) приверженец (один из четырёх",
    "continues_next_page": true
  }]"""


def ocr_page_gemini(image_path, client):
    """OCR a single page image using Gemini 3.1 Pro."""
    from google.genai import types

    with open(image_path, "rb") as f:
        image_data = f.read()

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[
            types.Part.from_text(text=OCR_PROMPT),
            types.Part.from_bytes(data=image_data, mime_type="image/png"),
        ],
    )
    return response.text


def strip_markdown_fences(text):
    """Strip markdown code fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()
    return cleaned


def extract_page(doc, page_num, output_path, dpi=300):
    """Extract a single page from a PDF as PNG."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)


def process_volume(pdf_path, force=False, dpi=300, skip_pages=0):
    """Process all pages of a single PDF volume."""
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    volume = os.path.splitext(os.path.basename(pdf_path))[0]

    vol_pages_dir = os.path.join(PAGES_DIR, volume)
    vol_json_dir = os.path.join(JSON_DIR, volume)
    os.makedirs(vol_pages_dir, exist_ok=True)
    os.makedirs(vol_json_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total = len(doc)
    start_page = skip_pages + 1

    print(f"\n{'='*60}")
    print(f"Volume: {volume} ({total} pages, starting at page {start_page})")
    print(f"Pages:  {vol_pages_dir}/")
    print(f"JSON:   {vol_json_dir}/")
    print(f"{'='*60}")

    processed = 0
    skipped = 0
    errors = 0
    error_pages = []

    for page_num in range(start_page, total + 1):
        page_name = f"{volume}_page_{page_num:04d}"
        png_path = os.path.join(vol_pages_dir, f"{page_name}.png")
        json_path = os.path.join(vol_json_dir, f"{page_name}.json")

        # Skip if already processed (idempotent)
        if os.path.isfile(json_path) and os.path.getsize(json_path) > 0 and not force:
            skipped += 1
            continue

        # Extract page image if needed
        if not os.path.isfile(png_path) or force:
            extract_page(doc, page_num, png_path, dpi)

        # OCR
        print(f"  [{page_num}/{total}] OCR...", end=" ", flush=True)
        start = time.time()
        try:
            raw = ocr_page_gemini(png_path, client)
            cleaned = strip_markdown_fences(raw)

            # Validate JSON
            json.loads(cleaned)

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            elapsed = time.time() - start
            print(f"OK ({elapsed:.0f}s, {len(cleaned)} chars)")
            processed += 1
        except json.JSONDecodeError as e:
            elapsed = time.time() - start
            # Save raw output for debugging
            err_path = json_path + ".error"
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(raw)
            print(f"BAD JSON ({elapsed:.0f}s) - saved raw to {err_path}")
            errors += 1
            error_pages.append(page_num)
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.0f}s): {e}")
            errors += 1
            error_pages.append(page_num)

    doc.close()

    print(f"\n--- {volume} summary ---")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped} (already done)")
    print(f"  Errors:    {errors}")
    if error_pages:
        print(f"  Error pages: {error_pages}")

    return processed, skipped, errors


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

    args = parser.parse_args()

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for pdf_path in sorted(args.pdfs):
        if not os.path.isfile(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping")
            continue
        p, s, e = process_volume(
            pdf_path, force=args.force, dpi=args.dpi, skip_pages=args.skip_pages
        )
        total_processed += p
        total_skipped += s
        total_errors += e

    if len(args.pdfs) > 1:
        print(f"\n{'='*60}")
        print(f"TOTAL: {total_processed} processed, {total_skipped} skipped, {total_errors} errors")


if __name__ == "__main__":
    main()
