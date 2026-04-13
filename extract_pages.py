#!/usr/bin/env python3
"""Extract pages from dictionary PDFs as high-resolution images for OCR testing.

Usage:
    python3 extract_pages.py <pdf_path> [--pages 1,5,10] [--output-dir ./test_pages] [--dpi 300]

Examples:
    # Extract specific pages from Roerich dictionary
    python3 extract_pages.py ~/dictionaries/roerich.pdf --pages 1,50,100

    # Extract first 3 pages at high DPI
    python3 extract_pages.py ~/dictionaries/das.pdf --pages 1,2,3 --dpi 400

    # Extract all pages (careful - can be large)
    python3 extract_pages.py ~/dictionaries/jaschke.pdf --all --output-dir ./jaschke_pages
"""

import argparse
import os
import sys

import fitz  # pymupdf
from PIL import Image


def extract_pages(pdf_path, page_numbers, output_dir, dpi=300):
    """Extract specified pages from a PDF as PNG images.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: List of 1-based page numbers to extract.
        output_dir: Directory to save extracted images.
        dpi: Resolution for rendering (default 300).

    Returns:
        List of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    basename = os.path.splitext(os.path.basename(pdf_path))[0]

    output_paths = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in page_numbers:
        if page_num < 1 or page_num > total_pages:
            print(f"Warning: page {page_num} out of range (1-{total_pages}), skipping")
            continue

        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=mat)
        output_path = os.path.join(output_dir, f"{basename}_page_{page_num:04d}.png")
        pix.save(output_path)
        output_paths.append(output_path)

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Extracted page {page_num}/{total_pages} -> {output_path} ({size_kb:.0f} KB)")

    doc.close()
    return output_paths


def get_page_info(pdf_path):
    """Print basic info about a PDF and return (total_pages, doc) - caller must close doc."""
    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"PDF: {pdf_path}")
    print(f"Pages: {total}")
    if total > 0:
        page = doc[0]
        print(f"Page size: {page.rect.width:.0f} x {page.rect.height:.0f} pts")
    return total, doc


def main():
    parser = argparse.ArgumentParser(
        description="Extract pages from dictionary PDFs as images for OCR testing"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-separated 1-based page numbers (e.g., 1,50,100)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all pages",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_pages",
        help="Output directory (default: ./test_pages)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering DPI (default: 300)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Just print PDF info without extracting",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"Error: file not found: {args.pdf_path}")
        sys.exit(1)

    total, doc = get_page_info(args.pdf_path)
    doc.close()

    if args.info:
        return

    if args.all:
        page_numbers = list(range(1, total + 1))
    elif args.pages:
        page_numbers = [int(p.strip()) for p in args.pages.split(",")]
    else:
        # Default: extract first page, a middle page, and a late page
        page_numbers = [1]
        if total > 10:
            page_numbers.append(total // 4)
        if total > 20:
            page_numbers.append(total // 2)
        print(f"No --pages specified, extracting sample pages: {page_numbers}")

    print(f"\nExtracting {len(page_numbers)} page(s) at {args.dpi} DPI...")
    paths = extract_pages(args.pdf_path, page_numbers, args.output_dir, args.dpi)
    print(f"\nDone. {len(paths)} image(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
