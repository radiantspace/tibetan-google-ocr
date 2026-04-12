#!/usr/bin/env python3
"""Merge OCR results from sequential pages, handling entries split across page boundaries.

Reads per-page JSON files and produces a single merged JSON with split entries combined.

Usage:
    # Merge all pages from a single PDF volume
    python merge_pages.py ocr_results/1Ka_page_*_gemini_31_pro.json -o merged/1Ka.json

    # Merge with verbose output showing detected splits
    python merge_pages.py ocr_results/1Ka_page_*_gemini_31_pro.json -o merged/1Ka.json -v

    # Dry run - show what would be merged without writing
    python merge_pages.py ocr_results/1Ka_page_*_gemini_31_pro.json --dry-run
"""

import argparse
import json
import os
import re
import sys


def detect_truncation(entry):
    """Heuristic: detect if an entry's definition appears cut off."""
    signals = []
    for field in ("english", "russian"):
        text = entry.get(field, "")
        if not text:
            continue
        # Strip numbered list markers like 1), 2), etc. before counting parens
        cleaned = re.sub(r"\b\d+\)", "", text)
        open_parens = cleaned.count("(") - cleaned.count(")")
        open_brackets = cleaned.count("[") - cleaned.count("]")
        if open_parens > 0:
            signals.append(f"unmatched '(' in {field}")
        if open_brackets > 0:
            signals.append(f"unmatched '[' in {field}")
        if text and text[-1] not in ".;)!?]\"'":
            signals.append(f"{field} ends without punctuation")
    return signals


def detect_continuation(entry):
    """Heuristic: detect if an entry appears to be a continuation from a previous page."""
    # If entry has a Tibetan headword, it's a new entry - not a continuation
    if entry.get("tibetan", "").strip():
        return []
    signals = []
    signals.append("missing tibetan headword")
    for field in ("english", "russian"):
        text = entry.get(field, "")
        if not text:
            continue
        if text[0].islower():
            signals.append(f"{field} starts lowercase")
        # Strip numbered list markers before counting parens
        cleaned = re.sub(r"\b\d+\)", "", text)
        close_parens = cleaned.count(")") - cleaned.count("(")
        if close_parens > 0:
            signals.append(f"unmatched ')' in {field}")
    return signals


def merge_entry_fields(tail_entry, head_entry):
    """Merge two partial entries into one combined entry."""
    merged = {}
    # Use the headword from whichever entry has one
    for key in ("tibetan", "wylie"):
        tail_val = tail_entry.get(key, "")
        head_val = head_entry.get(key, "")
        merged[key] = tail_val if tail_val else head_val

    # Concatenate text fields
    for key in ("english", "russian", "sanskrit"):
        tail_val = tail_entry.get(key, "")
        head_val = head_entry.get(key, "")
        if tail_val and head_val:
            # Add space if needed between fragments
            if tail_val[-1] in "-":
                merged[key] = tail_val + head_val
            else:
                merged[key] = tail_val + " " + head_val
        elif tail_val:
            merged[key] = tail_val
        elif head_val:
            merged[key] = head_val

    # Remove empty fields
    return {k: v for k, v in merged.items() if v}


def merge_pages(page_files, verbose=False):
    """Merge entries from sequential page files.

    Returns:
        tuple: (merged_entries, merge_log)
    """
    all_entries = []
    merge_log = []
    pending_tail = None
    pending_tail_source = None

    for page_file in page_files:
        with open(page_file, "r", encoding="utf-8") as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError as e:
                merge_log.append(f"WARN: {page_file}: invalid JSON - {e}")
                continue

        if not entries:
            merge_log.append(f"WARN: {page_file}: empty page")
            continue

        first = entries[0]
        last = entries[-1]

        # Check if first entry is a continuation
        is_continuation = first.get("continued_from_prev_page", False)
        if not is_continuation:
            heuristic_signals = detect_continuation(first)
            if heuristic_signals:
                is_continuation = True
                merge_log.append(
                    f"HEURISTIC: {page_file} first entry detected as continuation: "
                    + ", ".join(heuristic_signals)
                )

        # Merge with pending tail from previous page
        if pending_tail and is_continuation:
            merged = merge_entry_fields(pending_tail, first)
            merge_log.append(
                f"MERGED: '{pending_tail.get('tibetan', '?')}' "
                f"from {pending_tail_source} + {page_file}"
            )
            all_entries.append(merged)
            entries = entries[1:]  # skip the continuation entry
        elif pending_tail:
            # No continuation found - add the tail as-is with a flag
            pending_tail["_truncated"] = True
            merge_log.append(
                f"ORPHAN TAIL: '{pending_tail.get('tibetan', '?')}' "
                f"from {pending_tail_source} - no continuation found on {page_file}"
            )
            all_entries.append(pending_tail)
        elif is_continuation and not pending_tail:
            # Continuation without a preceding tail - add with flag
            first["_orphan_head"] = True
            merge_log.append(
                f"ORPHAN HEAD: first entry on {page_file} looks like a continuation "
                f"but no preceding page tail"
            )

        # Check if last entry is truncated
        is_truncated = last.get("continues_next_page", False)
        if not is_truncated:
            heuristic_signals = detect_truncation(last)
            if heuristic_signals:
                is_truncated = True
                merge_log.append(
                    f"HEURISTIC: {page_file} last entry detected as truncated: "
                    + ", ".join(heuristic_signals)
                )

        if is_truncated:
            # Hold the last entry for merging with next page
            pending_tail = last
            pending_tail_source = page_file
            body = entries[:-1] if not is_continuation else entries[:-1]
        else:
            pending_tail = None
            pending_tail_source = None
            body = entries if not is_continuation else entries

        # Add the body entries (excluding first if continuation, last if truncated)
        all_entries.extend(body)

    # Handle any remaining pending tail
    if pending_tail:
        pending_tail["_truncated"] = True
        merge_log.append(
            f"ORPHAN TAIL: '{pending_tail.get('tibetan', '?')}' "
            f"from {pending_tail_source} - last page"
        )
        all_entries.append(pending_tail)

    # Clean up internal flags from output
    for entry in all_entries:
        entry.pop("continues_next_page", None)
        entry.pop("continued_from_prev_page", None)

    return all_entries, merge_log


def natural_sort_key(path):
    """Sort key that handles numeric parts in filenames."""
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Merge OCR results from sequential pages"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Per-page JSON files in order (glob-expanded)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output merged JSON file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show merge log details",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing",
    )

    args = parser.parse_args()

    # Sort files naturally by page number
    page_files = sorted(args.files, key=natural_sort_key)

    print(f"Merging {len(page_files)} page(s)...")
    entries, log = merge_pages(page_files, verbose=args.verbose)

    # Print merge log
    if log:
        print(f"\nMerge log ({len(log)} events):")
        for msg in log:
            print(f"  {msg}")

    # Summary
    truncated = sum(1 for e in entries if e.get("_truncated"))
    orphan_heads = sum(1 for e in entries if e.get("_orphan_head"))
    print(f"\nResult: {len(entries)} entries")
    if truncated:
        print(f"  {truncated} truncated (no continuation found)")
    if orphan_heads:
        print(f"  {orphan_heads} orphan heads (continuation without preceding tail)")

    if args.dry_run:
        return

    # Write output
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output}")
    else:
        print(json.dumps(entries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
