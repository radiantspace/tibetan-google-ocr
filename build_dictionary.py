#!/usr/bin/env python3
"""
build_dictionary.py - Build unified Roerich dictionary from per-page JSON files.

Loads all OCR'd page JSONs, merges entries that span page boundaries using
explicit markers and heuristics, then exports:
  1. A single merged JSON file
  2. Two CSVs for SQLite import (tib-eng and tib-rus)
  3. A searchable Markdown file
"""

import argparse
import csv
import json
import os
import re
import sys


# Chapter ordering: sort by numeric prefix
def chapter_sort_key(dirname):
    m = re.match(r"(\d+)", dirname)
    return int(m.group(1)) if m else 999


def page_sort_key(filename):
    m = re.search(r"page_(\d+)", filename)
    return int(m.group(1)) if m else 999


def load_all_pages(json_dir):
    """Load all page JSON files in chapter/page order. Returns list of (filename, entries)."""
    pages = []
    chapters = sorted(
        [d for d in os.listdir(json_dir) if os.path.isdir(os.path.join(json_dir, d))],
        key=chapter_sort_key,
    )
    for chapter in chapters:
        chapter_dir = os.path.join(json_dir, chapter)
        files = sorted(
            [f for f in os.listdir(chapter_dir) if f.endswith(".json")],
            key=page_sort_key,
        )
        for fname in files:
            fpath = os.path.join(chapter_dir, fname)
            with open(fpath, encoding="utf-8") as fh:
                data = json.load(fh)
            pages.append((fname, data))
    return pages


def should_concatenate(prev_entry, curr_entry, prev_has_continues, curr_has_continued):
    """
    Decide whether curr_entry should be concatenated onto prev_entry.
    Returns (should_concat: bool, concat_reason: str or None).
    """
    if prev_has_continues and curr_has_continued:
        return True, "explicit_both"

    if prev_has_continues and not curr_has_continued:
        # Prev says it continues but next doesn't acknowledge it.
        # Conservative: only merge if next entry has no headword at all (pure text).
        # Sub-entries with their own wylie (even space-prefixed) are separate entries.
        if not curr_entry.get("tibetan") and not curr_entry.get("wylie"):
            return True, "missing_continued_from_no_headword"
        # Entry has its own headword - don't merge, it's a separate entry
        return False, None

    if not prev_has_continues and curr_has_continued:
        # Next says it's continued but prev doesn't say it continues.
        # Trust the continued_from marker - OCR missed the continues_next marker.
        return True, "missing_continues_next"

    return False, None


def merge_entries(prev, curr):
    """
    Merge curr into prev by concatenating text fields.
    Returns the merged entry (mutates prev).
    """
    # Concatenate english
    if curr.get("english"):
        prev_eng = prev.get("english", "")
        curr_eng = curr["english"]
        if prev_eng and not prev_eng.endswith(" ") and not curr_eng.startswith(" "):
            prev["english"] = prev_eng + " " + curr_eng
        else:
            prev["english"] = prev_eng + curr_eng

    # Concatenate russian
    if curr.get("russian"):
        prev_rus = prev.get("russian", "")
        curr_rus = curr["russian"]
        if prev_rus and not prev_rus.endswith(" ") and not curr_rus.startswith(" "):
            prev["russian"] = prev_rus + " " + curr_rus
        else:
            prev["russian"] = prev_rus + curr_rus

    # Concatenate sanskrit if present in curr but not in prev, or append
    if curr.get("sanskrit"):
        if prev.get("sanskrit"):
            prev["sanskrit"] = prev["sanskrit"] + "; " + curr["sanskrit"]
        else:
            prev["sanskrit"] = curr["sanskrit"]

    # Handle tibetan/wylie for sub-entries that have their own headword fragment
    if curr.get("tibetan") and not prev.get("tibetan"):
        prev["tibetan"] = curr["tibetan"]
    if curr.get("wylie") and not prev.get("wylie"):
        prev["wylie"] = curr["wylie"]

    return prev


def build_merged_dictionary(json_dir):
    """
    Load all pages and merge continuation entries.
    Returns list of merged dictionary entries.
    """
    pages = load_all_pages(json_dir)
    merged = []
    stats = {"explicit_both": 0, "missing_continued_from_no_headword": 0,
             "missing_continues_next": 0, "no_concat": 0}

    for page_fname, page_entries in pages:
        for i, entry in enumerate(page_entries):
            is_first = i == 0
            curr_has_continued = entry.get("continued_from_prev_page", False)

            # Clean entry: remove continuation markers from output
            clean = {k: v for k, v in entry.items()
                     if k not in ("continues_next_page", "continued_from_prev_page")}

            if merged and (is_first or curr_has_continued):
                prev = merged[-1]
                prev_has_continues = prev.get("_continues_next", False)

                do_concat, reason = should_concatenate(
                    prev, clean, prev_has_continues, curr_has_continued
                )

                if do_concat:
                    stats[reason] += 1
                    merge_entries(prev, clean)
                    # Track concat info
                    if "concat_info" not in prev:
                        prev["concat_info"] = []
                    prev["concat_info"].append(reason)
                    # Track source pages
                    if page_fname not in prev.get("source_pages", []):
                        prev["source_pages"].append(page_fname)
                    continue

            # Not concatenated - new entry
            stats["no_concat"] += 1
            clean["source_pages"] = [page_fname]
            # Track if this entry continues to next
            if entry.get("continues_next_page"):
                clean["_continues_next"] = True
            merged.append(clean)

    # Final cleanup: remove internal tracking fields, finalize concat_info
    for entry in merged:
        entry.pop("_continues_next", None)
        if "concat_info" not in entry:
            entry["concat_info"] = None
        # Strip leading/trailing whitespace from text fields
        for field in ("english", "russian", "tibetan", "wylie", "sanskrit"):
            if entry.get(field):
                entry[field] = entry[field].strip()

    print(f"Merge statistics:", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"Total merged entries: {len(merged)}", file=sys.stderr)

    return merged


def export_json(entries, output_path):
    """Export merged entries to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(entries)} entries to {output_path}", file=sys.stderr)


def export_csv(entries, output_path, lang_field, lang_col_name):
    """Export entries to CSV for SQLite import."""
    count = 0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wylie", "unicode", lang_col_name])
        for entry in entries:
            wylie = entry.get("wylie", "")
            tibetan = entry.get("tibetan", "")
            lang_val = entry.get(lang_field, "")
            if not lang_val or not wylie:
                continue
            writer.writerow([wylie, tibetan, lang_val])
            count += 1
    print(f"Wrote {count} rows to {output_path}", file=sys.stderr)


def export_markdown(entries, output_path):
    """Export entries to a searchable Markdown file."""
    current_letter = None
    lines = ["# Roerich Tibetan-Russian-English Dictionary\n"]

    for entry in entries:
        wylie = entry.get("wylie", "")
        tibetan = entry.get("tibetan", "")

        # Detect chapter letter change (first letter of wylie, uppercase)
        first_letter = wylie.lstrip(" ~").upper()[:1] if wylie else ""
        if first_letter and first_letter != current_letter:
            current_letter = first_letter
            lines.append(f"\n## {current_letter}\n")

        # Entry header
        if tibetan and wylie:
            lines.append(f"### {tibetan} ({wylie})")
        elif tibetan:
            lines.append(f"### {tibetan}")
        elif wylie:
            lines.append(f"### {wylie}")
        else:
            lines.append("### (unknown headword)")

        # Sanskrit
        if entry.get("sanskrit"):
            lines.append(f"*Skt.* {entry['sanskrit']}  ")

        # English
        if entry.get("english"):
            lines.append(f"**English:** {entry['english']}  ")

        # Russian
        if entry.get("russian"):
            lines.append(f"**Russian:** {entry['russian']}  ")

        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(entries)} entries to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Build unified Roerich dictionary from per-page JSON files."
    )
    parser.add_argument(
        "--json-dir",
        default="roerich_output/json",
        help="Directory containing chapter subdirs with page JSONs (default: roerich_output/json)",
    )
    parser.add_argument(
        "--output-dir",
        default="roerich_output",
        help="Output directory for merged files (default: roerich_output)",
    )
    parser.add_argument(
        "--skip-json", action="store_true", help="Skip JSON output"
    )
    parser.add_argument(
        "--skip-csv", action="store_true", help="Skip CSV output"
    )
    parser.add_argument(
        "--skip-md", action="store_true", help="Skip Markdown output"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    entries = build_merged_dictionary(args.json_dir)

    if not args.skip_json:
        export_json(entries, os.path.join(args.output_dir, "roerich_dictionary.json"))

    if not args.skip_csv:
        export_csv(entries, os.path.join(args.output_dir, "roe.csv"), "english", "english")
        export_csv(entries, os.path.join(args.output_dir, "ror.csv"), "russian", "russian")

    if not args.skip_md:
        export_markdown(entries, os.path.join(args.output_dir, "roerich_dictionary.md"))


if __name__ == "__main__":
    main()
