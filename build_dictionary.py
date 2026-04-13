#!/usr/bin/env python3
"""
build_dictionary.py - Build unified dictionary from per-page JSON files.

Supports both Roerich (chapter subdirs) and Jaeschke (flat dir) layouts.
Loads all OCR'd page JSONs, merges entries that span page boundaries using
explicit markers and heuristics, then exports:
  1. A single merged JSON file
  2. CSVs for SQLite import (one per language field present)
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
    """
    Load all page JSON files in order. Supports two layouts:
    - Chapter subdirs (Roerich): json_dir/<chapter>/<page>.json
    - Flat directory (Jaeschke): json_dir/<page>.json
    Returns list of (filename, entries).
    """
    pages = []

    subdirs = sorted(
        [d for d in os.listdir(json_dir) if os.path.isdir(os.path.join(json_dir, d))],
        key=chapter_sort_key,
    )

    if subdirs:
        for chapter in subdirs:
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
    else:
        files = sorted(
            [f for f in os.listdir(json_dir) if f.endswith(".json")],
            key=page_sort_key,
        )
        for fname in files:
            fpath = os.path.join(json_dir, fname)
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


def join_text(prev_text, curr_text):
    """
    Join two text fragments, fixing page-break hyphenation.
    If prev ends with '- ' or '-' and curr starts with a letter (possibly
    after a leading space), drop the trailing hyphen+space and leading space
    to rejoin the word.
    """
    if not prev_text:
        return curr_text
    if not curr_text:
        return prev_text

    import re
    # Check for hyphenation: prev ends with "- " or "-", curr starts with optional space + letter
    m_prev = re.search(r"-\s*$", prev_text)
    m_curr = re.match(r"\s*", curr_text)
    if m_prev:
        stem = prev_text[: m_prev.start()]
        rest = curr_text[m_curr.end() :]
        return stem + rest

    # Normal join
    if not prev_text.endswith(" ") and not curr_text.startswith(" "):
        return prev_text + " " + curr_text
    return prev_text + curr_text


def merge_entries(prev, curr):
    """
    Merge curr into prev by concatenating text fields.
    Returns the merged entry (mutates prev).
    """
    # Concatenate english
    if curr.get("english"):
        prev["english"] = join_text(prev.get("english", ""), curr["english"])

    # Concatenate russian
    if curr.get("russian"):
        prev["russian"] = join_text(prev.get("russian", ""), curr["russian"])

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
    if curr.get("jaeschke") and not prev.get("jaeschke"):
        prev["jaeschke"] = curr["jaeschke"]

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

    # Final cleanup: remove internal tracking fields
    for entry in merged:
        entry.pop("_continues_next", None)
        # Only include concat_info if entry was actually concatenated
        if "concat_info" not in entry:
            pass  # no concat_info field at all for non-concatenated entries
        # Strip leading/trailing whitespace from text fields
        for field in ("english", "russian", "tibetan", "wylie", "sanskrit", "jaeschke"):
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


def export_csv(entries, output_path, lang_field, lang_col_name, include_sanskrit=False):
    """Export entries to CSV for SQLite import."""
    count = 0
    header = ["wylie", "unicode", lang_col_name]
    if include_sanskrit:
        header.append("sanskrit")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for entry in entries:
            wylie = entry.get("wylie", "")
            tibetan = entry.get("tibetan", "")
            lang_val = entry.get(lang_field, "")
            if not lang_val or not wylie:
                continue
            row = [wylie, tibetan, lang_val]
            if include_sanskrit:
                row.append(entry.get("sanskrit", ""))
            writer.writerow(row)
            count += 1
    print(f"Wrote {count} rows to {output_path}", file=sys.stderr)


def export_markdown(entries, output_path, title):
    """Export entries to a searchable Markdown file."""
    lines = [f"# {title}\n"]

    for entry in entries:
        wylie = entry.get("wylie", "")
        tibetan = entry.get("tibetan", "")

        # Entry header
        if tibetan and wylie:
            lines.append(f"### {tibetan} ({wylie})")
        elif tibetan:
            lines.append(f"### {tibetan}")
        elif wylie:
            lines.append(f"### {wylie}")
        else:
            lines.append("### (unknown headword)")

        if entry.get("jaeschke"):
            lines.append(entry["jaeschke"])
        if entry.get("sanskrit"):
            lines.append(entry["sanskrit"])
        if entry.get("english"):
            lines.append(entry["english"])
        if entry.get("russian"):
            lines.append(entry["russian"])

        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(entries)} entries to {output_path}", file=sys.stderr)


DICT_PRESETS = {
    "roerich": {
        "json_dir": "roerich_output/json",
        "output_dir": "roerich_output",
        "name": "roerich",
        "title": "Roerich Tibetan-Russian-English Dictionary",
        "csv_specs": [
            ("roe.csv", "english", "english", True),
            ("ror.csv", "russian", "russian", False),
        ],
    },
    "jaeschke": {
        "json_dir": "jaeschke_output/json/(1882) Jaeschke - Tib-Eng",
        "output_dir": "jaeschke_output",
        "name": "jaeschke",
        "title": "Jaeschke Tibetan-English Dictionary (1882)",
        "csv_specs": [
            ("jae.csv", "english", "english", True),
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Build unified dictionary from per-page JSON files."
    )
    parser.add_argument(
        "dict",
        nargs="?",
        choices=list(DICT_PRESETS.keys()),
        help="Dictionary preset (roerich or jaeschke). Sets defaults for all options.",
    )
    parser.add_argument(
        "--json-dir",
        help="Directory containing page JSONs (overrides preset)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for merged files (overrides preset)",
    )
    parser.add_argument(
        "--name",
        help="Base name for output files (overrides preset)",
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

    if not args.dict and not args.json_dir:
        parser.error("Specify a dictionary preset or --json-dir")

    preset = DICT_PRESETS.get(args.dict, {})
    json_dir = args.json_dir or preset.get("json_dir", ".")
    output_dir = args.output_dir or preset.get("output_dir", ".")
    name = args.name or preset.get("name", "dictionary")
    title = preset.get("title", f"{name} Dictionary")
    csv_specs = preset.get("csv_specs", [("dictionary.csv", "english", "english", False)])

    os.makedirs(output_dir, exist_ok=True)

    entries = build_merged_dictionary(json_dir)

    if not args.skip_json:
        export_json(entries, os.path.join(output_dir, f"{name}_dictionary.json"))

    if not args.skip_csv:
        for csv_name, lang_field, col_name, sanskrit in csv_specs:
            export_csv(entries, os.path.join(output_dir, csv_name), lang_field, col_name, sanskrit)

    if not args.skip_md:
        export_markdown(entries, os.path.join(output_dir, f"{name}_dictionary.md"), title)


if __name__ == "__main__":
    main()
