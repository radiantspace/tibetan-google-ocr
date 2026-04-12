#!/usr/bin/env python3
"""Tests for parse_compact_entries and OCR prompt output handling.

Covers:
- Basic single/multi-entry parsing
- Page boundary flags (>, <)
- Multi-line field values
- Markdown fence stripping
- Edge cases: empty blocks, missing fields, colons in values
- Real Gemini output from actual dictionary pages
"""

import json
import unittest

from ocr_roerich import parse_compact_entries


class TestBasicParsing(unittest.TestCase):
    """Core parsing functionality."""

    def test_single_entry(self):
        text = """T:ཀ་ཀ་ནི་ལ
W:ka ka ni la
E:sapphire.
R:сапфир
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tibetan"], "ཀ་ཀ་ནི་ལ")
        self.assertEqual(entries[0]["wylie"], "ka ka ni la")
        self.assertEqual(entries[0]["english"], "sapphire.")
        self.assertEqual(entries[0]["russian"], "сапфир")

    def test_multiple_entries(self):
        text = """T:ཀ་པ
W:ka pa
E:first (volume of a series).
R:первый (том серии)
===
T:ཀ་པ་ལ
W:ka pa la
E:1) skull; 2) a cup made of the cranium.
R:1) череп; 2) чаша из черепа
S:kapala
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["tibetan"], "ཀ་པ")
        self.assertEqual(entries[1]["sanskrit"], "kapala")

    def test_missing_optional_fields(self):
        text = """T:ཀ་ལམ་པ།
W:ka lam pa/
R:см. ཀ་ལམ་བ།
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertNotIn("english", entries[0])
        self.assertNotIn("sanskrit", entries[0])
        self.assertEqual(entries[0]["russian"], "см. ཀ་ལམ་བ།")

    def test_entry_with_all_fields(self):
        text = """T:ཀ་པ་ལ
W:ka pa la
E:skull; cup made of the cranium.
R:череп; чаша из черепа
S:kapala
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        for key in ("tibetan", "wylie", "english", "russian", "sanskrit"):
            self.assertIn(key, entries[0])

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(parse_compact_entries(""), [])
        self.assertEqual(parse_compact_entries("   "), [])
        self.assertEqual(parse_compact_entries("\n\n"), [])

    def test_none_returns_empty_list(self):
        """Gemini can return None for response.text - must not crash."""
        self.assertEqual(parse_compact_entries(None), [])


class TestPageBoundaryFlags(unittest.TestCase):
    """Continuation and truncation flags."""

    def test_continued_from_prev_page(self):
        text = """>
E:of the monk Katyayana.
R:монаха Катьяяна.
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["continued_from_prev_page"])
        self.assertEqual(entries[0]["english"], "of the monk Katyayana.")
        self.assertNotIn("tibetan", entries[0])

    def test_continues_next_page(self):
        text = """T:ཐ་ཆེན
W:tha chen
E:1) building with columns; 2) supporter (one of the four
R:1) здание с колоннами; 2) приверженец (один из четырёх
<
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["continues_next_page"])
        self.assertEqual(entries[0]["tibetan"], "ཐ་ཆེན")

    def test_continuation_with_tibetan_headword(self):
        text = """>
T:ཀ་ཆེན།
W:ka chen
E:the remainder of the definition.
R:остаток определения
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["continued_from_prev_page"])
        self.assertEqual(entries[0]["tibetan"], "ཀ་ཆེན།")

    def test_both_flags_single_entry(self):
        """A page with a single entry that is both a continuation and truncated."""
        text = """>
E:middle part of a very long definition that spans three pages
R:средняя часть очень длинного определения
<
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["continued_from_prev_page"])
        self.assertTrue(entries[0]["continues_next_page"])

    def test_mixed_page_boundary_entries(self):
        text = """>
E:ear ornament.
R:ные украшения, серьги
===
T:ཀ་པ
W:ka pa
E:first (volume of a series).
R:первый (том серии)
===
T:ཀ་པོ་ཏ
W:ka po ta
E:1) pigeon; dove; 2) roof pinnacle (one of
R:1) голубь; 2) навершие крыши (один из
<
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 3)
        self.assertTrue(entries[0].get("continued_from_prev_page", False))
        self.assertFalse(entries[1].get("continued_from_prev_page", False))
        self.assertFalse(entries[1].get("continues_next_page", False))
        self.assertTrue(entries[2].get("continues_next_page", False))


class TestMarkdownFences(unittest.TestCase):
    """Gemini sometimes wraps output in markdown fences."""

    def test_strip_json_fence(self):
        text = """```
T:ཀ་པ
W:ka pa
E:first.
R:первый
===
```"""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tibetan"], "ཀ་པ")

    def test_strip_language_tagged_fence(self):
        text = """```text
T:ཀ་པ
W:ka pa
E:first.
R:первый
===
```"""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tibetan"], "ཀ་པ")

    def test_no_fences_passes_through(self):
        text = """T:ཀ་པ
W:ka pa
E:first.
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)


class TestEdgeCases(unittest.TestCase):
    """Tricky formats the model might produce."""

    def test_colons_in_values(self):
        text = """T:ཀ་ར།
W:ka ra
E:I 1) sugar; ~དཀར་པོ། white sugar; ~དཀར་སྨུག། brown sugar
R:I 1) сахар; ~དཀར་པོ། сахар-рафинад; ~དཀར་སྨུག། неочищенный сахар
S:sharkaraa, khanda, sitaa
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertIn("white sugar", entries[0]["english"])

    def test_roman_numeral_entries(self):
        """Entries with Roman numeral sub-meanings (I, II, III)."""
        text = """T:ཀ་པི
W:ka pi
E:I myth. divine language
R:I миф. божественный язык
===
T:ཀ་པི
W:ka pi
E:II 1) gum; resin; 2) bot. Spondias magnifera
R:II 1) камедь, гумми; смола; 2) бот. Spondias magnifera
===
T:ཀ་པི
W:ka pi
E:III monkey.
R:III обезьяна
S:kapi
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 3)
        self.assertIn("divine language", entries[0]["english"])
        self.assertIn("gum", entries[1]["english"])
        self.assertEqual(entries[2]["sanskrit"], "kapi")

    def test_empty_blocks_between_separators(self):
        text = """T:ཀ་པ
W:ka pa
E:first.
===
===
T:ཀ་པ་ལ
W:ka pa la
E:skull.
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 2)

    def test_no_trailing_separator(self):
        """Last entry without trailing ===."""
        text = """T:ཀ་པ
W:ka pa
E:first.
R:первый"""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tibetan"], "ཀ་པ")

    def test_multiline_field_value(self):
        """Model wraps a long value across lines."""
        text = """T:ཀ་པ
W:ka pa
E:a very long definition that the model has wrapped
across multiple lines for some reason
R:первый
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertIn("wrapped across multiple lines", entries[0]["english"])

    def test_brackets_in_wylie(self):
        text = """T:ཀ་ལན་ད[ཀ]
W:ka lan da [ka]
E:little bird; sparrow.
R:пташка; воробей
S:kalantaka
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertIn("[ཀ]", entries[0]["tibetan"])

    def test_slash_in_wylie(self):
        text = """T:ཀ་ཧཾ་ས།
W:ka haM sa/
E:n. of several species of the goose.
R:(назв. водоплавающей птицы)
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["wylie"], "ka haM sa/")

    def test_cross_reference_entry(self):
        """Entries that just say 'see X'."""
        text = """T:ཀ་པ་ལི
W:ka pa li
E:see ཀ་པ་ལ།
R:см. ཀ་པ་ལ།
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertIn("see", entries[0]["english"])

    def test_only_tibetan_and_wylie(self):
        """Entry with no definitions, just headword."""
        text = """T:ཀ་པི་ཏ
W:ka pi ta
R:см. ཀ་པི། II 1).
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tibetan"], "ཀ་པི་ཏ")
        self.assertNotIn("english", entries[0])

    def test_blank_lines_within_block(self):
        """Model inserts blank lines between fields."""
        text = """T:ཀ་པ
W:ka pa

E:first.

R:первый
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["english"], "first.")


class TestRealGeminiOutput(unittest.TestCase):
    """Test against actual Gemini compact-format output from error files."""

    def test_page_0005_real_output(self):
        """Partial real output from page 5 of 1Ka.pdf."""
        text = """>
E:ear ornament.
R:ные украшения, серьги
===
T:ཀ་པ
W:ka pa
E:first(volume of a series); first (chapter).
R:первый (том серии); первая (глава)
===
T:ཀ་པ་ལ
W:ka pa la
E:1) skull; 2) a cup made of the cranium; 3) the forehead.
R:1) череп; 2) чаша из черепа; 3) лоб
S:kapala
===
T:ཀ་པི
W:ka pi
E:I myth. divine language, in which the Bon "Royal-rabs" was compiled.
R:I миф. божественный язык, на котором была составлена бонская "История царей Тибета"
===
T:ཀ་པི
W:ka pi
E:II 1) gum; resin; 2) bot. Spondias magnifera; 3) bot. Pentaptira tomentosa; 4) yellow or-pigment.
R:II 1) камедь, гумми; смола; 2) бот. Spondias magnifera; 3) бот. Pentaptira tomentosa; 4) жёлтый краситель, аурипигмент
===
T:ཀ་པི
W:ka pi
E:III monkey.
R:III обезьяна
S:kapi
==="""
        entries = parse_compact_entries(text)
        self.assertGreater(len(entries), 5)
        # First entry is a continuation
        self.assertTrue(entries[0]["continued_from_prev_page"])
        self.assertNotIn("tibetan", entries[0])
        # Second entry is normal
        self.assertEqual(entries[1]["tibetan"], "ཀ་པ")
        # Third entry has Sanskrit
        self.assertEqual(entries[2]["sanskrit"], "kapala")
        # Roman numeral entries
        self.assertIn("myth.", entries[3]["english"])
        self.assertIn("gum", entries[4]["english"])

    def test_page_0008_real_output_truncated_end(self):
        """Page with truncated last entry - no trailing ===."""
        text = """T:ཀ་ཧཾ་ས།
W:ka haM sa/
E:n. of several species of the goose.
R:(назв. водоплавающей птицы)
S:kalahamsa
===
T:ཀ་ལག
W:ka lag
E:earth and water used instead of mortar.
R:смесь воды с землёй, употребляемая как штукатурка
===
T:ཀ་ལིབ།
W:ka lib/"""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[2]["tibetan"], "ཀ་ལིབ།")
        self.assertEqual(entries[2]["wylie"], "ka lib/")

    def test_output_is_valid_json_serializable(self):
        """Ensure all parsed entries can be serialized to JSON."""
        text = """>
E:of the monk Katyayana.
R:монаха Катьяяна.
===
T:ཀ་པི་ཀཙྪུ
W:ka pi kats+tshu
E:bot. Mucuna pruritus.
R:бот. Mucuna pruritus.
S:kapikacchu
==="""
        entries = parse_compact_entries(text)
        # Should not raise
        json_str = json.dumps(entries, ensure_ascii=False)
        # Should round-trip cleanly
        roundtrip = json.loads(json_str)
        self.assertEqual(roundtrip, entries)


if __name__ == "__main__":
    unittest.main()
