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
import tempfile
import threading
import unittest

from ocr_roerich import BatchState, parse_compact_entries


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


class TestJaeschkeFormat(unittest.TestCase):
    """Tests for Jaeschke dictionary format with J: field."""

    def test_jaeschke_field_parsed(self):
        text = """T:སྐྱེན་པ་
J:skyeN-pa
W:skyen pa
E:adj. 1. quick, swift.
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tibetan"], "སྐྱེན་པ་")
        self.assertEqual(entries[0]["jaeschke"], "skyeN-pa")
        self.assertEqual(entries[0]["wylie"], "skyen pa")
        self.assertEqual(entries[0]["english"], "adj. 1. quick, swift.")

    def test_jaeschke_with_sanskrit(self):
        text = """T:སྐྱེར་པ་
J:skyer-pa
W:skyer pa
E:Lex.: curcuma, turmeric; in W. barberry.
S:harita
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(entries[0]["jaeschke"], "skyer-pa")
        self.assertEqual(entries[0]["sanskrit"], "harita")

    def test_jaeschke_wylie_annotations_in_english(self):
        """E: field should preserve inline Wylie annotations."""
        text = """T:སྐྱེན་པ་
J:skyeN-pa
W:skyen pa
E:adj. 1. quick, swift Lex., kro- (khro-) or sdaN-skyen-pa (sdang skyen pa) quick to wrath.
==="""
        entries = parse_compact_entries(text)
        self.assertIn("(khro-)", entries[0]["english"])
        self.assertIn("(sdang skyen pa)", entries[0]["english"])

    def test_leading_space_stripped_except_english(self):
        """Leading whitespace stripped from T/J/W/S/R but not E."""
        text = """T:  སྐྱེན་པ་
J:  skyeN-pa
W:  skyen pa
E:  adj. quick.
S:  harita
R:  быстрый
==="""
        entries = parse_compact_entries(text)
        self.assertEqual(entries[0]["tibetan"], "སྐྱེན་པ་")
        self.assertEqual(entries[0]["jaeschke"], "skyeN-pa")
        self.assertEqual(entries[0]["wylie"], "skyen pa")
        self.assertEqual(entries[0]["english"], "  adj. quick.")
        self.assertEqual(entries[0]["sanskrit"], "harita")
        self.assertEqual(entries[0]["russian"], "быстрый")

    def test_jaeschke_no_russian(self):
        """Jaeschke entries typically have no Russian field."""
        text = """T:སྐྱེམ་པ་
J:skyem-pa
W:skyem pa
E:resp. to be thirsty.
==="""
        entries = parse_compact_entries(text)
        self.assertNotIn("russian", entries[0])

    def test_jaeschke_backward_compatible(self):
        """Roerich-style entries (no J:) still parse correctly."""
        text = """T:ཀ་པ
W:ka pa
E:first.
R:первый
==="""
        entries = parse_compact_entries(text)
        self.assertNotIn("jaeschke", entries[0])
        self.assertEqual(entries[0]["tibetan"], "ཀ་པ")


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


class TestBatchState(unittest.TestCase):
    """Tests for BatchState load/save/query."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state = BatchState(self.test_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_empty_state(self):
        self.assertEqual(self.state.data["uploaded_files"], {})
        self.assertEqual(self.state.data["batches"], {})
        self.assertEqual(self.state.data["completed_pages"], [])

    def test_save_and_reload(self):
        self.state.record_upload("1Ka_page_0001", "files/abc", "https://x")
        self.state.save()
        reloaded = BatchState(self.test_dir)
        self.assertTrue(reloaded.is_uploaded("1Ka_page_0001"))
        self.assertEqual(reloaded.get_upload("1Ka_page_0001")["file_name"], "files/abc")

    def test_atomic_save(self):
        """Save should not leave partial files on success."""
        self.state.record_upload("p1", "files/a", "https://a")
        self.state.save()
        # File should exist and be valid JSON
        with open(self.state.path) as f:
            data = json.load(f)
        self.assertIn("p1", data["uploaded_files"])

    def test_record_batch_and_query(self):
        self.state.record_batch(
            "batches/123", "1Ka", ["p1", "p2"], "roerich-1Ka"
        )
        self.assertEqual(len(self.state.get_active_batches()), 1)
        self.assertTrue(self.state.is_page_pending("p1"))
        self.assertFalse(self.state.is_page_pending("p3"))

    def test_batch_state_transitions(self):
        self.state.record_batch("batches/1", "1Ka", ["p1"], "test")
        self.state.update_batch_state("batches/1", "JOB_STATE_RUNNING")
        self.assertEqual(len(self.state.get_active_batches()), 1)

        self.state.update_batch_state("batches/1", "JOB_STATE_SUCCEEDED")
        self.assertEqual(len(self.state.get_active_batches()), 0)
        self.assertEqual(len(self.state.get_succeeded_batches()), 1)

    def test_succeeded_batch_excluded_after_collect(self):
        self.state.record_batch("batches/1", "1Ka", ["p1", "p2"], "test")
        self.state.update_batch_state("batches/1", "JOB_STATE_SUCCEEDED")
        self.assertEqual(len(self.state.get_succeeded_batches()), 1)

        # Mark all pages completed
        self.state.mark_completed("p1")
        self.state.mark_completed("p2")
        self.assertEqual(len(self.state.get_succeeded_batches()), 0)

    def test_failed_batches(self):
        self.state.record_batch("batches/1", "1Ka", ["p1"], "test")
        self.state.update_batch_state("batches/1", "JOB_STATE_FAILED")
        self.assertEqual(len(self.state.get_failed_batches()), 1)

        self.state.record_batch("batches/2", "2Kha", ["p2"], "test2")
        self.state.update_batch_state("batches/2", "JOB_STATE_EXPIRED")
        self.assertEqual(len(self.state.get_failed_batches()), 2)

    def test_remove_batch_and_clear_uploads(self):
        self.state.record_upload("p1", "files/a", "https://a")
        self.state.record_batch("batches/1", "1Ka", ["p1"], "test")
        self.state.remove_batch("batches/1")
        self.assertEqual(len(self.state.data["batches"]), 0)

        self.state.clear_uploads_for_pages(["p1"])
        self.assertFalse(self.state.is_uploaded("p1"))

    def test_idempotent_mark_completed(self):
        self.state.mark_completed("p1")
        self.state.mark_completed("p1")  # duplicate
        self.assertEqual(self.state.data["completed_pages"].count("p1"), 1)

    def test_is_completed(self):
        self.assertFalse(self.state.is_completed("p1"))
        self.state.mark_completed("p1")
        self.assertTrue(self.state.is_completed("p1"))

    def test_record_upload_and_save_periodic(self):
        """record_upload_and_save flushes every save_every uploads."""
        counter = [0]
        for i in range(25):
            self.state.record_upload_and_save(
                f"p{i}", f"files/{i}", f"https://{i}",
                save_every=10, counter=counter,
            )
        self.assertEqual(counter[0], 25)
        # Should have saved at 10 and 20
        reloaded = BatchState(self.test_dir)
        self.assertEqual(len(reloaded.data["uploaded_files"]), 20)
        # Final save captures all 25
        self.state.save()
        reloaded2 = BatchState(self.test_dir)
        self.assertEqual(len(reloaded2.data["uploaded_files"]), 25)


class TestBatchStateThreadSafety(unittest.TestCase):
    """Tests for concurrent BatchState access."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state = BatchState(self.test_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_concurrent_record_uploads(self):
        """Many threads recording uploads should not lose data."""
        num_threads = 10
        uploads_per_thread = 50
        errors = []

        def _worker(thread_id):
            try:
                for i in range(uploads_per_thread):
                    pk = f"t{thread_id}_p{i}"
                    self.state.record_upload(pk, f"files/{pk}", f"https://{pk}")
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(num_threads):
            th = threading.Thread(target=_worker, args=(t,))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])
        expected = num_threads * uploads_per_thread
        self.assertEqual(len(self.state.data["uploaded_files"]), expected)

    def test_concurrent_record_upload_and_save(self):
        """Parallel record_upload_and_save should not corrupt state."""
        num_threads = 8
        uploads_per_thread = 30
        counter = [0]
        errors = []

        def _worker(thread_id):
            try:
                for i in range(uploads_per_thread):
                    pk = f"t{thread_id}_p{i}"
                    self.state.record_upload_and_save(
                        pk, f"files/{pk}", f"https://{pk}",
                        save_every=10, counter=counter,
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(num_threads):
            th = threading.Thread(target=_worker, args=(t,))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])
        expected = num_threads * uploads_per_thread
        self.assertEqual(counter[0], expected)
        self.assertEqual(len(self.state.data["uploaded_files"]), expected)

        # Final save and reload should have all data
        self.state.save()
        reloaded = BatchState(self.test_dir)
        self.assertEqual(len(reloaded.data["uploaded_files"]), expected)

    def test_concurrent_mixed_operations(self):
        """Mix of uploads, batch records, and completions."""
        errors = []

        def _uploader():
            try:
                for i in range(30):
                    self.state.record_upload(f"up_{i}", f"files/{i}", f"https://{i}")
            except Exception as e:
                errors.append(e)

        def _batch_recorder():
            try:
                for i in range(10):
                    self.state.record_batch(
                        f"batches/{i}", f"vol{i}", [f"p{i}"], f"test-{i}"
                    )
            except Exception as e:
                errors.append(e)

        def _completer():
            try:
                for i in range(20):
                    self.state.mark_completed(f"done_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_uploader),
            threading.Thread(target=_batch_recorder),
            threading.Thread(target=_completer),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(self.state.data["uploaded_files"]), 30)
        self.assertEqual(len(self.state.data["batches"]), 10)
        self.assertEqual(len(self.state.data["completed_pages"]), 20)


class TestBatchCollectParsing(unittest.TestCase):
    """Test parsing of batch API JSONL response format."""

    def test_parse_batch_response_line(self):
        """Simulate a single JSONL response line from batch API."""
        response_line = json.dumps({
            "key": "1Ka_page_0002",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": "T:ཀ་པ\nW:ka pa\nE:first.\nR:первый\n==="
                        }],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }],
            },
        })

        parsed = json.loads(response_line)
        text = parsed["response"]["candidates"][0]["content"]["parts"][0]["text"]
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tibetan"], "ཀ་པ")

    def test_parse_batch_error_response(self):
        """Batch API error response should be handled gracefully."""
        response_line = json.dumps({
            "key": "1Ka_page_0076",
            "error": {"code": 400, "message": "Invalid request"},
        })
        parsed = json.loads(response_line)
        self.assertIn("error", parsed)
        self.assertNotIn("response", parsed)

    def test_parse_batch_empty_response(self):
        """Batch API can return empty candidates."""
        response_line = json.dumps({
            "key": "1Ka_page_0001",
            "response": {
                "candidates": [{
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "SAFETY",
                }],
            },
        })
        parsed = json.loads(response_line)
        parts = parsed["response"]["candidates"][0]["content"]["parts"]
        text = None
        for part in parts:
            if "text" in part:
                text = part["text"]
        self.assertIsNone(text)

    def test_parse_multi_entry_batch_response(self):
        """Full page with multiple entries from batch."""
        compact = """T:ཀ་ར།
W:ka ra
E:I 1) sugar
R:I 1) сахар
S:sharkaraa
===
T:ཀ་ར།
W:ka ra
E:II tent-pole.
R:II шест палатки
==="""
        response_line = json.dumps({
            "key": "1Ka_page_0007",
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{"text": compact}],
                        "role": "model",
                    },
                }],
            },
        })
        parsed = json.loads(response_line)
        text = parsed["response"]["candidates"][0]["content"]["parts"][0]["text"]
        entries = parse_compact_entries(text)
        self.assertEqual(len(entries), 2)
        self.assertIn("sugar", entries[0]["english"])
        self.assertIn("tent-pole", entries[1]["english"])


if __name__ == "__main__":
    unittest.main()
