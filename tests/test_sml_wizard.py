"""Tests for :mod:`lib.sml_wizard` — the session-driven book-to-SML wizard.

The module drives the ported pipeline through a plain ``dict`` session, so a
test needs no Gradio and no real BookNLP/LLM: the pipeline steps are patched
and the voice library is a temporary directory.
"""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from lib import sml_wizard as wizard

BOOK_TXT = "\n".join(
    [
        "Пётр вошёл в комнату и осмотрелся.",
        "[Пётр] Привет, Мария! Как ваши дела сегодня? [/]",
        "Мария улыбнулась ему в ответ.",
        "[Мария] Здравствуйте, Пётр. Всё хорошо, спасибо. [/]",
    ]
)

VOICE = "voices/rus/adult/male/Petr.wav"


def make_session(root, **overrides):
    session = {
        "session_id": "test",
        "session_dir": root,
        "ebook_mode": "text",
        "ebook_textarea": BOOK_TXT,
        "ebook_src": None,
        "ebook_textarea_src": None,
        "language": "rus",
        "tts_engine": "silero",
        "fine_tuned": "internal",
        "sml_output_dir": None,
        "sml_book_id": None,
        "sml_characters": [],
        "sml_text": None,
        "sml_applied": False,
    }
    session.update(overrides)
    return session


def make_voice_library(root):
    """A tiny library in the layout ``voices/<lang>/<age>/<gender>/<name>``."""
    for relative in (
        "voices/rus/adult/male/Petr.wav",
        "voices/rus/adult/female/Masha.wav",
        "voices/rus/child/female/Kid.wav",
    ):
        target = Path(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"voice")


def fake_analyze_book(book, output_dir, language=None, model=None, progress=None):
    text = Path(book).read_text(encoding="utf-8")
    return {
        "book_id": "test-book",
        "output_dir": str(output_dir),
        "text_file": str(book),
        "booknlp_data": {"book_txt": text},
        "characters": [
            {
                "normalized_name": "Narrator",
                "inferred_gender": "unknown",
                "inferred_age_category": "adult",
            },
            {
                "normalized_name": "Пётр",
                "inferred_gender": "male",
                "inferred_age_category": "adult",
            },
            {
                "normalized_name": "Мария",
                "inferred_gender": "female",
                "inferred_age_category": "adult",
            },
        ],
    }


def fake_cast(booknlp_data, characters, **kwargs):
    for character in characters:
        character["voice"] = VOICE
    return characters


def fake_describe(characters, booknlp_data=None, **kwargs):
    for character in characters:
        character.setdefault("description", f"Описание {character['normalized_name']}")
        character.setdefault("description_source", "heuristic")
    return characters


class WizardTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = self.tmp.name
        make_voice_library(self.root)
        wizard._books.clear()
        wizard._digests.clear()
        patcher = mock.patch.object(wizard.pipeline, "APP_ROOT", Path(self.root))
        self.addCleanup(patcher.stop)
        patcher.start()

    def tearDown(self):
        wizard._books.clear()
        wizard._digests.clear()
        self.tmp.cleanup()

    def analyze(self, session=None, **kwargs):
        session = session or make_session(self.root)
        with mock.patch.object(wizard.pipeline, "analyze_book", fake_analyze_book), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            message = wizard.analyze(session, **kwargs)
        return session, message


class SessionHelperTests(WizardTestCase):
    def test_wizard_dir_is_created_inside_the_session_dir(self):
        session = make_session(self.root)
        path = wizard.wizard_dir(session)
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(Path(path).parent, Path(self.root))

    def test_session_readers(self):
        session = make_session(
            self.root,
            language="RUS",
            tts_engine="silero",
            sml_characters=[{"normalized_name": "Пётр", "voice": "/v/Petr.wav"}],
        )
        self.assertEqual(wizard.language(session), "rus")
        self.assertEqual(wizard.engine(session), "silero")
        self.assertEqual(wizard.voices(session), {"Пётр": "/v/Petr.wav"})
        self.assertEqual(wizard.characters(session)[0]["normalized_name"], "Пётр")

    def test_names_and_voices_come_from_the_characters(self):
        session = make_session(root=self.root, sml_characters=[])
        self.assertEqual(wizard.names(session), [])
        self.assertEqual(wizard.voices(session), {})


class BookFileTests(WizardTestCase):
    def test_the_textarea_is_dumped_for_text_mode(self):
        session = make_session(self.root)
        path = wizard.book_file(session)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(Path(path).read_text(encoding="utf-8"), BOOK_TXT)

    def test_a_single_file_wins_over_the_textarea(self):
        source = Path(self.root, "book.txt")
        source.write_text("книга", encoding="utf-8")
        session = make_session(
            self.root, ebook_mode="single", ebook_src=str(source), ebook_textarea=""
        )
        self.assertEqual(wizard.book_file(session), str(source))

    def test_nothing_to_read_at_all(self):
        session = make_session(self.root, ebook_mode="single", ebook_textarea="")
        self.assertIsNone(wizard.book_file(session))


class AnalyzeTests(WizardTestCase):
    def test_analysis_fills_the_session_and_writes_characters(self):
        session = make_session(self.root)
        with mock.patch.object(wizard.pipeline, "analyze_book", fake_analyze_book), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            message = wizard.analyze(session)
        self.assertIn("персонажей 3", message)
        self.assertEqual(session["sml_book_id"], "test-book")
        self.assertTrue(os.path.isdir(session["sml_output_dir"]))
        self.assertEqual(len(session["sml_characters"]), 3)
        self.assertFalse(session["sml_applied"])
        self.assertIsNone(session["sml_text"])
        stored = json.loads(
            Path(self.root, "sml", "characters.json").read_text(encoding="utf-8")
        )
        self.assertEqual(stored["book_id"], "test-book")
        self.assertEqual(len(stored["characters"]), 3)

    def test_the_second_run_of_an_unchanged_book_is_skipped(self):
        session = make_session(self.root)
        calls = []

        def counting(book, output_dir, **kwargs):
            calls.append(book)
            return fake_analyze_book(book, output_dir, **kwargs)

        with mock.patch.object(wizard.pipeline, "analyze_book", counting), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            first = wizard.analyze(session)
            second = wizard.analyze(session)
        self.assertEqual(len(calls), 1)
        self.assertIn("Готово", first)
        self.assertIn("уже сделан", second)

    def test_a_changed_book_is_analyzed_again(self):
        session = make_session(self.root)
        calls = []

        def counting(book, output_dir, **kwargs):
            calls.append(book)
            return fake_analyze_book(book, output_dir, **kwargs)

        with mock.patch.object(wizard.pipeline, "analyze_book", counting), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            wizard.analyze(session)
            session["ebook_textarea"] = BOOK_TXT + "\n[Пётр] Ещё реплика для проверки. [/]"
            wizard.analyze(session)
        self.assertEqual(len(calls), 2)

    def test_force_reanalyzes_the_same_book(self):
        session = make_session(self.root)
        calls = []

        def counting(book, output_dir, **kwargs):
            calls.append(book)
            return fake_analyze_book(book, output_dir, **kwargs)

        with mock.patch.object(wizard.pipeline, "analyze_book", counting), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            wizard.analyze(session)
            wizard.analyze(session, force=True)
        self.assertEqual(len(calls), 2)

    def test_an_unsupported_language_is_reported(self):
        session = make_session(self.root, language="deu")
        with self.assertRaises(wizard.WizardError) as caught:
            wizard.analyze(session)
        self.assertIn("BookNLP", str(caught.exception))

    def test_an_empty_session_is_reported(self):
        session = make_session(self.root, ebook_mode="single", ebook_textarea="")
        with self.assertRaises(wizard.WizardError):
            wizard.analyze(session)

    def test_a_step_failure_stores_nothing(self):
        session = make_session(self.root)

        def failing(characters, *args, **kwargs):
            raise RuntimeError("модель недоступна")

        with mock.patch.object(wizard.pipeline, "analyze_book", fake_analyze_book), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", failing):
            with self.assertRaises(RuntimeError):
                wizard.analyze(session)
        # nothing was stored: the picker still shows its empty state
        self.assertIsNone(session["sml_book_id"])
        self.assertEqual(wizard.characters(session), [])


class RowsTests(WizardTestCase):
    def _analyzed(self):
        session = make_session(self.root)
        with mock.patch.object(wizard.pipeline, "analyze_book", fake_analyze_book), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            wizard.analyze(session)
        return session

    def test_rows_are_the_writer_shape_with_the_narrator_first(self):
        session = self._analyzed()
        rows = wizard.rows(session)
        self.assertEqual([row[0] for row in rows][0], "Narrator")
        self.assertEqual(len(rows[0]), 6)
        self.assertEqual(wizard.names(session), [row[0] for row in rows])
        self.assertIn("Пётр", wizard.names(session))

    def test_rows_survive_a_forgotten_analysis(self):
        session = self._analyzed()
        wizard._books.clear()
        rows = wizard.rows(session)
        self.assertEqual([row[0] for row in rows][0], "Narrator")
        self.assertIn("Пётр", [row[0] for row in rows])

    def test_rows_without_an_analysis_are_empty(self):
        session = make_session(self.root)
        self.assertEqual(wizard.rows(session), [])


class VoiceTests(WizardTestCase):
    def test_the_library_is_offered_by_language(self):
        session = make_session(self.root)
        with mock.patch.object(wizard.pipeline, "APP_ROOT", Path(self.root)):
            choices = wizard.voice_choices(session)
        paths = sorted(path for _label, path in choices)
        self.assertEqual(len(choices), 3)
        self.assertTrue(any(path.endswith("Petr.wav") for path in paths))
        self.assertTrue(all(os.path.isabs(path) for path in paths))

    def test_a_picked_voice_is_remembered(self):
        session = make_session(
            self.root,
            sml_characters=[
                {"normalized_name": "Пётр", "voice": None},
                {"normalized_name": "Мария", "voice": None},
            ],
        )
        self.assertTrue(wizard.set_voice(session, "Пётр", "/v/Petr.wav"))
        self.assertEqual(wizard.voice_of(session, "Пётр"), "/v/Petr.wav")
        self.assertIsNone(wizard.voice_of(session, "Мария"))

    def test_an_unknown_character_is_not_changed(self):
        session = make_session(self.root, sml_characters=[])
        self.assertFalse(wizard.set_voice(session, "Кто-то", "/v/x.wav"))
        self.assertIsNone(wizard.voice_of(session, "Кто-то"))


class PreviewTests(WizardTestCase):
    def _analyzed(self):
        session = make_session(self.root)
        with mock.patch.object(wizard.pipeline, "analyze_book", fake_analyze_book), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            wizard.analyze(session)
        return session

    def test_a_sample_exists_for_a_speaking_character(self):
        session = self._analyzed()
        self.assertIn("Привет", wizard.sample(session, "Пётр"))

    def test_an_unknown_character_falls_back_to_narration(self):
        session = self._analyzed()
        # a character with no line of its own is previewed with narration
        self.assertIn("вошёл в комнату", wizard.sample(session, "Нет такого"))

    def test_an_empty_name_has_no_sample(self):
        session = self._analyzed()
        self.assertEqual(wizard.sample(session, ""), "")

    def test_the_preview_lists_the_characters_and_the_opening(self):
        session = self._analyzed()
        text = wizard.preview(session)
        self.assertIn("Narrator", text)
        self.assertIn("— Начало книги —", text)

    def test_the_preview_without_an_analysis_asks_for_one(self):
        self.assertIn("Анализ книги", wizard.preview(make_session(self.root)))

    def test_the_preview_shows_the_head_of_a_built_document(self):
        session = self._analyzed()
        session["sml_text"] = "[voice:/v/Petr.wav]Привет[/voice]"
        self.assertIn("— SML (начало) —", wizard.preview(session))


class BuildTests(WizardTestCase):
    def _analyzed(self):
        session = make_session(self.root)
        with mock.patch.object(wizard.pipeline, "analyze_book", fake_analyze_book), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            wizard.analyze(session)
        return session

    def test_build_stores_the_document_and_reports_the_blocks(self):
        session = self._analyzed()
        text, status = wizard.build(session, use_emotions=False, use_stress=False)
        self.assertEqual(session["sml_text"], text)
        self.assertIn("[voice:", text)
        self.assertIn("блоков", status)
        self.assertFalse(session["sml_applied"])

    def test_build_before_the_analysis_is_reported(self):
        with self.assertRaises(wizard.WizardError):
            wizard.build(make_session(self.root))

    def test_stress_marks_are_kept_for_silero_only(self):
        # a fake accentor that marks one word, so the test does not need the model
        accentor = lambda: (lambda text: text.replace("Привет", "Прив+ет"))
        session = self._analyzed()
        with mock.patch.object(wizard.pipeline.stress, "get_accentor", accentor):
            silero, _ = wizard.build(session, use_emotions=False, use_stress=True)
            session["tts_engine"] = "xtts"
            plain, _ = wizard.build(session, use_emotions=False, use_stress=True)
        self.assertIn("Прив+ет", silero)
        self.assertNotIn("+", plain)
        self.assertIn("Привет", plain)

    def test_apply_hands_the_document_over(self):
        session = self._analyzed()
        text, _ = wizard.build(session, use_emotions=False, use_stress=False)
        self.assertEqual(wizard.apply(session), text)
        self.assertTrue(session["sml_applied"])

    def test_apply_without_a_document_is_reported(self):
        with self.assertRaises(wizard.WizardError):
            wizard.apply(make_session(self.root))

    def test_cancel_clears_the_applied_flag(self):
        session = self._analyzed()
        wizard.build(session, use_emotions=False, use_stress=False)
        wizard.apply(session)
        wizard.cancel(session)
        self.assertFalse(session["sml_applied"])


class PersistenceTests(WizardTestCase):
    def test_restore_brings_an_analysis_from_an_earlier_session_back(self):
        session = make_session(self.root)
        with mock.patch.object(wizard.pipeline, "analyze_book", fake_analyze_book), \
             mock.patch.object(wizard.pipeline, "cast_characters", fake_cast), \
             mock.patch.object(wizard.pipeline, "describe_characters", fake_describe):
            wizard.analyze(session)
        fresh = make_session(self.root)
        self.assertTrue(wizard.restore(fresh))
        self.assertEqual(fresh["sml_book_id"], "test-book")
        self.assertEqual(len(wizard.characters(fresh)), 3)
        # the digest is remembered, so a second "analyze" is a no-op again
        self.assertTrue(wizard._digests.get(fresh["sml_output_dir"]))

    def test_restore_without_a_file_fails(self):
        self.assertFalse(wizard.restore(make_session(self.root)))

    def test_restore_keeps_an_already_loaded_book(self):
        session = make_session(self.root, sml_book_id="kept")
        self.assertTrue(wizard.restore(session))
        self.assertEqual(session["sml_book_id"], "kept")

    def test_a_broken_characters_file_is_ignored(self):
        session = make_session(self.root)
        target = Path(self.root, "sml")
        target.mkdir(parents=True, exist_ok=True)
        Path(target, "characters.json").write_text("{not json", encoding="utf-8")
        self.assertFalse(wizard.restore(session))


if __name__ == "__main__":
    unittest.main()
