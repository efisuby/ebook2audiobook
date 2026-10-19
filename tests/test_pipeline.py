"""Tests for :mod:`sml_extractor.pipeline` — the glue the wizard drives."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from sml_extractor import pipeline, sml_generator

BOOK_TXT = "\n".join(
    [
        "Пётр вошёл в комнату. Он был суровым человеком.",
        "[Пётр] Привет, Мария! Как ваши дела сегодня? [/]",
        "Мария улыбнулась ему в ответ.",
        "[Мария] Здравствуйте, Пётр. Всё хорошо, спасибо. [/]",
    ]
)


def make_book():
    return {"book_txt": BOOK_TXT}


def make_characters():
    return [
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
    ]


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


class SegmentTests(unittest.TestCase):
    def test_narration_and_quotes_keep_their_order(self):
        segments = sml_generator.book_segments(make_book())
        self.assertEqual(
            segments,
            [
                (None, "Пётр вошёл в комнату. Он был суровым человеком."),
                ("Пётр", "Привет, Мария! Как ваши дела сегодня?"),
                (None, "Мария улыбнулась ему в ответ."),
                ("Мария", "Здравствуйте, Пётр. Всё хорошо, спасибо."),
            ],
        )

    def test_a_book_without_data_is_rejected(self):
        with self.assertRaises(ValueError):
            sml_generator.book_segments({})


class CastTests(unittest.TestCase):
    def test_every_speaker_gets_an_existing_absolute_voice(self):
        with TemporaryDirectory() as tmp:
            make_voice_library(tmp)
            characters = make_characters()
            cast = pipeline.cast_characters(
                make_book(), characters, language="rus", app_root=tmp
            )

            names = [item["normalized_name"] for item in cast]
            self.assertEqual(names, ["Narrator", "Пётр", "Мария"])
            for item in cast:
                self.assertTrue(Path(item["voice"]).is_absolute(), item["voice"])
                self.assertTrue(Path(item["voice"]).is_file(), item["voice"])
            by_name = {item["normalized_name"]: item["voice"] for item in cast}
            self.assertEqual(Path(by_name["Пётр"]).parent.name, "male")
            self.assertEqual(Path(by_name["Мария"]).parent.name, "female")

    def test_a_voice_typed_on_the_character_is_kept(self):
        with TemporaryDirectory() as tmp:
            make_voice_library(tmp)
            characters = make_characters()
            characters[1]["voice"] = "/custom/petr.wav"
            cast = pipeline.cast_characters(
                make_book(), characters, language="rus", app_root=tmp
            )
            by_name = {item["normalized_name"]: item["voice"] for item in cast}
            self.assertEqual(by_name["Пётр"], "/custom/petr.wav")

    def test_the_override_map_wins_over_the_character(self):
        with TemporaryDirectory() as tmp:
            make_voice_library(tmp)
            characters = make_characters()
            characters[1]["voice"] = "/custom/petr.wav"
            cast = pipeline.cast_characters(
                make_book(),
                characters,
                language="rus",
                app_root=tmp,
                voice_overrides={"Пётр": "/picked/petr.wav"},
            )
            by_name = {item["normalized_name"]: item["voice"] for item in cast}
            self.assertEqual(by_name["Пётр"], "/picked/petr.wav")

    def test_a_speaker_missing_from_the_cast_is_added(self):
        with TemporaryDirectory() as tmp:
            make_voice_library(tmp)
            characters = []
            cast = pipeline.cast_characters(
                make_book(), characters, language="rus", app_root=tmp
            )
            names = [item["normalized_name"] for item in cast]
            self.assertEqual(names, ["Narrator", "Пётр", "Мария"])
            # the records are appended so a later describe pass covers them
            self.assertEqual(
                [item["normalized_name"] for item in characters],
                ["Narrator", "Пётр", "Мария"],
            )
            self.assertTrue(all(item["voice"] for item in characters))


class DescribeTests(unittest.TestCase):
    def test_descriptions_are_written_without_the_model(self):
        characters = make_characters()
        pipeline.describe_characters(
            characters, make_book(), language="rus", use_llm=False
        )
        for character in characters:
            self.assertTrue(character["description"])
            self.assertEqual(character["description_source"], "heuristic")


class RowTests(unittest.TestCase):
    def test_rows_carry_what_the_picker_shows(self):
        characters = make_characters()
        pipeline.describe_characters(
            characters, make_book(), language="rus", use_llm=False
        )
        characters[2]["voice"] = "/voices/rus/adult/female/Masha.wav"

        rows = pipeline.character_rows(make_book(), characters)
        self.assertEqual([row["name"] for row in rows], ["Narrator", "Пётр", "Мария"])
        self.assertEqual(rows[0]["lines"], 0)
        self.assertEqual(rows[0]["voice_name"], "")
        maria = rows[2]
        self.assertEqual(maria["lines"], 1)
        self.assertEqual(maria["voice_name"], "Masha")
        self.assertEqual(maria["gender"], "female")
        self.assertTrue(maria["description"])


class SampleTests(unittest.TestCase):
    def test_a_character_gets_its_own_longest_early_line(self):
        self.assertEqual(
            pipeline.sample_line(make_book(), "Мария"),
            "Здравствуйте, Пётр. Всё хорошо, спасибо.",
        )

    def test_the_narrator_gets_narration(self):
        self.assertEqual(
            pipeline.sample_line(make_book(), "Narrator"),
            "Пётр вошёл в комнату. Он был суровым человеком.",
        )

    def test_a_character_without_a_line_falls_back_to_narration(self):
        book = {"book_txt": "Пётр вошёл в комнату и осмотрелся. Он был суровым человеком."}
        self.assertEqual(
            pipeline.sample_line(book, "Сергей"),
            "Пётр вошёл в комнату и осмотрелся. Он был суровым человеком.",
        )

    def test_no_data_gives_an_empty_line(self):
        self.assertEqual(pipeline.sample_line({}, "Пётр"), "")


class PreviewTests(unittest.TestCase):
    def test_preview_lists_every_character_and_the_opening(self):
        characters = make_characters()
        pipeline.describe_characters(
            characters, make_book(), language="rus", use_llm=False
        )
        lines = pipeline.preview_lines(make_book(), characters)

        joined = "\n".join(lines)
        self.assertIn("Narrator — без голоса —", joined)
        self.assertIn("— Начало книги —", joined)
        self.assertIn("«Здравствуйте, Пётр. Всё хорошо, спасибо.»", joined)
        self.assertIn("Пётр: Привет, Мария! Как ваши дела сегодня?", joined)
        self.assertIn("Narrator: Мария улыбнулась ему в ответ.", joined)

    def test_short_quotes_are_trimmed_to_the_width(self):
        characters = make_characters()
        lines = pipeline.preview_lines(
            make_book(), characters, per_character=1, width=20
        )
        quotes = [line for line in lines if line.startswith("    «")]
        self.assertTrue(quotes)
        for line in quotes:
            # "    «" + the trimmed text + "»"
            self.assertLessEqual(len(line), 4 + 1 + 20 + 1)


class StressPreviewTests(unittest.TestCase):
    def _accentor(self):
        """A zero-argument ``get_accentor`` replacement that marks one word."""
        accentor = lambda text: text.replace("Привет", "Прив+ет")
        return lambda: accentor

    def test_silero_marks_the_text_but_not_the_tags(self):
        characters = make_characters()
        characters[1]["voice"] = "/voices/rus/adult/male/Petr.wav"
        with mock.patch.object(
            pipeline.stress,
            "get_accentor",
            lambda: (lambda text: text.replace("Привет", "Прив+ет")),
        ):
            preview = pipeline.stress_preview(
                make_book(), characters, tts_engine="silero"
            )
        self.assertIn("Прив+ет", preview)
        self.assertIn("[voice:/voices/rus/adult/male/Petr.wav]", preview)

    def test_other_engines_get_the_text_without_marks(self):
        characters = make_characters()
        characters[1]["voice"] = "/voices/rus/adult/male/Petr.wav"
        with mock.patch.object(pipeline.stress, "get_accentor", self._accentor()):
            preview = pipeline.stress_preview(
                make_book(), characters, tts_engine="xtts"
            )
        self.assertNotIn("+", preview)
        self.assertIn("[voice:/voices/rus/adult/male/Petr.wav]", preview)


class BuildSmlTests(unittest.TestCase):
    def _accentor(self):
        """A zero-argument ``get_accentor`` replacement that marks one word."""
        accentor = lambda text: text.replace("Привет", "Прив+ет")
        return lambda: accentor

    def _voices(self):
        return {
            "Пётр": "/voices/rus/adult/male/Petr.wav",
            "Мария": "/voices/rus/adult/female/Masha.wav",
        }

    def test_voice_tags_and_emotions_are_rendered(self):
        segments = sml_generator.collect_speech_segments(make_book())
        emotion_map = {segments[0][1]: "joy"}
        characters = make_characters()
        with mock.patch.object(
            pipeline.emotion, "analyze_book_emotions", lambda *a, **k: emotion_map
        ):
            sml = pipeline.build_sml(
                make_book(),
                characters,
                voices={"Пётр": "/voices/rus/adult/male/Petr.wav"},
                tts_engine="xtts",
            )
        lines = sml.splitlines()
        # Each sentence is now its own block
        self.assertIn(
            "[voice:/voices/rus/adult/male/Petr.wav][emotion:joy]Привет, Мария![/emotion][/voice]",
            lines,
        )
        self.assertIn(
            "[voice:/voices/rus/adult/male/Petr.wav][emotion:joy]Как ваши дела сегодня?[/emotion][/voice]",
            lines,
        )
        # the narrator line stays bare, so E2A reads it with the block voice
        self.assertIn("Пётр вошёл в комнату. Он был суровым человеком.", sml)

    def test_emotions_can_be_switched_off(self):
        characters = make_characters()
        with mock.patch.object(pipeline.emotion, "analyze_book_emotions") as fake:
            sml = pipeline.build_sml(
                make_book(), characters, use_emotions=False, use_stress=False
            )
        self.assertFalse(fake.called)
        self.assertNotIn("[emotion:", sml)

    def test_stress_marks_are_kept_for_silero_and_stripped_for_the_rest(self):
        characters = make_characters()
        with mock.patch.object(pipeline.stress, "get_accentor", self._accentor()):
            silero = pipeline.build_sml(
                make_book(),
                characters,
                tts_engine="silero",
                use_emotions=False,
            )
            xtts = pipeline.build_sml(
                make_book(),
                characters,
                tts_engine="xtts",
                use_emotions=False,
            )
        self.assertIn("Прив+ет", silero)
        self.assertNotIn("+", xtts)
        self.assertIn("Привет", xtts)

    def test_the_stress_step_can_be_skipped(self):
        characters = make_characters()
        with mock.patch.object(pipeline.stress, "get_accentor", self._accentor()):
            self.assertNotIn(
                "+",
                pipeline.build_sml(
                    make_book(),
                    characters,
                    tts_engine="silero",
                    use_emotions=False,
                    use_stress=False,
                ),
            )


class BlockTests(unittest.TestCase):
    def test_the_document_becomes_one_block_per_line(self):
        self.assertEqual(
            pipeline.sml_blocks("[voice:а]раз[/voice]\n\nтекст\n   \n[break]\n"),
            ["[voice:а]раз[/voice]", "текст", "[break]"],
        )

    def test_empty_document_gives_no_blocks(self):
        self.assertEqual(pipeline.sml_blocks(""), [])
        self.assertEqual(pipeline.sml_blocks(None), [])

    def test_silero_is_recognised(self):
        self.assertTrue(pipeline.is_silero("Silero"))
        self.assertFalse(pipeline.is_silero("qwen3tts"))


class AnalyzeTests(unittest.TestCase):
    def _patches(self, calls, ebook=False):
        def fake_convert(book_file, output_dir):
            calls["convert"] = (book_file, output_dir)
            return str(Path(output_dir) / "book.txt")

        def fake_run(**kwargs):
            calls["run"] = kwargs
            return {"book_id": "b-1"}

        def fake_load(output_dir, book_id):
            calls["load"] = (output_dir, book_id)
            return {"book_txt": "текст", "characters": []}

        def fake_extract(booknlp_data, language="eng"):
            calls["extract"] = language
            return [{"normalized_name": "Пётр"}]

        patches = [
            mock.patch.object(pipeline, "run_booknlp", fake_run),
            mock.patch.object(pipeline, "load_booknlp_output", fake_load),
            mock.patch.object(pipeline, "extract_characters", fake_extract),
            mock.patch.object(pipeline, "convert_ebook_to_txt", fake_convert),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_a_text_file_goes_straight_to_booknlp(self):
        calls = {}
        self._patches(calls)
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "book.txt"
            source.write_text(BOOK_TXT, encoding="utf-8")
            result = pipeline.analyze_book(source, Path(tmp) / "out", language="rus")
            self.assertTrue(Path(result["output_dir"]).is_dir())

        self.assertNotIn("convert", calls)
        self.assertEqual(calls["run"]["input_file"], str(source.resolve()))
        self.assertEqual(calls["run"]["model"], "small")
        self.assertEqual(calls["run"]["language"], "rus")
        self.assertEqual(calls["load"][1], "b-1")
        self.assertEqual(calls["extract"], "rus")
        self.assertEqual(result["book_id"], "b-1")
        self.assertEqual(result["characters"], [{"normalized_name": "Пётр"}])

    def test_an_ebook_goes_through_calibre_first(self):
        calls = {}
        self._patches(calls)
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "book.epub"
            source.write_bytes(b"ebook")
            pipeline.analyze_book(source, Path(tmp) / "out", language="rus")

        self.assertIn("convert", calls)
        self.assertEqual(calls["run"]["input_file"], calls["convert"][1] + "/book.txt")

    def test_progress_is_forwarded_in_fractions(self):
        calls = {}
        self._patches(calls)
        seen = []
        with TemporaryDirectory() as tmp:
            pipeline.analyze_book(
                Path(tmp) / "book.txt",
                Path(tmp) / "out",
                progress=lambda fraction, desc=None: seen.append((fraction, desc)),
            )
        calls["run"]["progress_callback"]("Parsing", 40)
        self.assertEqual(seen, [(0.4, "Parsing")])


class LoadBookTests(unittest.TestCase):
    def test_the_analysis_is_reloaded_from_disk(self):
        with mock.patch.object(
            pipeline, "load_booknlp_output", lambda output_dir, book_id: {"book_id": book_id}
        ), mock.patch.object(
            pipeline, "extract_characters", lambda data, language="eng": [{"n": language}]
        ):
            result = pipeline.load_book("/tmp/out", "b-9", language="rus")
        self.assertEqual(result["book_id"], "b-9")
        self.assertEqual(result["characters"], [{"n": "rus"}])


if __name__ == "__main__":
    unittest.main()
