"""End-to-end tests for the Russian analysis pipeline.

A synthetic Russian book goes through ``RussianBookNLP.process`` and then
through the same downstream steps the GUI/CLI use (load -> extract characters
-> speaking characters -> SML), asserting Cyrillic names, dialogue
attribution and gender inference survive every stage.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from booknlp.russian import RussianBookNLP
from booknlp.russian import russian_coref, russian_nlp_utils
from booknlp.russian.russian_booknlp import _camel
from booknlp.russian.russian_coref import RussianCorefResolver, configured_coref_model
from booknlp.russian.russian_nlp_utils import load_russian_spacy_model
from sml_extractor.core import (
    check_booknlp_installation,
    extract_characters,
    load_booknlp_output,
)
from sml_extractor.sml_generator import (
    _normalize_name,
    generate_sml_output,
    speaking_characters,
)

RU_TEXT = """Анна тихо вошла в комнату.

— Ты здесь? — спросил Иван, не поднимая глаз.

— Я здесь, — ответила Анна. — Я ждала тебя.

«Никогда он не слушает», — подумала Анна.

Иван Петров наконец поднял голову. — Пойдём, сказал он. — Нам нужно поговорить.

— Хорошо, — сказала Анна.

— Я не помню этого, — призналась она.

Они вышли на улицу. Петров стоял у ворот.
"""


class RussianPipelineTests(unittest.TestCase):
    """One shared pipeline run for the whole class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="ru_pipeline_")
        root = Path(cls._tmp.name)
        cls.out_dir = root / "out"
        book = root / "demo.txt"
        book.write_text(RU_TEXT, encoding="utf-8")

        pipeline = RussianBookNLP("rus", {"model": "small", "model_path": str(root)})
        pipeline.process(str(book), str(cls.out_dir), "demo")
        cls.data = load_booknlp_output(str(cls.out_dir), "demo")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- environment ------------------------------------------------------

    def test_booknlp_installation_check_passes_for_rus(self):
        ok, message = check_booknlp_installation("rus")
        self.assertTrue(ok, message)

    # -- output files -----------------------------------------------------

    def test_booknlp_compatible_files_written(self):
        for suffix in (
            ".tokens",
            ".quotes",
            ".entities",
            ".book",
            ".characters_simple.json",
        ):
            path = self.out_dir / f"demo{suffix}"
            self.assertTrue(path.is_file(), f"missing {path.name}")

    def test_characters_simple_starts_with_narrator(self):
        import json

        payload = json.loads(
            (self.out_dir / "demo.characters_simple.json").read_text(encoding="utf-8")
        )
        characters = payload["characters"]
        self.assertEqual(characters[0]["normalized_name"], "Narrator")
        self.assertTrue(all(c["language"] == "rus" for c in characters))

    # -- extraction quality ------------------------------------------------

    def _canonical_name(self, char_id):
        for character in self.data["book_data"]["characters"]:
            if str(character["id"]) == str(char_id):
                proper = character["mentions"]["proper"]
                return proper[0]["n"] if proper else None
        return None

    def _quote_text(self, quote):
        start = int(quote["quote_start"])
        end = int(quote["quote_end"])
        words = [token["word"] for token in self.data["tokens"]]
        return " ".join(words[start : end + 1])

    def test_characters_have_cyrillic_names_and_gender(self):
        characters = self.data["book_data"]["characters"]
        by_name = {
            character["mentions"]["proper"][0]["n"]: character
            for character in characters
        }
        self.assertIn("Анна", by_name)
        self.assertIn("Иван", by_name)
        self.assertEqual(by_name["Анна"]["g"], "female")
        self.assertEqual(by_name["Иван"]["g"], "male")
        # Иван Петров and Петров merged into the same cluster as Иван.
        surfaces = {
            mention["n"]
            for character in characters
            for mention in character["mentions"]["proper"]
        }
        self.assertIn("Иван Петров", surfaces)
        self.assertIn("Петров", surfaces)

    def test_dialogue_attributed_to_the_right_speakers(self):
        quotes = self.data["quotes"]
        self.assertGreaterEqual(len(quotes), 6)

        attributed = 0
        found = {}
        for quote in quotes:
            char_id = str(quote["char_id"])
            name = self._canonical_name(char_id) if char_id not in ("None", "") else None
            if name is not None:
                attributed += 1
            found[self._quote_text(quote)] = name

        # Narrator keeps the non-dialogue spans, speakers the rest.
        self.assertGreaterEqual(attributed, 5)

        def speaker_of(fragment):
            for text, name in found.items():
                if fragment in text:
                    return name
            self.fail(f"quote containing {fragment!r} not found: {list(found)}")

        self.assertIn("Иван", speaker_of("Ты здесь"))
        self.assertIn("Анна", speaker_of("Я здесь"))
        self.assertIn("Анна", speaker_of("Никогда"))
        # “— Пойдём, сказал он.” — pronoun coreference resolves “он” to Иван.
        self.assertIn("Иван", speaker_of("Пойдём"))
        # “— …, призналась она.” — “она” resolves to the female cluster (Анна).
        self.assertIn("Анна", speaker_of("Я не помню"))

    def test_speaking_characters_and_sml_end_to_end(self):
        characters = extract_characters(self.data, "rus")
        self.assertEqual(characters[0]["normalized_name"], "Narrator")
        names = {character["normalized_name"] for character in characters}
        self.assertIn("Анна", names)
        self.assertIn("Иван", names)

        speakers = speaking_characters(self.data, characters)
        speaker_names = {character["normalized_name"] for character in speakers}
        self.assertLessEqual({"Narrator", "Анна", "Иван"}, speaker_names)

        sml_path = Path(self._tmp.name) / "demo.e2a.sml.txt"
        generate_sml_output(self.data, speakers, str(sml_path), voice_assignments=None)
        content = sml_path.read_text(encoding="utf-8")
        self.assertTrue(sml_path.is_file())
        self.assertIn("Ты здесь", content)
        self.assertIn("Никогда", content)
        # Cyrillic character names survive into voice-tag macro names.
        self.assertIn("Анна", content)


        self.assertIn("Анна", content)


class JunkNameFilterTests(unittest.TestCase):
    """Sentence-initial function words must never become characters.

    ``ru_core_news_sm`` capitalizes every sentence-initial word and sometimes
    tags such words as PROPN/PER, which used to turn «кто», «то», «ну» and
    friends into bogus characters.
    """

    JUNK = (
        "кто", "то", "ну", "на", "после", "от", "мой", "иной", "никто",
        "возможно", "итак", "спасибо", "верно", "правда", "сколько",
        "зато", "тебе", "слишком", "кого", "шестьдесят", "десятая",
        "первого", "вперёд", "по", "не",
    )
    REAL = ("локен", "хорус", "каркази", "зиндерманн", "иван", "анна", "петров")

    def test_function_words_are_not_plausible_names(self):
        for word in self.JUNK:
            self.assertFalse(RussianBookNLP._plausible_name((word,)), word)

    def test_real_names_are_plausible(self):
        for word in self.REAL:
            self.assertTrue(RussianBookNLP._plausible_name((word,)), word)

    def test_sentence_start_fallback_uses_previous_punctuation(self):
        text = "Он вошёл. Локен ждал."
        self.assertTrue(
            RussianBookNLP._starts_sentence(text, None, text.index("Локен"))
        )
        self.assertFalse(
            RussianBookNLP._starts_sentence(text, None, text.index("ждал"))
        )


class CyrillicNormalizationTests(unittest.TestCase):
    def test_normalize_name_keeps_cyrillic(self):
        self.assertEqual(_normalize_name("Иван Петров"), "ИванПетров")
        self.assertEqual(_normalize_name("«Анна»"), "Анна")
        # Hyphens survive as word characters; capitalize() lowercases the tail.
        self.assertEqual(_normalize_name("Ольга-Сергеевна"), "Ольга-сергеевна")
        self.assertEqual(_normalize_name("Пётр!"), "Пётр")

    def test_camel_matches_sml_normalize_name(self):
        # russian_booknlp writes .characters_simple.json with _camel while
        # sml_generator matches .book names with _normalize_name — the two
        # must always agree or quote attribution silently breaks.
        for name in (
            "Иван Петров",
            "Анна-Мария",
            "Ольга-Сергеевна",
            "«Петров»",
            "Д'Артаньян",
            "О'Коннор-младший",
        ):
            self.assertEqual(_camel(name), _normalize_name(name), name)


class CorefConfigurationTests(unittest.TestCase):
    def setUp(self):
        self._previous = os.environ.get("E2A_SML_COREF_MODEL")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._previous is None:
            os.environ.pop("E2A_SML_COREF_MODEL", None)
        else:
            os.environ["E2A_SML_COREF_MODEL"] = self._previous

    def test_default_model_is_configured(self):
        os.environ.pop("E2A_SML_COREF_MODEL", None)
        self.assertIn("paraphrase-multilingual", configured_coref_model())

    def test_disabled_via_env_uses_lexical_backend(self):
        os.environ["E2A_SML_COREF_MODEL"] = "none"
        self.assertIsNone(configured_coref_model())
        resolver = RussianCorefResolver()
        self.assertEqual(resolver.backend, "lexical (disabled)")


class ModelCachingTests(unittest.TestCase):
    """A book run must not reload the encoders it already has in memory."""

    def setUp(self):
        self._saved = dict(russian_coref._MODEL_CACHE)
        russian_coref._MODEL_CACHE.clear()
        self.addCleanup(self._restore_models)

    def _restore_models(self):
        russian_coref._MODEL_CACHE.clear()
        russian_coref._MODEL_CACHE.update(self._saved)

    def test_encoder_is_loaded_once_across_resolvers(self):
        loaded: list[str] = []

        class FakeEncoder:
            def __init__(self, name: str):
                loaded.append(name)

        with mock.patch("sentence_transformers.SentenceTransformer", FakeEncoder):
            first = RussianCorefResolver("fake-model")
            second = RussianCorefResolver("fake-model")
            first._ensure_model()
            second._ensure_model()

        self.assertEqual(loaded, ["fake-model"])
        self.assertIs(first._model, second._model)
        self.assertIn("bert (fake-model)", first.backend)

    def test_gpu_oom_retries_the_encoder_on_the_cpu(self):
        calls: list[tuple] = []

        class FakeEncoder:
            def __init__(self, name, device=None):
                calls.append((name, device))
                if device is None:
                    raise RuntimeError("CUDA out of memory")

        with mock.patch("sentence_transformers.SentenceTransformer", FakeEncoder):
            resolver = RussianCorefResolver("mini-model")
            resolver._ensure_model()

        self.assertEqual(calls, [("mini-model", None), ("mini-model", "cpu")])
        self.assertIn("bert (mini-model)", resolver.backend)

    def test_unavailable_encoder_is_only_attempted_once(self):
        attempts: list[tuple] = []

        def failing(name: str, device=None):
            attempts.append((name, device))
            raise RuntimeError("no network")

        with mock.patch("sentence_transformers.SentenceTransformer", failing):
            first = RussianCorefResolver("missing-model")
            second = RussianCorefResolver("missing-model")
            first._ensure_model()
            second._ensure_model()

        # One GPU attempt, one CPU retry — and nothing on the second resolver.
        self.assertEqual(
            attempts, [("missing-model", None), ("missing-model", "cpu")]
        )
        self.assertEqual(first.backend, "lexical")
        self.assertEqual(second.backend, "lexical")

    def test_spacy_pipeline_is_cached(self):
        russian_nlp_utils._SPACY_MODEL_CACHE.clear()
        self.addCleanup(russian_nlp_utils._SPACY_MODEL_CACHE.clear)
        first = load_russian_spacy_model()
        if first is None:
            self.skipTest("ru_core_news_sm is not installed")
        self.assertIs(load_russian_spacy_model(), first)

    def test_spacy_load_is_attempted_once(self):
        russian_nlp_utils._SPACY_MODEL_CACHE.clear()
        self.addCleanup(russian_nlp_utils._SPACY_MODEL_CACHE.clear)
        calls: list[str] = []

        class FakeSpacy:
            @staticmethod
            def load(name: str):
                calls.append(name)
                return object()

        with mock.patch.dict("sys.modules", {"spacy": FakeSpacy}):
            first = load_russian_spacy_model()
            second = load_russian_spacy_model()

        self.assertEqual(calls, [russian_nlp_utils.RU_SPACY_MODEL])
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
