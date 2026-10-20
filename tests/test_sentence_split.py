"""Tests for how a block is cut into the pieces the engine speaks.

``lib.core.get_sentences`` splits a paragraph into TTS sentences.  The cap is
``lib.SENTENCE_MAX_CHARS`` (1024): an ordinary sentence must stay in **one**
piece, because a piece that ends on an internal comma makes the engine treat it
as a sentence end and finish the intonation there — the «одно предложение — два
блока» report.  ``_force_split_segment`` is still there for a pathological
sentence, and it must prefer a real terminator (or a space) over a comma.
"""

import unittest
from types import SimpleNamespace

from lib import core, SENTENCE_MAX_CHARS


# One grammatical sentence, with the two commas that were once chosen as cut
# points when the cap was max_chars/2 (91 chars for «rus»).
LONG_SENTENCE = (
    "По воле благословенного Императора, решившего отойти от ратных дел, "
    "бразды правления этой беспримерной кампанией были вручены Хорусу, "
    "примарху Легиона Лунных Волков."
)


def make_session(**overrides):
    session = {
        "language": "rus",
        "translate_enabled": False,
        "translate": None,
        "tts_engine": "silero",
        "is_gui_process": False,
    }
    session.update(overrides)
    return session


class SentenceSplitTests(unittest.TestCase):
    def setUp(self):
        # ``get_sentences`` reaches the session through the module-global
        # ``context``, injected by the running app; a test installs its own.
        self._context = core.context
        self.session = make_session()
        core.context = SimpleNamespace(get_session=lambda session_id: self.session)

    def tearDown(self):
        core.context = self._context

    def sentences(self, text):
        return core.get_sentences("session", text)

    def test_the_cap_is_1024(self):
        self.assertEqual(SENTENCE_MAX_CHARS, 1024)
        self.assertEqual(core.SENTENCE_MAX_CHARS, 1024)

    def test_an_ordinary_long_sentence_stays_in_one_piece(self):
        # 165 chars: over the old 91-char cap, well under the new 1024 cap
        self.assertGreater(len(LONG_SENTENCE), 91)
        self.assertLess(len(LONG_SENTENCE), SENTENCE_MAX_CHARS)
        self.assertEqual(self.sentences(LONG_SENTENCE), [LONG_SENTENCE])

    def test_a_short_sentence_is_left_whole(self):
        text = "Мария кивнула и вышла."
        self.assertEqual(self.sentences(text), [text])

    def test_a_comma_inside_a_sentence_is_untouched(self):
        text = "Пётр читает, Мария кивает."
        self.assertEqual(self.sentences(text), [text])


class ForceSplitTests(unittest.TestCase):
    """``_force_split_segment`` is a nested helper, so it is reached through
    ``get_sentences`` with a sentence far over the cap."""

    def setUp(self):
        self._context = core.context
        self.session = make_session()
        core.context = SimpleNamespace(get_session=lambda session_id: self.session)

    def tearDown(self):
        core.context = self._context

    def sentences(self, text):
        return core.get_sentences("session", text)

    def test_a_pathological_sentence_is_split(self):
        text = "слово " * 400  # 2400 chars, far over 1024
        result = self.sentences(text)
        self.assertGreater(len(result), 1)
        for piece in result:
            self.assertTrue(piece.strip())
            self.assertLessEqual(len(piece), SENTENCE_MAX_CHARS)

    def test_no_piece_ends_on_a_comma(self):
        # the input must not itself end on a comma, or the last piece would
        # inherit it and the assertion would be meaningless
        clause = ("Он вошёл в зал, и она ждала его там, а он молчал, " * 39
                  + "Он вошёл в зал, и она ждала его там, а он молчал")
        result = self.sentences(clause)
        self.assertGreater(len(result), 1)
        for piece in result:
            self.assertFalse(piece.endswith(","), piece[-30:])


if __name__ == "__main__":
    unittest.main()
