"""Tests for the silero-stress pre-processing helpers."""

import unittest

from sml_extractor import stress


def fake_accentor(text: str) -> str:
    """A tiny stand-in for silero-stress: mark one known word."""
    return text.replace("замок", "зам+ок").replace("Замок", "Зам+ок")


class AccentTests(unittest.TestCase):
    def test_accent_marks_the_text(self):
        self.assertEqual(stress.accent_text("замок", accentor=fake_accentor), "зам+ок")

    def test_accent_keeps_the_tags_untouched(self):
        text = "[voice:/v/rus/adult/male/Пётр.wav]Замок был на двери.[/voice]"
        self.assertEqual(
            stress.accent_sml(text, accentor=fake_accentor),
            "[voice:/v/rus/adult/male/Пётр.wav]Зам+ок был на двери.[/voice]",
        )

    def test_voice_paths_are_never_sent_to_the_accentor(self):
        seen = []

        def accentor(text):
            seen.append(text)
            return text

        stress.accent_sml("[voice:/v/rus/adult/male/Пётр.wav]текст[/voice][pause:1][break]", accentor=accentor)
        self.assertEqual(seen, ["текст"])

    def test_punctuation_only_chunks_are_skipped(self):
        seen = []
        stress.accent_sml("[break] . .. [pause]", accentor=lambda text: seen.append(text) or text)
        self.assertEqual(seen, [])

    def test_without_the_package_the_text_is_unchanged(self):
        original = stress.get_accentor
        try:
            stress._accentor = False
            self.assertIsNone(stress.get_accentor())
            self.assertEqual(stress.accent_sml("замок"), "замок")
        finally:
            stress.get_accentor = original
            stress.reset_accentor()

    def test_a_failing_accentor_keeps_the_fragment(self):
        def accentor(text):
            raise RuntimeError("boom")

        self.assertEqual(stress.accent_sml("замок", accentor=accentor), "замок")

    def test_empty_input(self):
        self.assertEqual(stress.accent_sml("", accentor=fake_accentor), "")


class StripTests(unittest.TestCase):
    def test_marks_before_vowels_are_removed(self):
        self.assertEqual(stress.strip_text("зам+ок и м+оет"), "замок и моет")

    def test_other_plus_signs_survive(self):
        self.assertEqual(stress.strip_text("2+2 и A+B и C+1"), "2+2 и A+B и C+1")

    def test_strip_keeps_the_tags(self):
        self.assertEqual(
            stress.strip_sml("[pause:1]зам+ок[/pause][voice:/v/Пётр.wav]м+оет[/voice]"),
            "[pause:1]замок[/pause][voice:/v/Пётр.wav]моет[/voice]",
        )

    def test_empty_input(self):
        self.assertEqual(stress.strip_sml(""), "")


class EngineRoutingTests(unittest.TestCase):
    def test_silero_is_accented(self):
        for engine in ("silero", "Silero", " SILERO "):
            self.assertTrue(stress.is_silero(engine))
            self.assertEqual(stress.stress_for_engine("замок", engine, accentor=fake_accentor), "зам+ок")

    def test_other_engines_are_stripped(self):
        for engine in ("xtts", "Qwen3-TTS", None, ""):
            self.assertFalse(stress.is_silero(engine))
            self.assertEqual(stress.stress_for_engine("зам+ок", engine), "замок")


if __name__ == "__main__":
    unittest.main()
