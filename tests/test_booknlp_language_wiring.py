"""Regression tests for the project <-> BookNLP language-code mapping.

``run_booknlp`` uses the project's ``eng``/``rus`` codes, but the vendored
English BookNLP shim keys its pipeline construction on ``en``. Passing
``eng`` straight through left the shim half-initialised and English analysis
failed at runtime with ``'BookNLP' object has no attribute 'booknlp'``.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sml_extractor.core as core


class BookNlpLanguageWiringTests(unittest.TestCase):
    def _run(self, language):
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.object(core, "configure_booknlp_cache", return_value=Path(folder)),
                patch.object(core, "check_booknlp_installation", return_value=(True, "ok")),
                patch("booknlp.booknlp.BookNLP") as english_shim,
                patch("booknlp.russian.RussianBookNLP") as russian_pipeline,
            ):
                core.run_booknlp("book.txt", folder, "small", language)
        return english_shim, russian_pipeline

    def test_english_code_is_translated_for_the_vendored_shim(self):
        english_shim, russian_pipeline = self._run("eng")

        english_shim.assert_called_once()
        self.assertEqual(english_shim.call_args.args[0], "en")
        russian_pipeline.assert_not_called()

    def test_russian_uses_the_russian_pipeline(self):
        english_shim, russian_pipeline = self._run("rus")

        russian_pipeline.assert_called_once()
        self.assertEqual(russian_pipeline.call_args.args[0], "rus")
        english_shim.assert_not_called()


if __name__ == "__main__":
    unittest.main()
