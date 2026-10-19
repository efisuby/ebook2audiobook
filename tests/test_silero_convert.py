"""Regression tests for :class:`lib.classes.tts_engines.silero.Silero`.

The v5 checkpoints raise a bare ``ValueError`` for fragments they cannot
pronounce at all (digits, latin words, stray markup). Because
``convert_chapters2audio`` aborts the whole book on one ``return False``, a single
digit-only sentence used to kill the run -- and its missing output file would have
broken ``combine_audio_sentences`` as well.

The engine is built with ``__new__`` so these tests never load a real checkpoint;
only ``apply_tts`` and ``audio_save`` are faked.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lib.classes.tts_engines.silero import Silero


class FakeCheckpoint:
    """A checkpoint stand-in: fails on text with no Cyrillic letters."""

    speakers = ['aidar', 'baya', 'kseniya', 'eugene', 'xenia']

    def __init__(self):
        self.calls = []

    def apply_tts(self, text, speaker, sample_rate):
        self.calls.append((text, speaker))
        if not any('а' <= char.lower() <= 'я' or char.lower() == 'ё' for char in text):
            raise ValueError()
        return np.linspace(-0.5, 0.5, max(16, sample_rate // 20), dtype=np.float32)


def make_engine(tmp):
    engine = Silero.__new__(Silero)
    engine.session = {'voice': 'voices/rus/adult/male/Petr.wav', 'tts_engine': 'silero'}
    engine.tts_engine = 'silero'
    engine.tts_key = 'silero-internal'
    engine.cache_dir = tmp
    engine.device = 'cpu'
    engine.engine = FakeCheckpoint()
    engine.resampler_cache = {}
    engine.audio_segments = []
    engine.accentor = False
    engine.use_stress = False
    engine.params = {
        'samplerate': 48000,
        'voice_speakers': {},
        'speaker_cursor': {'male': 0, 'female': 0},
        'block_voice': 'voices/rus/adult/male/Petr.wav',
        'current_voice': 'voices/rus/adult/male/Petr.wav',
    }
    # the session-dependent and side-effecting helpers of the base classes
    engine._set_voice = lambda voice: (voice, None)
    engine.cleanup_memory = lambda: None
    engine.log_exception = lambda where, error: f'{where}: {error}'

    def audio_save(sentence_file, segment_tensor, samplerate):
        Path(sentence_file).write_bytes(b'RIFF' + b'\x00' * 64)
        return True

    engine.audio_save = audio_save
    return engine


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = self._tmp.name
        self.engine = make_engine(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _convert(self, sentence):
        target = os.path.join(self.root, 'out.wav')
        if os.path.exists(target):
            os.remove(target)
        result = self.engine.convert(target, sentence)
        return result, target

    def test_a_plain_sentence_is_saved(self):
        (success, error), target = self._convert('Привет, как дела?')
        self.assertEqual((success, error), (True, None))
        self.assertTrue(os.path.isfile(target))
        self.assertTrue(self.engine.engine.calls)

    def test_a_digit_only_sentence_does_not_fail_the_run(self):
        (success, error), target = self._convert('12345')
        self.assertEqual((success, error), (True, None))
        # a file is written anyway: a missing index breaks the chapter combiner
        self.assertTrue(os.path.isfile(target))

    def test_a_punctuation_only_sentence_still_writes_a_file(self):
        (success, error), target = self._convert('...')
        self.assertEqual((success, error), (True, None))
        self.assertTrue(os.path.isfile(target))

    def test_the_checkpoint_is_never_called_for_a_punctuation_only_sentence(self):
        self._convert('...')
        self.assertEqual(self.engine.engine.calls, [])

    def test_the_speaker_is_chosen_by_the_voice_gender(self):
        self._convert('Привет')
        self.assertIn(self.engine.engine.calls[0][1], Silero.speaker_pools['male'])


if __name__ == '__main__':
    unittest.main()
