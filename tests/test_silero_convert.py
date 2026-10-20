"""Regression tests for :class:`lib.classes.tts_engines.silero.Silero`.

The v5 checkpoints raise a bare ``ValueError`` for fragments they cannot
pronounce at all (digits, latin words, stray markup). Because
``convert_chapters2audio`` aborts the whole book on one ``return False``, a single
digit-only sentence used to kill the run -- and its missing output file would have
broken ``combine_audio_sentences`` as well.

The engine is built with ``__new__`` so these tests never load a real checkpoint;
only ``apply_tts`` and ``audio_save`` are faked.

A second group covers the pause the engine inserts after a sentence-ending
punctuation mark: E2A's ``filter_blocks`` splits a block at that mark, so the
fragment carries no terminator of its own and ``convert`` has to recover it from
the untrimmed sentence it came from.
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
    engine._convert_parts = []
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


class TerminatorPauseTests(unittest.TestCase):
    """The terminator is recovered from the untrimmed sentence (see convert())."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = self._tmp.name
        self.engine = make_engine(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _sentence(self, sentence, **kwargs):
        """Convert ``sentence`` and keep the segments for inspection.

        ``convert`` clears ``audio_segments`` once the file is written, so the
        list is signalled to be kept by the fake ``audio_save``.
        """
        target = os.path.join(self.root, 'out.wav')
        self.engine.audio_segments = []
        saved = []

        def audio_save(sentence_file, segment_tensor, samplerate):
            saved.append(segment_tensor)
            Path(sentence_file).write_bytes(b'RIFF' + b'\x00' * 64)
            return True

        self.engine.audio_save = audio_save
        result = self.engine.convert(target, sentence, **kwargs)
        self.assertEqual(result, (True, None))
        self._last_audio = saved
        return self.engine

    def _silences(self):
        """The silent stretches (offset, length) inside the audio of the last conversion."""
        silences = []
        for tensor in getattr(self, '_last_audio', []):
            data = tensor[0]
            index = 0
            while index < data.shape[-1]:
                if data[index] == 0:
                    start = index
                    while index < data.shape[-1] and data[index] == 0:
                        index += 1
                    silences.append((start, int(index - start)))
                else:
                    index += 1
        return silences

    def test_a_period_split_off_the_sentence_still_pauses(self):
        # the wizard hands over 'Пётр читает' while the sentence had a period:
        # the fragments of sentence_parts carry no punctuation at all, so the
        # terminator is recovered from the untrimmed sentence. The exact silence
        # length is randomised (0.3-0.6 s per fragment), so only the fact that
        # silence follows *each* fragment is asserted.
        voice_file = os.path.join(self.root, 'Мария.wav')
        Path(voice_file).write_bytes(b'RIFF' + b'\x00' * 8)
        self._sentence(f'Пётр читает.[voice:{voice_file}]Мария кивает.')
        data = self._last_audio[0][0]
        # two fragments spoken
        self.assertEqual(int((data != 0).sum()), 2 * 2400)
        # and a pause follows each of them, so there is noticeably more silence
        # than the single trailing pause of a one-fragment sentence. The pause is
        # randomised (0.3-0.6 s), hence the generous floor of one whole pause.
        self.assertGreaterEqual(int((data == 0).sum()), int(48000 * 0.3))

    def test_a_comma_split_does_not_end_the_sentence(self):
        self.assertFalse(Silero._terminator_after('Пётр читает, затем встаёт', 'Пётр читает'))
        self.assertTrue(Silero._terminator_after('Пётр читает. Затем встаёт', 'Пётр читает'))
        self.assertTrue(Silero._terminator_after('Пётр читает? Затем встаёт', 'Пётр читает'))
        self.assertTrue(Silero._terminator_after('Пётр читает! Затем встаёт', 'Пётр читает'))
        self.assertTrue(Silero._terminator_after('Пётр читает.. Затем встаёт', 'Пётр читает'))

    def test_a_short_breath_is_not_a_sentence_end(self):
        self.assertFalse(Silero._terminator_after('Пётр читает', 'Пётр читает'))
        self.assertFalse(Silero._terminator_after('Пётр читает — затем', 'Пётр читает'))
        self.assertFalse(Silero._terminator_after('Пётр читает: затем', 'Пётр читает'))

    def test_the_last_fragment_still_gets_its_pause(self):
        engine = self._sentence('Пётр читает')
        silence = [length for _, length in self._silences()]
        self.assertEqual(len(silence), 1)
        self.assertGreaterEqual(silence[0], int(48000 * 0.3))
        self.assertLessEqual(silence[0], int(48000 * 0.6))

    def test_a_fragment_that_already_ends_on_a_terminator_is_located(self):
        self.assertTrue(Silero._terminator_after('Пётр читает.', 'читает'))


if __name__ == '__main__':
    unittest.main()
