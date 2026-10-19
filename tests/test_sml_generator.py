import tempfile
import unittest
from pathlib import Path

from sml_extractor.sml_generator import (
    _segments_from_book_txt,
    _segments_from_tokens,
    collect_speech_segments,
    generate_sml_output,
    portable_voice_assignments,
    speaking_characters,
)


def render(data: dict, voices: dict, **kwargs) -> str:
    """Write the SML for ``data`` and return it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'book.sml.txt'
        generate_sml_output(
            data, [], str(path), voice_assignments=voices,
            use_macros=False, **kwargs,
        )
        return path.read_text(encoding='utf-8')


QUOTE_DATA = {
    'tokens': [
        {'token_ID_within_document': str(i), 'paragraph_ID': '0', 'word': word}
        for i, word in enumerate(['“', 'Hello', '”', 'he', 'said', '.',
                                  '“', 'Again', '”'])
    ],
    'quotes': [
        {'quote_start': '0', 'quote_end': '2', 'char_id': '167'},
        {'quote_start': '6', 'quote_end': '8', 'char_id': '167'},
    ],
    'book_data': {'characters': [
        {'id': 167, 'mentions': {'proper': [{'n': 'Jeremiah'}]}}
    ]},
}


class VoiceAssignmentTests(unittest.TestCase):
    def test_speaking_characters_include_narrator_and_attributed_speakers(self):
        data = {
            'book_data': {'characters': [
                {'id': 167, 'mentions': {'proper': [{'n': 'Jeremiah'}]}},
                {'id': 3, 'mentions': {'proper': []}},
            ]},
            'quotes': [{'char_id': '167'}, {'char_id': '3'}],
        }
        characters = [
            {'normalized_name': 'Narrator'},
            {'normalized_name': 'Jeremiah'},
            {'normalized_name': 'Unused'},
        ]
        names = [c['normalized_name'] for c in speaking_characters(data, characters)]
        self.assertEqual(names, ['Narrator', 'Jeremiah', 'Character3'])

    def test_library_voice_paths_are_relative_for_e2a(self):
        paths = portable_voice_assignments(
            {'Jeremiah': 'voices/eng/adult/male/voice.wav'}, '/tmp/e2a'
        )
        self.assertEqual(paths['Jeremiah'], 'voices/eng/adult/male/voice.wav')


class SegmentTests(unittest.TestCase):
    def test_token_segments_split_on_quote_boundaries(self):
        segments = _segments_from_tokens(
            QUOTE_DATA['tokens'], QUOTE_DATA['quotes'], QUOTE_DATA['book_data']
        )
        self.assertEqual(segments, [
            ('Jeremiah', '“Hello”'),
            ('Narrator', 'he said.'),
            ('Jeremiah', '“Again”'),
        ])

    def test_book_txt_segments_keep_paragraph_breaks(self):
        segments = _segments_from_book_txt(
            '[Jeremiah] Hello [/]\n\nUndecorated line\n'
        )
        self.assertEqual(segments, [
            ('Jeremiah', 'Hello'),
            (None, ''),
            (None, 'Undecorated line'),
        ])

    def test_collect_speech_segments_matches_the_written_text(self):
        collected = collect_speech_segments(QUOTE_DATA)
        self.assertEqual(
            [speaker for speaker, _ in collected],
            ['Jeremiah', 'Narrator', 'Jeremiah'],
        )
        self.assertIn('“Hello”', collected[0][1])


class RenderTests(unittest.TestCase):
    def test_dialogue_and_narration_switch_voices(self):
        output = render(QUOTE_DATA, {
            'Narrator': 'narrator.wav', 'Jeremiah': 'jeremiah.wav',
        })
        self.assertEqual(output, (
            '[voice:jeremiah.wav]“Hello”[/voice]\n'
            '[voice:narrator.wav]he said.[/voice]\n'
            '[voice:jeremiah.wav]“Again”[/voice]'
        ))

    def test_unassigned_narrator_does_not_inherit_previous_voice(self):
        output = render(
            {'book_txt': '[Jeremiah] Hello [/]\n[Narrator] he said [/]\n'
                         '[Jeremiah] Again [/]'},
            {'Jeremiah': 'jeremiah.wav'},
        )
        self.assertEqual(output, (
            '[voice:jeremiah.wav]Hello[/voice]\n'
            'he said\n[voice:jeremiah.wav]Again[/voice]'
        ))

    def test_all_caps_line_keeps_voice_tags_together_for_calibre(self):
        output = render(
            {'book_txt': '[Narrator] before [/]\n[Narrator] PRINCESS MATTERS [/]\n'
                         '[Narrator] after [/]'},
            {'Narrator': 'narrator.wav'},
        )
        for line in output.splitlines():
            self.assertEqual(line.count('[voice:'), line.count('[/voice]'))
        self.assertIn('[voice:narrator.wav]PRINCESS MATTERS[/voice]', output)


class EmotionTagTests(unittest.TestCase):
    def test_emotion_tags_nest_inside_the_voice_block(self):
        segments = collect_speech_segments(QUOTE_DATA)
        text = segments[0][1]
        output = render(
            QUOTE_DATA,
            {'Narrator': 'narrator.wav', 'Jeremiah': 'jeremiah.wav'},
            emotion_map={text: 'angry'},
        )
        self.assertIn(
            f'[voice:jeremiah.wav][emotion:angry]{text}[/emotion][/voice]',
            output,
        )
        self.assertEqual(output.count('[emotion:'), output.count('[/emotion]'))

    def test_unlabelled_lines_stay_untagged(self):
        output = render(
            QUOTE_DATA,
            {'Narrator': 'narrator.wav', 'Jeremiah': 'jeremiah.wav'},
            emotion_map={},
        )
        self.assertNotIn('[emotion:', output)

    def test_an_unvoiced_line_can_still_carry_an_emotion(self):
        segments = collect_speech_segments(QUOTE_DATA)
        text = segments[1][1]
        output = render(QUOTE_DATA, {}, emotion_map={text: 'sad'})
        self.assertIn(f'[emotion:sad]{text}[/emotion]', output)


if __name__ == '__main__':
    unittest.main()
