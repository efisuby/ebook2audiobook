"""E2A-facing glue for the ported SML pipeline.

The steps themselves live in the ported modules — BookNLP (:mod:`core`),
descriptions (:mod:`character_profile`), casting (:mod:`voice_matcher`),
emotion (:mod:`emotion`), stress (:mod:`stress`) and the writer
(:mod:`sml_generator`).  This module runs them in the order the wizard needs
and hands back plain data, so the GUI can show every step (characters,
descriptions, voices, stress, SML) to the user before a single sentence is
synthesized.

Nothing here imports E2A: the caller passes the language, the target engine
and, when the library lives elsewhere, the application root.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from . import character_profile, emotion, sml_generator, stress, voice_matcher
from . import traced
from .core import (
    convert_ebook_to_txt,
    extract_characters,
    load_booknlp_output,
    run_booknlp,
)

NARRATOR = "Narrator"

#: The E2A checkout that carries this package: ``voices/`` and the default
#: ``data/`` live next to it.
APP_ROOT = Path(__file__).resolve().parent.parent

#: Formats BookNLP can read directly instead of going through Calibre.
TEXT_SUFFIXES = {".txt", ".md", ".text"}

#: A single preview line shorter than this reads as a grunt rather than as a
#: sample of the voice, so the narrator sample asks for a longer one.
MIN_SAMPLE_CHARS = 40

#: A voice is judged on a passage, not on one line: the wizard asks for a few
#: dozen words and stops at a sentence boundary near the upper bound.
DEFAULT_SAMPLE_WORDS = 40
MAX_SAMPLE_WORDS = 90


def _progress_shim(progress):
    """Adapt ``progress(fraction, desc=…)`` to BookNLP's ``(message, pct)``."""
    if not progress:
        return None

    def report(message, percent):
        fraction = max(0.0, min(1.0, float(percent) / 100.0))
        try:
            progress(fraction, desc=str(message))
        except TypeError:
            progress(fraction)

    return report


@traced(
    "Разбор книги (BookNLP)",
    describe=lambda result: (
        f"книга {result['book_id']}, персонажей {len(result['characters'])}"
    ),
)
def analyze_book(
    book_file,
    output_dir,
    language="rus",
    model="small",
    data_dir=None,
    progress=None,
) -> dict:
    """Parse a book with BookNLP and extract its characters.

    ``book_file`` may be any ebook format or a plain text file.  The token and
    quote tables come back inside ``booknlp_data``: they are large, so a caller
    keeps them in memory (or reloads them from ``output_dir`` with
    :func:`load_book`) instead of storing them in a session.

    Returns ``{'book_id', 'output_dir', 'text_file', 'booknlp_data',
    'characters'}``.
    """
    book_file = Path(book_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if book_file.suffix.lower() in TEXT_SUFFIXES:
        text_file = str(book_file)
    else:
        text_file = convert_ebook_to_txt(str(book_file), str(output_dir))

    result = run_booknlp(
        input_file=text_file,
        output_dir=str(output_dir),
        model=model,
        language=language,
        progress_callback=_progress_shim(progress),
        data_root=data_dir,
    )
    book_id = result.get('book_id') or Path(text_file).stem
    booknlp_data = load_booknlp_output(str(output_dir), book_id)

    return {
        'book_id': book_id,
        'output_dir': str(output_dir),
        'text_file': text_file,
        'booknlp_data': booknlp_data,
        'characters': extract_characters(booknlp_data, language=language),
    }


@traced(
    "Загрузка разобранной книги",
    describe=lambda result: f"книга {result['book_id']}, персонажей {len(result['characters'])}",
)
def load_book(output_dir, book_id, language="rus") -> dict:
    """Reload an already analyzed book (a session keeps only the paths)."""
    output_dir = str(Path(output_dir).expanduser().resolve())
    booknlp_data = load_booknlp_output(output_dir, book_id)
    return {
        'book_id': book_id,
        'output_dir': output_dir,
        'booknlp_data': booknlp_data,
        'characters': extract_characters(booknlp_data, language=language),
    }


@traced(
    "Описания персонажей",
    describe=lambda result: f"описаний {len(result)}",
)
def describe_characters(
    characters,
    booknlp_data=None,
    language="rus",
    use_llm=True,
    progress=None,
) -> list:
    """Write a short description for every character.

    The heuristic is written first and the local model replaces it for the
    characters it has enough material for (see
    :mod:`sml_extractor.character_profile`); this wrapper never raises.
    """
    return character_profile.describe_characters(
        list(characters or []),
        booknlp_data,
        use_llm=use_llm,
        language=language,
        progress=progress,
    )


@traced(
    "Подбор голосов",
    describe=lambda result: (
        f"ролей {len(result)}, с голосом {sum(1 for c in result if c.get('voice'))}"
    ),
)
def cast_characters(
    booknlp_data,
    characters,
    language="rus",
    app_root=None,
    custom_voices=None,
    voice_overrides=None,
) -> list:
    """Give every speaking character a voice from the E2A voice library.

    The narrator and the characters with at least one quote of their own are
    kept, in reading order (narrator first, then by number of lines).  Each one
    gets an absolute path to a voice file, picked by gender and age category; a
    voice already stored on the character, or one in ``voice_overrides``
    (``{name: path}``), is never replaced.

    A speaker heard in the book but missing from ``characters`` (possible when
    the analysis ran on a plain text file, where there are no quote tables to
    list the cast from) is added to ``characters`` as an unknown adult, so the
    wizard can offer it a voice too.  The narrator is always part of the cast,
    even when the analysis found no narration for it.  Returns the cast list
    with ``voice`` filled in; call :func:`describe_characters` afterwards to
    cover the added records as well.
    """
    characters = characters if isinstance(characters, list) else list(characters or [])
    root = Path(app_root).expanduser().resolve() if app_root else APP_ROOT
    speaking = sml_generator.speaking_characters(booknlp_data or {}, characters)
    known = {character.get('normalized_name') for character in speaking}
    by_name = {
        character.get('normalized_name'): character for character in characters
    }
    if NARRATOR not in known:
        # The narration has to be cast as well, even when BookNLP did not list
        # a narrator of its own: without a narrator the wizard would offer no
        # voice for the narration at all.
        known.add(NARRATOR)
        speaking.append(
            by_name.get(NARRATOR)
            or {
                'normalized_name': NARRATOR,
                'inferred_gender': 'unknown',
                'inferred_age_category': 'adult',
                'voice': None,
            }
        )
    for speaker, _text in sml_generator.collect_speech_segments(booknlp_data or {}):
        if not speaker or speaker in known:
            continue
        known.add(speaker)
        speaking.append(
            by_name.get(speaker)
            or {
                'normalized_name': speaker,
                'inferred_gender': 'unknown',
                'inferred_age_category': 'adult',
                'voice': None,
            }
        )
    # Every record the cast is made of must exist in ``characters`` as well:
    # the descriptions are written there and the wizard writes the voice back.
    for record in speaking:
        name = record.get('normalized_name')
        if name and name not in by_name:
            by_name[name] = record
            characters.append(record)
    lines = Counter(
        speaker for speaker, _text in sml_generator.collect_speech_segments(booknlp_data or {})
    )
    order = {
        character.get('normalized_name'): index
        for index, character in enumerate(characters or [])
    }
    speaking.sort(
        key=lambda character: (
            character.get('normalized_name') != NARRATOR,
            -lines.get(character.get('normalized_name'), 0),
            order.get(character.get('normalized_name'), len(order)),
        )
    )

    library = voice_matcher.scan_voice_library(str(root), language)
    assigned = voice_matcher.auto_assign_voices(
        speaking, library, custom_voices=custom_voices
    )
    overrides = dict(voice_overrides or {})

    for character in speaking:
        name = character.get('normalized_name')
        path = overrides.get(name) or character.get('voice') or assigned.get(name)
        if path and not os.path.isabs(str(path)):
            path = str(root / path)
        character['voice'] = path or None
    return speaking


@traced(
    "Строки таблицы персонажей",
    describe=lambda result: f"строк {len(result)}",
)
def character_rows(booknlp_data, characters, voices=None) -> list[dict]:
    """The data behind the wizard's character table, narrator first.

    Each row carries what the voice picker shows next to a dropdown:
    ``name``, ``gender``, ``age``, ``lines`` (quotes of that character),
    ``description`` (+ its ``description_source``), the ``voice`` path and the
    voice name to display.  The narrator comes first, then the busiest
    characters, then the order they appear in ``characters``.
    """
    # Try to compute from BookNLP data; if unavailable/empty, fall back to
    # the ``lines`` field already stored on the character records (session restore).
    lines_from_booknlp = Counter(
        speaker for speaker, _text in sml_generator.collect_speech_segments(booknlp_data or {})
    )
    order = {
        character.get('normalized_name'): index
        for index, character in enumerate(characters or [])
    }
    rows = []
    for character in characters or []:
        name = character.get('normalized_name')
        if not name:
            continue
        path = (voices or {}).get(name) or character.get('voice') or ''
        # prefer BookNLP count; if 0 (data missing), use saved lines field
        line_count = lines_from_booknlp.get(name, 0)
        if line_count == 0:
            line_count = character.get('lines', 0)
        rows.append(
            {
                'name': name,
                'gender': character.get('inferred_gender') or 'unknown',
                'age': character.get('inferred_age_category') or 'adult',
                'lines': line_count,
                'description': character.get('description') or '',
                'description_source': character.get('description_source') or '',
                'voice': path,
                'voice_name': Path(str(path)).stem if path else '',
            }
        )
    rows.sort(
        key=lambda row: (
            row['name'] != NARRATOR,
            -row['lines'],
            order.get(row['name'], len(order)),
        )
    )
    return rows


def sample_line(booknlp_data, name, min_chars=MIN_SAMPLE_CHARS) -> str:
    """A single line of ``name`` — the text the voice preview synthesizes.

    A character gets its longest line among the first few quotes (the opening
    one is often just "Да."), the narrator the first narration long enough to
    show the voice.  A book that carries no data at all gives an empty line
    instead of an error: the wizard asks for a sample before it has one.
    """
    if not name:
        return ''
    try:
        return _sample_line(booknlp_data or {}, name, min_chars)
    except ValueError:
        return ''


def _sample_line(booknlp_data, name, min_chars):
    segments = sml_generator.book_segments(booknlp_data)

    if name != NARRATOR:
        quotes = [
            text
            for speaker, text in sml_generator.collect_speech_segments(booknlp_data)
            if speaker == name and text.strip()
        ]
        if quotes:
            return max(quotes[:5], key=len).strip()

    for speaker, text in segments:
        if not speaker and len(text.strip()) >= min_chars:
            return text.strip()
    for _speaker, text in segments:
        if text.strip():
            return text.strip()
    return ''


def sample_text(
    booknlp_data,
    name,
    words=DEFAULT_SAMPLE_WORDS,
    max_words=MAX_SAMPLE_WORDS,
) -> str:
    """A passage of ``name`` long enough to judge a voice.

    A few dozen words say much more about a voice than a single line.  The
    longest run of *consecutive* segments of that speaker is preferred; when the
    book only gives short one-line quotes, that character's quotes are strung
    together instead.  The result is cut at a segment boundary near
    ``max_words``.  Falls back to :func:`sample_line` when the speaker has
    nothing of its own (the narrator reads the narration).
    """
    if not name:
        return ''
    try:
        segments = sml_generator.book_segments(booknlp_data or {})
    except ValueError:
        return ''

    def mine(speaker) -> bool:
        return not speaker if name == NARRATOR else speaker == name

    runs: list[list[str]] = []
    current: list[str] = []
    for speaker, text in segments:
        text = (text or '').strip()
        if not text:
            continue
        if mine(speaker):
            current.append(text)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    limit = max(1, int(max_words))
    best = max(runs, key=_word_count, default=[])
    if _word_count(best) >= max(1, int(words)):
        return _join_words(best, limit)
    own = [text for speaker, text in segments if mine(speaker) and (text or '').strip()]
    if own:
        return _join_words(own, limit)
    return sample_line(booknlp_data, name)


def _word_count(chunks) -> int:
    return sum(len(str(chunk).split()) for chunk in (chunks or []))


def _join_words(chunks, limit) -> str:
    """Join whole chunks while the running word count stays under ``limit``."""
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        chunk = str(chunk).strip()
        if not chunk:
            continue
        count = len(chunk.split())
        if parts and total + count > limit:
            break
        parts.append(chunk)
        total += count
    return ' '.join(parts).strip()


def preview_lines(
    booknlp_data,
    characters,
    voices=None,
    per_character=1,
    opening=6,
    width=160,
) -> list[str]:
    """The wizard's text preview.

    One line per character — its voice, its description, a sample of its own
    speech — followed by the opening of the book as it will be read.
    ``voices`` overrides the cast (the wizard passes the single voice of the
    «Разные голоса»-off mode, so the preview shows what will really be heard).
    """
    segments = sml_generator.book_segments(booknlp_data or {})
    samples: dict = {}
    for speaker, text in segments:
        if not speaker or not text.strip():
            continue
        bucket = samples.setdefault(speaker, [])
        if len(bucket) < max(1, per_character):
            bucket.append(text.strip())

    lines = []
    for row in character_rows(booknlp_data, characters, voices=voices):
        lines.append(
            f"{row['name']} — {row['voice_name'] or 'без голоса'} — {row['description']}"
        )
        for text in samples.get(row['name'], []):
            lines.append(f"    «{_trim(text, width)}»")

    spoken = [pair for pair in segments if (pair[1] or '').strip()]
    if spoken:
        lines.append('')
        lines.append('— Начало книги —')
        for speaker, text in spoken[: max(0, opening)]:
            who = NARRATOR if not speaker else speaker
            lines.append(f"{who}: {_trim(text.strip(), width)}")
    return lines


@traced("Сборка SML", describe=lambda result: f"символов {len(result)}")
def build_sml(
    booknlp_data,
    characters,
    voices=None,
    language="rus",
    tts_engine=None,
    use_emotions=True,
    use_stress=True,
    client=None,
    progress=None,
) -> str:
    """Render the whole book as SML: voice tags, emotions and stress.

    Emotions come from the local model (a broken model falls back to the
    heuristic).  Stress marks are kept for Silero and stripped for every other
    engine, so the same document can be synthesized by any of them.
    """
    booknlp_data = booknlp_data or {}
    voice_assignments = _voice_assignments(characters, voices)

    emotion_map = {}
    if use_emotions:
        emotion_map = emotion.analyze_book_emotions(
            sml_generator.collect_speech_segments(booknlp_data),
            client=client,
            progress=progress,
        )

    sml_text = sml_generator.render_sml_text(
        booknlp_data,
        list(characters or []),
        voice_assignments=voice_assignments,
        use_macros=False,
        emotion_map=emotion_map,
    )
    if use_stress:
        sml_text = stress.stress_for_engine(sml_text, tts_engine)
    return sml_text


def stress_preview(
    booknlp_data,
    characters,
    voices=None,
    tts_engine="silero",
    lines=6,
) -> str:
    """The opening of the book as SML, run through the stress step.

    Shows the words the engine will hear marked (``зам+ок``) — or, for a
    non-Silero engine, the same fragment without any marks.
    """
    voice_assignments = _voice_assignments(characters, voices)
    fragments = []
    for speaker, text in sml_generator.book_segments(booknlp_data or {})[: max(1, lines)]:
        text = (text or '').strip()
        if not text:
            continue
        path = voice_assignments.get(speaker) if speaker else None
        fragments.append(f"[voice:{path}]{text}[/voice]" if path else text)
    return stress.stress_for_engine('\n'.join(fragments), tts_engine)


def sml_blocks(sml_text) -> list[str]:
    """Split an SML document into the blocks E2A synthesizes.

    The SML writer puts every voice block on its own line precisely so that a
    line is a block; E2A then cuts the sentences (and the inline tags) itself.
    """
    return [line.strip() for line in str(sml_text or '').splitlines() if line.strip()]


def is_silero(tts_engine) -> bool:
    """Whether the stress marks of this document will actually be used."""
    return stress.is_silero(tts_engine)


def _voice_assignments(characters, voices) -> dict:
    """Merge the explicit ``{name: path}`` map with the character records."""
    assignments = {
        name: path for name, path in dict(voices or {}).items() if name and path
    }
    for character in characters or []:
        name = character.get('normalized_name')
        if name and name not in assignments and character.get('voice'):
            assignments[name] = character['voice']
    return assignments


def _trim(text: str, width: int) -> str:
    text = ' '.join(str(text).split())
    return text if len(text) <= width else text[: max(1, width - 1)].rstrip() + '…'
