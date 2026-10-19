"""The book-to-SML wizard behind the interface.

Pure helpers over the session dict: the GUI only has to call one function per
button and render what comes back.  The BookNLP result of the book lives in
``<session_dir>/sml/`` and the parsed tables (tokens, quotes) are kept in a
small in-process cache, so moving between characters does not re-read them.

Nothing here imports Gradio, and the session is just a dict-like object: any
caller can drive the whole wizard (see ``tests/test_sml_wizard.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import wave
from collections import OrderedDict

from sml_extractor import pipeline
from sml_extractor.voice_matcher import scan_voice_library

# ISO-639-3 codes BookNLP ships models for (see sml_extractor.core).
BOOKNLP_LANGUAGES = ("eng", "rus")
# lib.core.ebook_modes['TEXT'] — the mode that takes the text from the textarea.
BOOK_MODE_TEXT = "text"
# How many parsed books stay in memory (the tables are megabytes).
_CACHE_LIMIT = 2

#: The engine whose checkpoint has built-in voices of its own.
SILERO_ENGINE = "silero"
#: Marks a built-in speaker in the voice dropdown (the label only, never the value).
SILERO_TAG = "[silero]"
#: Mirrors ``Silero.speaker_pools`` — kept here so listing the voices does not
#: have to import an engine (and torch).
SILERO_SPEAKERS = (
    ("male", ("aidar", "eugene")),
    ("female", ("baya", "kseniya", "xenia")),
)
#: The genders and age categories a character can be edited to (see
#: ``sml_extractor.voice_matcher``).
GENDERS = ("male", "female", "unknown")
AGE_CATEGORIES = ("child", "teen", "adult", "elder", "unknown")

_books: "OrderedDict[tuple, dict]" = OrderedDict()
# md5 of the file the analysis was made from, keyed by the output directory:
# pressing "Анализ книги" twice must not run BookNLP again.  The textarea is
# dumped into the same path on every call, so a timestamp would always differ.
_digests: "dict[str, str]" = {}


class WizardError(Exception):
    """Something the user has to fix; the message is shown as it is."""


# ── session lookups ───────────────────────────────────────────────────────────

def wizard_dir(session) -> str:
    """Directory for the wizard's own files (inside the session directory)."""
    base = session.get("session_dir") or os.path.join("tmp", str(session.get("session_id") or "sml"))
    path = os.path.join(base, "sml")
    os.makedirs(path, exist_ok=True)
    return path


def language(session) -> str:
    """The book language as an ISO-639-3 code (``eng``, ``rus``, …)."""
    return str(session.get("language") or "").lower()


def engine(session) -> str:
    """The engine the user picked in the interface."""
    return str(session.get("tts_engine") or "")


def characters(session) -> list:
    return list(session.get("sml_characters") or [])


def voices(session) -> dict:
    """``{name: voice path}`` for the characters that already have one."""
    return {
        character.get("normalized_name"): character["voice"]
        for character in characters(session)
        if character.get("normalized_name") and character.get("voice")
    }


def book_file(session) -> str | None:
    """The file (or a dump of the textarea) the analysis should read."""
    text = (session.get("ebook_textarea") or "").strip()
    if session.get("ebook_mode") == BOOK_MODE_TEXT and text:
        return _dump_text(session, text)
    for key in ("ebook_src", "ebook_textarea_src"):
        candidate = session.get(key)
        if candidate and os.path.isfile(candidate):
            return candidate
    if text:
        return _dump_text(session, text)
    return None


def _dump_text(session, text: str) -> str:
    path = os.path.join(wizard_dir(session), "book.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _source_digest(path: str) -> str:
    """md5 of the file contents: the only stable identifier of the book."""
    digest = hashlib.md5()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


# ── parsed book cache ─────────────────────────────────────────────────────────

def _remember(data: dict) -> None:
    key = (data.get("output_dir"), data.get("book_id"))
    _books[key] = data
    _books.move_to_end(key)
    while len(_books) > _CACHE_LIMIT:
        _books.popitem(last=False)


def book_data(session) -> dict | None:
    """The parsed book: from memory, from disk, or ``None`` if not analysed."""
    output_dir = session.get("sml_output_dir")
    book_id = session.get("sml_book_id")
    if not output_dir or not book_id:
        return None
    key = (output_dir, book_id)
    if key in _books:
        _books.move_to_end(key)
        return _books[key]
    try:
        loaded = pipeline.load_book(output_dir, book_id, language=language(session))
    except Exception:
        return None
    # ``load_book`` wraps the tables; the cache (and every caller) wants the
    # tables themselves plus the two paths, exactly like ``analyze`` stores.
    booknlp_data = loaded.get("booknlp_data") if isinstance(loaded, dict) else None
    if not booknlp_data or not booknlp_data.get("book_txt"):
        return None
    data = {**booknlp_data, "output_dir": output_dir, "book_id": book_id}
    _remember(data)
    return data


def forget_book(session) -> None:
    """Drop the cached tables (a new analysis or a forgotten book)."""
    output_dir = session.get("sml_output_dir")
    for key in [key for key in _books if key[0] == output_dir]:
        _books.pop(key, None)


# ── steps ─────────────────────────────────────────────────────────────────────

def analyze(session, model: str = "small", use_llm: bool = True, progress=None,
            force: bool = False) -> str:
    """Parse the book, cast voices and describe the characters.

    Returns the status line for the interface.  Every call walks the whole
    path, so an unchanged book is only reported back (the digest of the source
    file is remembered per output directory; ``force`` re-runs it anyway).
    Raises :class:`WizardError` when there is nothing to analyse or BookNLP has
    no model for the language.
    """
    book = book_file(session)
    if not book:
        raise WizardError("Сначала выберите книгу (или вставьте текст) — анализировать нечего.")
    digest = _source_digest(book)
    if (
        not force
        and session.get("sml_characters")
        and book_data(session)
        and _digests.get(session.get("sml_output_dir")) == digest
    ):
        found = characters(session)
        narrated = sum(
            1 for character in found if character.get("description_source") == "llm"
        )
        return (
            f"Анализ уже сделан: персонажей {len(found)}, из них с описанием от "
            f"модели {narrated}. Книга с тех пор не менялась."
        )
    book_language = language(session)
    if book_language not in BOOKNLP_LANGUAGES:
        raise WizardError(
            f"BookNLP не поддерживает язык «{book_language or 'не выбран'}»: "
            f"доступны {' и '.join(BOOKNLP_LANGUAGES)}."
        )

    output_dir = wizard_dir(session)
    result = pipeline.analyze_book(
        book, output_dir, language=book_language, model=model, progress=progress
    )
    booknlp_data = result["booknlp_data"]
    found = list(result["characters"])
    pipeline.cast_characters(booknlp_data, found, language=book_language)
    pipeline.describe_characters(
        found, booknlp_data, language=book_language, use_llm=use_llm,
        progress=_scaled(progress, 0.8, 1.0),
    )

    forget_book(session)
    session["sml_output_dir"] = output_dir
    session["sml_book_id"] = result["book_id"]
    session["sml_characters"] = _plain(found)
    session["sml_text"] = None
    session["sml_applied"] = False
    session["sml_removed"] = []
    session["sml_selected"] = None
    _remember({**booknlp_data, "output_dir": output_dir, "book_id": result["book_id"]})
    _digests[output_dir] = digest
    save(session)

    narrated = sum(1 for character in found if character.get("description_source") == "llm")
    return (
        f"Готово: персонажей {len(found)}, из них с описанием от модели {narrated}. "
        f"Книга: {os.path.basename(book)}."
    )


def rows(session) -> list:
    """The character table: one list per row, ready for ``gr.Dataframe``."""
    found = characters(session)
    if not found:
        return []
    data = book_data(session)
    if data is not None:
        return [
            [row["name"], row["gender"], row["age"], row["lines"], row["description"], row["voice_name"]]
            for row in pipeline.character_rows(data, found)
        ]
    # the tables are gone (forgotten session): show what the session kept
    # count quotes from the session characters
    quote_counts = {}
    for character in found:
        name = character.get("normalized_name")
        if name:
            quote_counts[name] = character.get("lines", 0)
    return [
        [
            character.get("normalized_name", ""),
            character.get("inferred_gender", ""),
            character.get("inferred_age_category", ""),
            str(quote_counts.get(character.get("normalized_name", ""), "")),
            character.get("description", ""),
            _stem(character.get("voice")),
        ]
        for character in sorted(found, key=lambda item: item.get("normalized_name") != pipeline.NARRATOR)
    ]


def names(session) -> list:
    """Character names in the order of the table."""
    return [row[0] for row in rows(session)]


def voice_choices(session) -> list:
    """``[(label, absolute path)]`` of every voice of the book language.

    For Silero the checkpoint's own speakers come first, each tagged
    ``[silero]`` in the label, so they are easy to spot next to the library.
    """
    choices = _library_choices(session)
    if engine(session) == SILERO_ENGINE:
        choices = silero_choices(session) + choices
    return choices


def _library_choices(session) -> list:
    """``[(label, absolute path)]`` of every voice of the book language."""
    library = scan_voice_library(pipeline.APP_ROOT, language(session))
    choices = []
    for age in sorted(library):
        for gender in sorted(library[age]):
            for relative in sorted(library[age][gender]):
                path = os.path.join(str(pipeline.APP_ROOT), relative)
                choices.append((f"{_stem(path)} · {gender}/{age}", path))
    return choices


# ── the engine's own voices (Silero) ──────────────────────────────────────────

def silero_choices(session) -> list:
    """``[(label, absolute path)]`` for every built-in Silero speaker.

    The label carries ``[silero]`` so the dropdown tells them from the voice
    library; the value is a real (silent) wav file that exists only so the
    existing ``[voice:…]`` pipeline accepts it.  See :func:`builtin_speaker`.
    """
    if engine(session) != SILERO_ENGINE or not language(session):
        return []
    choices = []
    for gender, speakers in SILERO_SPEAKERS:
        for speaker in speakers:
            path = _silero_voice_file(session, gender, speaker)
            choices.append((f"{SILERO_TAG} {speaker} · {gender}", path))
    return choices


def silero_voices_dir(session) -> str:
    """Where the placeholder wavs of the built-in speakers live."""
    return os.path.join(wizard_dir(session), "voices")


def builtin_speaker(session, path) -> str:
    """The Silero speaker a wizard voice path stands for, or ``''``.

    ``Silero._speaker_for`` reads the *stem* of the voice path, while the
    engine's ``_set_voice`` only checks that the file exists and that the
    language and the gender appear in it — the placeholder wav is never read.
    """
    if not path:
        return ""
    resolved = os.path.abspath(str(path))
    root = os.path.abspath(silero_voices_dir(session))
    if not resolved.startswith(root + os.sep):
        return ""
    stem = _stem(resolved).lower()
    for _gender, speakers in SILERO_SPEAKERS:
        if stem in speakers:
            return stem
    return ""


def _silero_voice_file(session, gender, speaker) -> str:
    path = os.path.join(
        silero_voices_dir(session), language(session), "adult", gender, f"{speaker}.wav"
    )
    if not os.path.isfile(path):
        _write_silence(path)
    return os.path.abspath(path)


def _write_silence(path, rate=16000, seconds=0.05) -> None:
    """A valid, tiny, silent wav — a voice file the engine never decodes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * (2 * int(rate * seconds)))


# ── editing the cast ──────────────────────────────────────────────────────────

def remove_character(session, name: str) -> bool:
    """Drop a name that is not a character of the book.

    Its lines are no longer tagged, so the narrator reads them (see
    :func:`_voice_map`).  The name is remembered so a later edit cannot bring
    it back, and the built document is invalidated because it still refers to
    the deleted voice.
    """
    if not name:
        return False
    found = characters(session)
    kept = [character for character in found if character.get("normalized_name") != name]
    if len(kept) == len(found):
        return False
    session["sml_characters"] = kept
    removed = list(session.get("sml_removed") or [])
    if name not in removed:
        removed.append(name)
    session["sml_removed"] = removed
    if session.get("sml_selected") == name:
        session["sml_selected"] = None
    _invalidate(session)
    return True


def removed(session) -> list:
    """The names the user deleted (they are read by the narrator)."""
    return list(session.get("sml_removed") or [])


def set_gender(session, name: str, gender) -> bool:
    """Change a character's gender (what the voice is matched by)."""
    return _set_field(session, name, "inferred_gender", gender)


def set_age(session, name: str, age) -> bool:
    """Change a character's age category (what the voice is matched by)."""
    return _set_field(session, name, "inferred_age_category", age)


def _set_field(session, name: str, field: str, value) -> bool:
    if not name:
        return False
    found = characters(session)
    hit = False
    for character in found:
        if character.get("normalized_name") == name:
            # a programmatic re-render (the restore step) must not invalidate
            # a document that is already built, so an unchanged value is a no-op
            if (character.get(field) or None) == (value or None):
                return True
            if value:
                character[field] = value
            else:
                character.pop(field, None)
            hit = True
    if not hit:
        return False
    session["sml_characters"] = found
    _invalidate(session)
    return True


def _invalidate(session) -> None:
    """The cast changed, so a document built from it is out of date."""
    session["sml_text"] = None
    session["sml_applied"] = False
    save(session)


def selected(session) -> str | None:
    """The character the user last picked (kept across a restart)."""
    return session.get("sml_selected") or None


def set_selected(session, name) -> None:
    session["sml_selected"] = name or None


def set_voice(session, name: str, voice: str | None) -> bool:
    """Remember the voice the user picked for one character.

    A changed voice makes an already built document stale, so it is dropped
    (with the same unchanged-value guard as the gender/age fields, because a
    re-render feeds the stored value back through the picker).
    """
    for character in characters(session):
        if character.get("normalized_name") == name:
            if (character.get("voice") or None) == (voice or None):
                return True
            character["voice"] = voice or None
            _invalidate(session)
            return True
    return False


def voice_of(session, name: str) -> str | None:
    for character in characters(session):
        if character.get("normalized_name") == name:
            return character.get("voice")
    return None


def sample(session, name: str, words: int = pipeline.DEFAULT_SAMPLE_WORDS) -> str:
    """A passage of ``name`` — a few dozen words, not a single line.

    The voice preview synthesizes this, so a character that speaks in short
    lines still gets something to listen to.
    """
    data = book_data(session)
    if data is None or not name:
        return ""
    return pipeline.sample_text(data, name, words=words)


def preview(session, per_character: int = 1, opening: int = 6, sml_lines: int = 12) -> str:
    """The text preview: a line per character, the opening, and the SML head."""
    found = characters(session)
    data = book_data(session)
    if data is None or not found:
        return "Пока нечего показывать: нажмите «Анализ книги»."
    lines = pipeline.preview_lines(
        data, found, voices=voices(session), per_character=per_character, opening=opening
    )
    text = "\n".join(lines)
    sml = session.get("sml_text")
    if sml:
        head = "\n".join(pipeline.sml_blocks(sml)[:sml_lines])
        text += "\n\n— SML (начало) —\n" + head
    return text


def build(session, use_emotions: bool = True, use_stress: bool = True, progress=None) -> tuple[str, str]:
    """Render the SML document.  Returns ``(text, status line)``."""
    found = characters(session)
    data = book_data(session)
    if data is None or not found:
        raise WizardError("Сначала нажмите «Анализ книги».")
    text = pipeline.build_sml(
        data,
        found,
        voices=_voice_map(session),
        language=language(session),
        tts_engine=engine(session),
        use_emotions=use_emotions,
        use_stress=use_stress,
        progress=progress,
    )
    session["sml_text"] = text
    session["sml_applied"] = False
    blocks = pipeline.sml_blocks(text)
    spoken = sum(1 for block in blocks if block.startswith("[voice:"))
    return text, f"SML готов: блоков {len(blocks)}, из них с голосом {spoken}."


def _voice_map(session) -> dict:
    """``{name: path}`` for the document, with the two edits applied.

    A deleted name is mapped to the narrator's voice, so its lines are read by
    the narrator instead of the block voice.  The built-in speaker placeholders
    of Silero are dropped for every other engine: a placeholder is silence, and
    a cloning engine would happily clone it.
    """
    chosen = dict(voices(session))
    narrator_voice = chosen.get(pipeline.NARRATOR)
    for name in removed(session):
        if narrator_voice:
            chosen[name] = narrator_voice
    if engine(session) != SILERO_ENGINE:
        chosen = {
            name: path
            for name, path in chosen.items()
            if not builtin_speaker(session, path)
        }
    return chosen


def apply(session) -> str:
    """Hand the document over to the normal conversion path."""
    text = session.get("sml_text")
    if not text:
        raise WizardError("Сначала соберите SML.")
    session["sml_text"] = text
    session["sml_applied"] = True
    # keep it for the session persistence: a reload restores the textarea with
    # the document already in it, so the user stays on the "ready to voice" step
    session["ebook_textarea"] = text
    return text


def cancel(session) -> None:
    """The user went back to the voice list: forget the document as applied."""
    session["sml_applied"] = False


# ── persistence ───────────────────────────────────────────────────────────────

def save(session) -> None:
    """Store the analysis next to the BookNLP output (survives a restart)."""
    if not session.get("sml_book_id"):
        return
    output_dir = session.get("sml_output_dir")
    path = os.path.join(wizard_dir(session), "characters.json")
    payload = {
        "book_id": session.get("sml_book_id"),
        "output_dir": output_dir,
        "characters": characters(session),
        "removed": removed(session),
        "selected": selected(session),
        "source": _digests.get(output_dir),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)


def restore(session) -> bool:
    """Bring back an analysis made in an earlier session, if there is one."""
    if session.get("sml_book_id"):
        return True
    path = os.path.join(wizard_dir(session), "characters.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return False
    session["sml_book_id"] = payload.get("book_id")
    session["sml_output_dir"] = payload.get("output_dir")
    session["sml_characters"] = payload.get("characters") or []
    session["sml_removed"] = payload.get("removed") or []
    session["sml_selected"] = payload.get("selected")
    if payload.get("source"):
        _digests[session["sml_output_dir"]] = payload["source"]
    return bool(session["sml_book_id"])


def state(session) -> dict:
    """Everything the interface needs to show the wizard as it was left.

    Called after a reload or a restart: the session already carries the whole
    state, this is only the render of it.  ``restore`` is the disk fallback for
    a session that lost it (a new browser, a cleaned temp directory).
    """
    if not characters(session):
        restore(session)
    found = characters(session)
    if not found:
        return {
            "status": "Книга ещё не разобрана: нажмите «Анализ книги».",
            "rows": [],
            "names": [],
            "voices": [],
            "selected": None,
            "description": "",
            "gender": None,
            "age": None,
            "preview": preview(session),
        }
    name = selected(session)
    description = gender = age = None
    for row in rows(session):
        if row and row[0] == name:
            description, gender, age = row[4], row[1], row[2]
            break
    else:
        if name is not None:
            # the selected character is gone (deleted, or a fresh analysis)
            name = None
    text = session.get("sml_text")
    if text:
        status = f"Восстановлено: персонажей {len(found)}, SML из {len(pipeline.sml_blocks(text))} блоков."
    else:
        status = (
            f"Восстановлено: персонажей {len(found)}. "
            "Выберите голоса и нажмите «Собрать SML»."
        )
    return {
        "status": status,
        "rows": rows(session),
        "names": names(session),
        "voices": voice_choices(session),
        "selected": name,
        "description": description or "",
        "gender": gender,
        "age": age,
        "preview": preview(session),
    }


# ── small helpers ─────────────────────────────────────────────────────────────

def _scaled(progress, start: float, end: float):
    """Map a 0..1 callback of a later step onto ``start..end`` of the bar."""
    if progress is None:
        return None

    def scaled(fraction, desc=None):
        value = start + (end - start) * float(fraction or 0.0)
        try:
            progress(value, desc=desc)
        except TypeError:
            progress(value)

    return scaled


def _plain(value):
    """Session values must stay JSON-friendly: paths become strings."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _stem(path) -> str:
    return os.path.splitext(os.path.basename(path))[0] if path else ""
