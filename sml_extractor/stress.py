"""Stress (accent) pre-processing of the SML text.

silero-stress marks the stressed vowel of every russian word with a ``+`` and
resolves the homographs (``замок`` -> ``зам+ок``). That ``+`` is what the SML
document stores and what the model hears (``Г+ород``) — the checkpoints were
trained on it. Every other engine would read the mark as a stray character, so
the helpers here can take it back out.

Stress is applied as the LAST stage of the text filtering (see
lib.core.filter_blocks): the normalizers — math2words in particular — would
verbalize ``+`` into the word «плюс», so incoming marks are stripped before
them and re-applied afterwards. The helpers mark only the plain text, leaving
the SML tags (and the voice paths inside them) untouched.

U+00B4 (´) and the combining acute are handled only as leftovers of an earlier
variant of this file that wrote ``´`` instead of ``+``: they are converted back
to ``+`` or stripped, never written again.
"""

from __future__ import annotations

import re

# The four tags of lib/conf_models.TTS_SML. The pattern is duplicated here so
# that this package stays importable without E2A's lib/ (it is also used on its
# own); keep the tag names in sync with TTS_SML.
SML_TAG = re.compile(r"\[/?(?:break|pause|voice|emotion)\b[^\]]*\]", re.IGNORECASE)

# Engines that synthesize the stress marks themselves.
SILERO_ENGINES = ("silero",)

_VOWELS = "аеёиоуыэюяaeiouy"
# Stress marker: a '+' right before the stressed vowel ('Г+ород'), the form
# silero-stress returns and the Silero checkpoints were trained on.
STRESS_MARK = "+"
# U+00B4 ACUTE ACCENT — the mark an earlier variant of this file wrote in front
# of the vowel. Never written again, only converted back to '+' or stripped.
LEGACY_STRESS_MARK = "´"
# A mark is only a symbol right in front of a vowel: '2+2', 'A+B' or '+2' are
# ordinary text and must survive untouched.
_MARK = re.compile(r"\+(?=[%s])" % _VOWELS, re.IGNORECASE)
_MARK_LEGACY = re.compile(r"%s(?=[%s])" % (re.escape(LEGACY_STRESS_MARK), _VOWELS), re.IGNORECASE)
# combining acute (U+0301) after a vowel — the other common stress notation
_MARK_COMBINING = re.compile(r"(?<=[%s])\u0301" % _VOWELS, re.IGNORECASE)

_accentor = None


def get_accentor():
    """Load silero-stress once for the whole process, None when unavailable."""
    global _accentor
    if _accentor is None:
        try:
            from silero_stress import load_accentor

            _accentor = load_accentor("ru")
        except Exception as exc:  # missing or broken install: no marks at all
            print(f"[stress] silero-stress unavailable, the text is left unaccented: {exc}")
            _accentor = False
    return _accentor or None


def reset_accentor() -> None:
    """Forget the loaded accentor (used by the tests and by a re-install)."""
    global _accentor
    _accentor = None


def _outside_tags(text: str, decorate) -> str:
    """Run ``decorate`` on every plain chunk of ``text``, keep the tags as is."""
    if not text:
        return text
    parts = SML_TAG.split(text)
    tags = SML_TAG.findall(text)
    out = []
    for index, part in enumerate(parts):
        out.append(decorate(part))
        if index < len(tags):
            out.append(tags[index])
    return "".join(out)


def accent_text(text: str, accentor=None) -> str:
    """Mark the stress of a plain text (no SML tags).

    silero-stress returns '+' before the stressed vowel ('Г+ород') and that is
    exactly the form kept: the model was trained on it. A leftover U+00B4 mark
    is translated to '+' first, so a document written by the earlier variant of
    this file still reaches the engine as '+'.
    """
    if not text:
        return text
    if LEGACY_STRESS_MARK in text:
        text = text.replace(LEGACY_STRESS_MARK, STRESS_MARK)
    if not any(character.isalnum() for character in text):
        return text
    accentor = accentor if accentor is not None else get_accentor()
    if accentor is None:
        return text
    try:
        return accentor(text) or text
    except Exception as exc:
        print(f"[stress] accenting failed, the fragment is left as is: {exc}")
        return text


def strip_text(text: str) -> str:
    """Remove every stress mark from a plain text (no SML tags)."""
    if not text:
        return text
    if STRESS_MARK in text:
        text = _MARK.sub("", text)
    if LEGACY_STRESS_MARK in text:
        text = _MARK_LEGACY.sub("", text)
    if "\u0301" in text:
        text = _MARK_COMBINING.sub("", text)
    return text


def accent_sml(text: str, accentor=None) -> str:
    """Mark the stress in the text of an SML document, tags untouched."""
    return _outside_tags(text, lambda chunk: accent_text(chunk, accentor=accentor))


def strip_sml(text: str) -> str:
    """Remove the stress marks from an SML document, tags untouched."""
    return _outside_tags(text, strip_text)


def is_silero(tts_engine) -> bool:
    return str(tts_engine or "").strip().lower() in SILERO_ENGINES


def stress_for_engine(text: str, tts_engine=None, accentor=None) -> str:
    """Handle stress marks for the target engine.

    The marks are applied as the LAST filtering stage (lib.core.filter_blocks),
    after the normalizers — otherwise math2words would verbalize '+' into
    «плюс».

    For Silero: mark the stress with '+' (``Г+ород``), the form the checkpoints
    were trained on. For other engines: strip every mark, they would read it as
    a literal character.
    """
    if is_silero(tts_engine):
        return accent_sml(text, accentor=accentor)
    return strip_sml(text)
