"""Stress (accent) pre-processing of the SML text.

silero-stress marks the stressed vowel of every russian word with a ``+`` and
resolves the homographs (``замок`` -> ``зам+ок``). The Silero checkpoints
honour such marks; any other engine would read the ``+`` as a stray character.
The helpers here therefore mark only the plain text and leave the SML tags (and
the voice paths inside them) untouched, and can take the marks back out when
the target engine is not Silero.
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
# A mark is a '+' right in front of a vowel: '+2' or 'A+B' are left alone.
_MARK = re.compile(r"\+(?=[%s])" % _VOWELS, re.IGNORECASE)

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
    """Mark the stress of a plain text (no SML tags)."""
    if not text or not any(character.isalnum() for character in text):
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
    """Remove the stress marks from a plain text (no SML tags)."""
    if not text or "+" not in text:
        return text
    return _MARK.sub("", text)


def accent_sml(text: str, accentor=None) -> str:
    """Mark the stress in the text of an SML document, tags untouched."""
    return _outside_tags(text, lambda chunk: accent_text(chunk, accentor=accentor))


def strip_sml(text: str) -> str:
    """Remove the stress marks from an SML document, tags untouched."""
    return _outside_tags(text, strip_text)


def is_silero(tts_engine) -> bool:
    return str(tts_engine or "").strip().lower() in SILERO_ENGINES


def stress_for_engine(text: str, tts_engine=None, accentor=None) -> str:
    """Accent the SML text for Silero, take the marks out for anything else."""
    if is_silero(tts_engine):
        return accent_sml(text, accentor=accentor)
    return strip_sml(text)
