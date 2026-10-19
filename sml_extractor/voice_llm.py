"""LLM-assisted voice casting driven by short character descriptions.

The heuristic matcher picks a voice from gender/age alone.  When the user
provides key characters with descriptions (or simply enables the LLM), this
module asks a local LLM to suggest a specific voice per character and then
applies the suggestion defensively: unknown names are ignored, voices must
exist, genders must match, and every character still gets a deterministic
fallback voice if the model is unavailable or unconvincing.
"""

from __future__ import annotations

from pathlib import Path

from sml_extractor.llm_client import LlmError, get_client
from sml_extractor.voice_matcher import GENDERS, _find_best_voice, is_voice_excluded

# A novel can list hundreds of one-line speakers; the model cannot cast all of
# them in one answer (the JSON would be truncated), so only the important ones
# are sent and the rest are filled deterministically afterwards.
MAX_CAST = 40

_SYSTEM_PROMPT = """You are a casting director for audiobooks. For every character pick exactly one voice from the available list.

Rules:
- Honour the gender whenever it is known: never give a male character a female voice or vice versa.
- Keep voices distinct; reuse a voice only if the list is too small for the cast.
- Give the Narrator a clear, neutral, pleasant voice.
- Use the character description when it is present. When it is missing, rely on the name, gender and age.
- Prefer voices of the right age category (adult/elder/teen/child).
- Do not invent voices: the "voice" field must be copied verbatim from the available list.

Return STRICT JSON only:
{"assignments":[{"character":"<name exactly as listed>","voice":"<voice name>","why":"<short reason>"}]}
"""

_LANGUAGE_NAMES_FOR_REASON = {
    "rus": "Russian",
    "ru": "Russian",
    "eng": "English",
    "en": "English",
}

# One assignment line per character; bound the answer so a reasoning model
# cannot ramble until the request times out.
_CAST_MAX_TOKENS = 4096


def _candidates(voice_library: dict, custom_voices: list | None = None) -> list:
    """Flatten a voice library into ``{name, path, gender, age}`` dicts.

    Excluded (cartoon / meme / advert) voices are dropped from the bundled
    library; explicit custom voices are always kept.
    """
    candidates = []
    for age, genders in (voice_library or {}).items():
        for gender, paths in (genders or {}).items():
            for path in paths or []:
                if is_voice_excluded(path):
                    continue
                candidates.append({
                    "name": Path(path).stem,
                    "path": path,
                    "gender": gender,
                    "age": age,
                })
    for path in custom_voices or []:
        candidates.append({
            "name": Path(path).stem,
            "path": path,
            "gender": "unknown",
            "age": "unknown",
        })
    return candidates


def quote_counts(booknlp_data: dict) -> dict:
    """Count attributed quotes per character name (``{"Narrator": 120, ...}``)."""
    if not booknlp_data:
        return {}
    # Imported lazily so this module stays importable without the NLP stack.
    from sml_extractor.sml_generator import _build_coref_name_map

    name_map = _build_coref_name_map(booknlp_data.get("book_data"))
    counts = {}
    for quote in booknlp_data.get("quotes", []):
        name = name_map.get(str(quote.get("char_id")), "Narrator")
        counts[name] = counts.get(name, 0) + 1
    return counts


def select_cast(
    characters: list,
    counts: dict | None = None,
    limit: int = MAX_CAST,
) -> list:
    """Pick the characters worth an LLM suggestion.

    Key characters and the Narrator are always included; the remaining slots go
    to the characters with the most quotes.  Everyone else keeps a voice from
    the deterministic matcher, which is what a long tail of one-line speakers
    gets anyway.
    """
    if limit <= 0 or len(characters) <= limit:
        return list(characters)

    counts = counts or {}
    priority = []
    rest = []
    for char in characters:
        if char.get("key_character") or char.get("normalized_name") == "Narrator":
            priority.append(char)
        else:
            rest.append(char)
    rest.sort(
        key=lambda char: -counts.get(char.get("normalized_name", ""), 0)
    )
    return priority[:limit] + rest[: max(0, limit - len(priority))]


def _system_prompt(language: str) -> str:
    reason_language = _LANGUAGE_NAMES_FOR_REASON.get(
        (language or "").lower(), "the language of the descriptions"
    )
    return (
        _SYSTEM_PROMPT
        + f'\nWrite every "why" value in {reason_language}, at most 12 words.\n'
    )


def _user_prompt(characters: list, candidates: list) -> str:
    character_lines = []
    for char in characters:
        name = char.get("normalized_name", "")
        if not name:
            continue
        gender = char.get("inferred_gender", "unknown")
        age = char.get("inferred_age_category", "unknown")
        description = str(char.get("description") or "").replace("\n", " ").strip()
        suffix = f" | {description[:220]}" if description else ""
        character_lines.append(f"- {name} | {gender} | {age}{suffix}")

    voice_lines = [
        f"- {candidate['name']} | {candidate['gender']} | {candidate['age']}"
        for candidate in candidates
    ]

    return (
        "CHARACTERS (name | gender | age | notes):\n"
        + "\n".join(character_lines)
        + "\n\nAVAILABLE VOICES (name | gender | age):\n"
        + "\n".join(voice_lines)
        + "\n\nAssign exactly one voice to every character."
    )


def _parse_proposal(payload, candidates: list):
    """Turn the raw LLM answer into ``(proposal, attributes)``.

    ``proposal`` maps a character to ``{"path": ..., "why": ...}``.  Reasons are
    only reported once a suggestion is actually accepted, so a rejected one
    never explains a fallback voice.
    """
    if isinstance(payload, dict):
        items = payload.get("assignments") or payload.get("characters") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    by_name = {candidate["name"].lower(): candidate for candidate in candidates}
    by_path = {str(candidate["path"]): candidate for candidate in candidates}

    proposal = {}
    attributes = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("character") or item.get("name") or "").strip()
        if not name:
            continue

        voice_name = str(item.get("voice") or item.get("voice_name") or "").strip()
        candidate = by_name.get(voice_name.lower()) or by_path.get(voice_name)
        if candidate is not None and name not in proposal:
            proposal[name] = {
                "path": candidate["path"],
                "why": str(item.get("why") or item.get("reason") or "").strip(),
            }

        gender = str(item.get("gender") or "").strip().lower()
        age = str(item.get("age") or "").strip().lower()
        if (gender in ("male", "female") or age) and name not in attributes:
            attributes[name] = (gender, age)

    return proposal, attributes


def _apply_attributes(characters: list, attributes: dict) -> None:
    """Fill in gender/age the LLM inferred, never overwriting known values."""
    for char in characters:
        name = char.get("normalized_name", "")
        if name not in attributes:
            continue
        gender, age = attributes[name]
        gender = str(gender or "").strip().lower()
        age = str(age or "").strip().lower()

        if char.get("inferred_gender") in (None, "", "unknown") and gender in {"male", "female"}:
            char["inferred_gender"] = gender
        if (
            char.get("inferred_age_category") in (None, "", "unknown")
            and age in {"child", "teen", "adult", "elder"}
        ):
            char["inferred_age_category"] = age


def _assign(
    characters: list,
    voice_library: dict,
    custom_voices: list | None,
    candidates: list,
    proposal: dict,
    counts: dict | None = None,
    allow_reuse: bool = True,
) -> dict:
    """Apply the LLM proposal, then fill the gaps deterministically.

    The deterministic pass walks the cast by importance (key characters first,
    then the busiest speakers) because the voice pool is much smaller than a
    novel's cast; when it is exhausted a voice is reused rather than dropped.

    Returns ``(assignments, reasons)``; a reason is reported only for the
    suggestion that was actually used.
    """
    gender_by_path = {
        str(candidate["path"]): candidate["gender"] for candidate in candidates
    }
    known_paths = set(gender_by_path)

    assignments = {}
    reasons = {}
    used = set()

    for char in characters:
        name = char.get("normalized_name", "")
        if not name or name in assignments:
            continue
        suggestion = proposal.get(name)
        if not suggestion:
            continue
        voice = suggestion["path"]
        if voice in used or voice not in known_paths:
            continue
        char_gender = char.get("inferred_gender", "unknown")
        voice_gender = gender_by_path.get(voice, "unknown")
        if (
            char_gender in GENDERS
            and voice_gender in GENDERS
            and voice_gender != char_gender
        ):
            continue
        assignments[name] = voice
        used.add(voice)
        if suggestion.get("why"):
            reasons[name] = suggestion["why"]

    counts = counts or {}
    priority = sorted(
        characters,
        key=lambda item: (
            0 if item.get("key_character") else 1,
            -counts.get(item.get("normalized_name", ""), 0),
        ),
    )
    for char in priority:
        name = char.get("normalized_name", "")
        if not name or name in assignments:
            continue
        voice = _find_best_voice(
            char.get("inferred_gender", "unknown"),
            char.get("inferred_age_category", "adult"),
            voice_library,
            used,
            custom_voices,
        )
        if voice:
            assignments[name] = voice
            used.add(voice)

    if allow_reuse:
        # The pool ran out, but a shared voice beats silence.
        for char in priority:
            name = char.get("normalized_name", "")
            if not name or name in assignments:
                continue
            voice = _find_best_voice(
                char.get("inferred_gender", "unknown"),
                char.get("inferred_age_category", "adult"),
                voice_library,
                set(),
                custom_voices,
            )
            if voice:
                assignments[name] = voice

    return assignments, reasons


def assign_voices_with_llm(
    characters: list,
    voice_library: dict,
    custom_voices: list | None = None,
    client=None,
    language: str = "eng",
    counts: dict | None = None,
    limit: int = MAX_CAST,
) -> tuple[dict, dict]:
    """Cast voices with an LLM, falling back to the heuristic matcher.

    Args:
        counts: optional ``quote_counts()`` map used to pick which characters
            the LLM is asked about; the rest are cast deterministically.
        limit: maximum number of characters sent to the LLM.

    Returns:
        ``(assignments, reasons)`` — a character→voice-path map plus a short
        explanation per character.  Both are empty when no cast could be made
        with the LLM; callers should then run the plain matcher.
    """
    candidates = _candidates(voice_library, custom_voices)
    if not characters or not candidates:
        return {}, {}

    cast = select_cast(characters, counts, limit)
    client = client or get_client()
    try:
        payload = client.chat_json(
            _system_prompt(language),
            _user_prompt(cast, candidates),
            max_tokens=_CAST_MAX_TOKENS,
        )
    except LlmError as exc:
        print(f"[voice_llm] LLM casting skipped: {exc}")
        return {}, {}

    proposal, attributes = _parse_proposal(payload, candidates)
    _apply_attributes(characters, attributes)
    assignments, reasons = _assign(
        characters, voice_library, custom_voices, candidates, proposal, counts
    )
    if not assignments:
        return {}, {}
    return assignments, reasons
