"""Explicit key-character lists: parsing and merging with BookNLP output.

The GUI lets the user paste free-form notes about the important characters of a
book (often a bulleted list grouped by faction, sometimes with the role first
and the name second).  This module turns that text into structured records and
merges them into the character list produced by BookNLP, so the named
characters always show up, carry a short description, and can be cast a voice.

The LLM does the heavy lifting (free-form text → JSON); a small regex parser is
kept as an offline fallback.
"""

from __future__ import annotations

import re

from sml_extractor.core import _infer_gender_from_russian_name
from sml_extractor.llm_client import LlmError, get_client
from sml_extractor.sml_generator import _normalize_name

GENDERS = {"male", "female"}
AGES = {"child", "teen", "adult", "elder"}

# A reasoning model can ramble; bound the structured answer.
_PARSE_MAX_TOKENS = 4096

# Roles that appear *before* the dash in notes such as "Первый капитан — Севатар."
# Roles that appear *before* the dash in notes such as "Первый капитан — Севатар."
_ROLE_WORDS = (
    "капитан", "примарх", "сержант", "легион", "команд", "совет", "летопис",
    "итератор", "администратор", "офицер", "куратор", "архитектор", "воитель",
    "налётчик", "капеллан", "аброкарий", "магистр", "апотекарий",
    "герой", "рассказчик", "персонаж", "флот", "империум",
)

_BULLET_RE = re.compile(r"^\s*(?:[•·*\-–—▪]+\s*|\d+[.)]\s*)")
_SEPARATOR_RE = re.compile(r"\s*(?:—|–|―|-{1,2}|:)\s+")
_PROPER_CHUNK_RE = re.compile(
    r"^([А-ЯЁA-Z][\w'\-]*(?:\s+(?:[А-ЯЁA-Z][\w'\-]*|[а-яё]+\s+[А-ЯЁA-Z][\w'\-]*)){0,3})",
    re.UNICODE,
)

_SYSTEM_PROMPT = """You prepare a cast list for an audiobook voice director.

The user pastes free-form notes about the key characters of a book. The notes
are usually a bulleted list, sometimes grouped under faction or section
headings, and sometimes written in reverse ("Первый капитан — Севатар.").

Return STRICT JSON only, with this shape:
{"characters":[{"name":"...","aliases":["..."],"gender":"male|female|unknown","age":"child|teen|adult|elder|unknown","description":"..."}]}

Rules:
- One entry per distinct person; merge duplicate spellings and nicknames.
- "name": the shortest unambiguous canonical spelling, exactly as written in
  the notes (keep the original language and alphabet).
- "aliases": every other spelling, full name or nickname mentioned (empty list
  if none).
- Drop faction names, section headings, places and narrative commentary — only
  real people become entries.
- "gender": infer from the notes when possible, otherwise "unknown".
- "age": one of child/teen/adult/elder, otherwise "unknown".
- "description": one short line in the same language as the notes, mentioning
  the role, faction and anything useful for casting a voice (mood, age,
  gravitas). Do not invent facts that are not implied by the notes.
"""


def parse_key_characters(text: str, client=None, use_llm: bool = True) -> list:
    """Parse pasted key-character notes into structured records.

    Args:
        text: Raw user text.
        client: Optional :class:`~sml_extractor.llm_client.LlmClient` (mainly
            for tests).
        use_llm: When False, only the offline regex parser runs.

    Returns:
        List of dicts with keys ``name``, ``aliases``, ``gender``, ``age`` and
        ``description``.
    """
    text = (text or "").strip()
    if not text:
        return []

    if use_llm:
        try:
            raw = (client or get_client()).chat_json(
                _SYSTEM_PROMPT, text, max_tokens=_PARSE_MAX_TOKENS
            )
            parsed = _normalize_records(raw)
            if parsed:
                return parsed
        except LlmError as exc:
            print(f"[character_input] LLM parsing skipped: {exc}")

    return parse_key_characters_heuristic(text)


def parse_key_characters_heuristic(text: str) -> list:
    """Offline parser for ``Имя — описание`` style lists."""
    records = []
    seen = set()

    for line in (text or "").splitlines():
        stripped = _BULLET_RE.sub("", line).strip()
        stripped = stripped.strip("•·* \t")
        if len(stripped) < 3:
            continue
        # Faction/section headings such as "XVI легион, позже — Сыны Хоруса:".
        if stripped.endswith(":"):
            continue

        entry = _split_entry(stripped)
        if not entry:
            continue

        name, description = entry
        normalized = _normalize_name(name)
        if not normalized or normalized.lower() == "unknown" or normalized in seen:
            continue
        seen.add(normalized)

        gender = _guess_gender(name) or "unknown"
        records.append({
            "name": name,
            "aliases": [],
            "gender": gender,
            "age": "unknown",
            "description": description,
        })

    return _drop_cross_references(records)


_CROSS_REFERENCE_RE = re.compile(r"^(?:см\.?|смотри|see|same|то же|он же|aka)\b", re.IGNORECASE)


def _drop_cross_references(records: list) -> list:
    """Drop "Локен — см. выше." style entries that duplicate another name."""
    kept = []
    for record in records:
        if _CROSS_REFERENCE_RE.match(record.get("description", "")):
            name = _normalize_name(record["name"])
            if any(
                _match_score(name, _normalize_name(other["name"]))
                for other in records
                if other is not record
            ):
                continue
        kept.append(record)
    return kept


def _guess_gender(name: str) -> str | None:
    """Guess a Russian name's gender from its last (then first) word."""
    words = [word for word in re.split(r"\s+", name.strip()) if word]
    for word in (words[-1:], words[:1]):
        for candidate in word:
            gender = _infer_gender_from_russian_name(candidate.strip("«»\"'().,"))
            if gender:
                return gender
    return None


def _split_entry(line: str):
    """Split one note into ``(name, description)`` or return None."""
    # Drop discourse markers such as "Далее, Саул Тарвиль".
    line = re.sub(r"^(?:далее|также|затем|потом|кроме того|в частности)\s*[,:]\s*",
                  "", line, flags=re.IGNORECASE)
    parts = _SEPARATOR_RE.split(line, maxsplit=1)

    if len(parts) == 2:
        left, right = parts[0].strip(), parts[1].strip()
        if not left:
            return None
        # "Первый капитан — Севатар." puts the role first; take the proper
        # noun from the right-hand side as the name.
        if _looks_like_role(left):
            chunk = _PROPER_CHUNK_RE.match(right)
            if chunk:
                name = chunk.group(1).strip()
                rest = right[len(name):].strip(" .,;—-")
                description = " — ".join(part for part in (left, rest) if part)
                return name, description
        if right:
            return left.rstrip(" .,;—-"), right
        return None

    # No separator: keep the line only when it is a bare name.
    if len(line.split()) <= 3:
        return line.rstrip(" .,;—-"), ""
    return None


def _looks_like_role(text: str) -> bool:
    lowered = text.lower()
    if any(word in lowered for word in _ROLE_WORDS):
        return True
    # A multi-word phrase whose words are mostly lowercase is a role/heading,
    # not a name.
    words = text.split()
    return len(words) > 1 and not all(word[:1].isupper() for word in words)


def _normalize_records(raw) -> list:
    """Validate and clean an LLM answer into the canonical record shape."""
    if isinstance(raw, dict):
        items = raw.get("characters") or raw.get("characters_list") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    records = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("character") or "").strip()
        if not name:
            continue
        normalized = _normalize_name(name)
        if normalized in seen:
            continue
        seen.add(normalized)

        gender = str(item.get("gender") or "unknown").strip().lower()
        age = str(item.get("age") or "unknown").strip().lower()
        aliases = item.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        records.append({
            "name": name,
            "aliases": [str(a).strip() for a in aliases if str(a).strip()],
            "gender": gender if gender in GENDERS else "unknown",
            "age": age if age in AGES else "unknown",
            "description": str(item.get("description") or "").strip(),
        })

    return records


def _match_score(variant: str, existing: str) -> int:
    """Score how well a key-character spelling matches a character name.

    Names are CamelCase-normalized, so word boundaries survive as capitals
    ("ХорусЛуперсий" vs "Хорус" is a match, "Иванов" vs "Иван" is not).
    """
    if not variant or not existing:
        return 0
    if variant.lower() == existing.lower():
        return 3

    short, long_ = (variant, existing) if len(variant) <= len(existing) else (existing, variant)
    if len(short) < 3 or len(short) >= len(long_):
        return 0

    if long_.startswith(short):
        rest = long_[len(short):]
        if rest[:1].isupper():
            return 1
    if long_.endswith(short):
        head = long_[:-len(short)]
        if head[-1:].islower():
            return 1
    return 0


def merge_key_characters(characters: list, key_characters: list) -> list:
    """Return a copy of ``characters`` with key characters present.

    Existing characters are matched to key entries one-to-one (best score
    first) and enriched with the description, aliases and any missing gender or
    age.  Key characters BookNLP missed are appended.
    """
    result = [dict(char) for char in (characters or [])]
    if not key_characters:
        return result

    normalized = [
        _normalize_name(char.get("normalized_name", "")) for char in result
    ]

    candidates = []
    for order, entry in enumerate(key_characters):
        variants = [_normalize_name(entry.get("name", ""))]
        variants += [_normalize_name(alias) for alias in entry.get("aliases", [])]
        for position, existing in enumerate(normalized):
            score = max((_match_score(v, existing) for v in variants), default=0)
            if score:
                candidates.append((score, order, position))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    linked = {}
    claimed = set()
    for _score, order, position in candidates:
        if order in linked or position in claimed:
            continue
        linked[order] = position
        claimed.add(position)

    for order, entry in enumerate(key_characters):
        gender = entry.get("gender", "unknown")
        age = entry.get("age", "unknown")

        if order in linked:
            char = result[linked[order]]
        else:
            char = {
                "normalized_name": _normalize_name(entry.get("name", "")),
                "inferred_gender": "unknown",
                "inferred_age_category": "unknown",
            }
            result.append(char)

        if char.get("inferred_gender") in (None, "", "unknown") and gender in GENDERS:
            char["inferred_gender"] = gender
        if char.get("inferred_age_category") in (None, "", "unknown") and age in AGES:
            char["inferred_age_category"] = age

        char["key_character"] = True
        if entry.get("description"):
            char["description"] = entry["description"]
        if entry.get("aliases"):
            char["aliases"] = [str(a) for a in entry["aliases"]]

    return result
