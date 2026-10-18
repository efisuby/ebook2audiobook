# AGENTS.md — ebook2audiobook (fork notes)

This checkout is a fork of
[DrewThomasson/ebook2audiobook](https://github.com/DrewThomasson/ebook2audiobook)
published at <https://github.com/efisuby/ebook2audiobook>. Everything from
upstream applies; this file only records the changes made in this fork so the
next agent does not have to rediscover them.

## Fork change: the `[emotion:LABEL]` SML tag

The companion project [E2A-SML](https://github.com/efisuby/E2A-SML) labels the
emotion of every spoken line and emits

```
[voice:…][emotion:LABEL]text[/emotion][/voice]
```

Upstream E2A does not know the tag, so `SML_TAG_PATTERN` (built from `TTS_SML`
keys) would not match it and it would be read aloud. The tag is therefore added
to the engine grammar.

### Files touched

| File | Change |
|------|--------|
| `lib/conf_models.py` | `TTS_SML["emotion"] = {"paired": True}` — the single source of truth for `sml_tag_keys`, `SML_TAG_PATTERN`, the translation-protection list and `normalize_sml_tags`. Also `TTS_SML_DEFAULT_EMOTION = "neutral"` and `TTS_SML_EMOTIONS` (per-label `temperature` / `top_p` / `speed` / `gain`). |
| `lib/classes/tts_engines/common/headers.py` | Export `TTS_SML_EMOTIONS`, `TTS_SML_DEFAULT_EMOTION` (so every engine sees them via `from ...common.headers import *`). |
| `lib/classes/tts_engines/common/utils.py` | `_convert_sml` gained an `elif tag == 'emotion':` branch: on open it snapshots `self.fine_tuned_params`, overrides only the keys already present, and stores `self.params["emotion_gain"]`; on close it restores both. A stack (`self._sml_emotion_stack`) makes nested tags safe. |
| `lib/classes/tts_engines/xtts.py` | In `convert()`, multiply the finished segment by `self.params.get("emotion_gain")` before `self.audio_segments.append(part_tensor)`. |
| `README.md`, `readme/README_rus.md` | Document `[emotion:LABEL]…[/emotion]` (labels, `neutral` no-op, gain on engines without tuned params). |

### Design notes for future edits

- **Adding any new SML tag means adding a `TTS_SML` entry first.** Everything
  else is derived from it; a tag missing from `TTS_SML` silently leaks into the
  spoken text.
- `TTS_SML_EMOTIONS` values are applied only to keys already present in
  `fine_tuned_params` (XTTS and Bark), so an engine without them gets a no-op —
  no crash.
- `neutral` is a no-op and is normally omitted by the writer.
- The `ipa` branch of `_convert_sml` is dead code — do not copy it.
- The companion E2A-SML project is the only writer of this tag; the SML file is
  still valid for stock E2A except for the unknown-tag behaviour above.

### How to check

E2A is normally run through Docker, so there is no ready local venv. Verify an
edit by parsing it and, when a subset of the deps is available, importing the
modules in isolation:

```bash
python -c "import ast; [ast.parse(open(p).read()) for p in ('lib/conf_models.py', 'lib/classes/tts_engines/common/utils.py', 'lib/classes/tts_engines/xtts.py')]"
python -c "from lib.conf_models import TTS_SML, TTS_SML_EMOTIONS; print(list(TTS_SML), sorted(TTS_SML_EMOTIONS))"
```
