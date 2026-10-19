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

## Fork change: the Qwen3-TTS engine

Adds a `qwen3tts` engine (`lib/classes/tts_engines/qwen3tts.py`,
`class Qwen3TTS(TTSUtils, TTSRegistry, name='qwen3tts')`). It mirrors `xtts.py`
and uses the **Base** voice-clone models, so the per-character reference `.wav`
cast by E2A becomes the clone reference — x-vector-only, i.e. no transcript is
needed. Output is 24 kHz mono.

### Files touched

| File | Change |
|------|--------|
| `lib/conf_models.py` | `TTS_ENGINES["QWEN3TTS"] = "qwen3tts"` and a `default_engine_settings` entry (same 17-language map as XTTS, `samplerate: 24000`, `files: []`, `rating`). |
| `lib/classes/tts_engines/presets/qwen3tts_presets.py` | `models` with `"internal"` (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`) and `"0.6B"` (`…-0.6B-Base`) presets, both `lang: "multi"`. |
| `lib/classes/tts_engines/qwen3tts.py` | New engine. `load_engine()` builds `Qwen3TTSModel.from_pretrained(..., device_map, dtype, cache_dir)`; `_voice_clone_prompt()` caches `create_voice_clone_prompt(ref_audio=…, x_vector_only_mode=True)` per voice; `convert()` splits on SML, honours `[voice:…]`/`[emotion:…]` and calls `generate_voice_clone(text, language, voice_clone_prompt)`. |
| `lib/classes/tts_engines/__init__.py` | `from .qwen3tts import Qwen3TTS` (this is what registers the engine). |
| `requirements.txt` | `qwen-tts` plus its runtime deps (`accelerate`, `librosa`, `einops`, `sox`). |
| `lib/classes/device_installer.py` | `no_deps_pkgs = ['qwen-tts']` (installed with `--no-deps`); `select_pkg('transformers')` returns the global pin `transformers==4.57.3` and `pip_overrides` forces `huggingface-hub<1.0` through a uv `--overrides` file. |
| `README.md`, `readme/README_rus.md` | `Qwen3-TTS` ticked in the engines list + a short usage note. |

### Design notes for future edits

- **The preset key `internal` is mandatory**: `default_fine_tuned = 'internal'`,
  `_update_gr_fine_tuned_list` falls back to it, and `_restore_interface` reads
  `models[default_fine_tuned]['files']`.
- No `qwen3tts_*` session keys were added (unlike `xtts_*`), so
  `self.fine_tuned_params = {}`. The `[emotion:LABEL]` tag therefore reaches
  this engine only through `self.params["emotion_gain"]` (applied in
  `convert()`), and the `[emotion:…]` branch in `_convert_sml` is a safe no-op
  for the tuned params.
- `qwen-tts` targets an old `transformers`; it is installed with `--no-deps` and
  its runtime dependencies must stay listed in `requirements.txt`. The runtime is
  held at `transformers==4.57.3` for it (see the next section), not at 5.x.
- No Gradio param tab is needed: the engine shows up in the dropdown through
  `get_compatible_tts_engines(language)`, derived from the `languages` map in
  `default_engine_settings`. `visible_xtts` / `visible_bark` stay `False` for it.
- `attn_implementation="flash_attention_2"` is deliberately **not** passed, to
  avoid a hard flash-attn dependency.
- `load_engine()` installs a logging filter that drops transformers' per-call
  "Setting `pad_token_id` to `eos_token_id`… for open-end generation" warning: the id
  it fills in lives in a nested Qwen3-TTS generation config and passing `pad_token_id`
  through `generate_voice_clone` does not reach it, so it would print once per
  sentence. It reports correct behaviour (pad = eos is what transformers picks).
  Only that record is dropped; other warnings from the same logger still print.
- **The engine's only spacing rules are the SML ones.** Qwen3-TTS pads every fragment
  it generates with leading and trailing silence (measured 200–500 ms per side), and a
  sentence is one fragment per `[voice:…]`/`[/voice]` run. Untouched, those paddings used
  to pile up with the tag silences and with the punctuation pause into seconds of dead
  air between two speakers. So `convert()`:
  - trims each generated fragment with `trim_audio()` (falling back to the raw audio if
    nothing survives the threshold, instead of failing the sentence),
  - appends the 0.3–0.6 s punctuation pause only when nothing else already pauses there —
    `_silence_follows()` looks past structural tags (`voice`, `emotion`) and past
    punctuation-only fragments to see whether a `[break]`/`[pause]` is coming — and
    always on the last fragment that carries words (`last_audible`), otherwise two
    sentences would be glued together now that the trailing padding is gone.
  Nothing else spaces the fragments out: in particular a `[voice:…]` switch inserts no
  pause of its own — spacing at a speaker change comes from the tags the SML writer put
  there.
- `unload_tts_manager()` needs no `engine.unload()`; it drops `loaded_tts[key]`
  and frees the GPU.

### How to check

```bash
python -c "import ast; [ast.parse(open(p).read()) for p in ('lib/conf_models.py', 'lib/classes/tts_engines/qwen3tts.py', 'lib/classes/tts_engines/presets/qwen3tts_presets.py', 'lib/classes/tts_engines/__init__.py', 'lib/classes/device_installer.py')]"
```

A live run needs `pip install --no-deps qwen-tts accelerate librosa einops sox`
and the `Qwen/Qwen3-TTS-12Hz-1.7B-Base` weights (~4.5 GB).

## Fork change: the Silero TTS engine

Adds a `silero` engine (`lib/classes/tts_engines/silero.py`,
`class Silero(TTSUtils, TTSRegistry, name='silero')`) for **Russian only**. Silero
cannot clone a voice: the per-character reference file cast by E2A only picks one of
the model's five built-in speakers. It runs on CPU (a sentence costs ~50–100 ms), so
the GPU is left to the other engines. Output is 24 kHz mono.

### Files touched

| File | Change |
|------|--------|
| `lib/conf_models.py` | `TTS_ENGINES["SILERO"] = "silero"` and a `default_engine_settings` entry: one language (`{"rus": "ru"}`), `samplerate: 24000`, `files: []`, `voices: {}`, `rating {"VRAM": 0, "CPU": 5, "RAM": 2, "Realism": 4}`. |
| `lib/classes/tts_engines/presets/silero_presets.py` | `models = {"internal": {...}}` only — `repo` is the `v5_6_ru.pt` URL on `models.silero.ai`. |
| `lib/classes/tts_engines/silero.py` | New engine. `load_engine()` downloads the `.pt` with `torch.hub.download_url_to_file` and unpacks it with `torch.package.PackageImporter(...).load_pickle('tts_models', 'model')`; `_speaker_for()` maps a voice file to a built-in speaker; `convert()` copies `qwen3tts.convert()`. |
| `lib/classes/tts_engines/__init__.py` | `from .silero import Silero` (this is what registers the engine). |
| `README.md`, `readme/README_rus.md` | `Silero` in the engines list + a usage note (licence, download, CPU). |

`requirements.txt` is untouched: the checkpoint is a torch-package file loaded straight
from disk, so nothing but `torch` is needed. The `silero` PyPI package is deliberately
not used — it only wraps the same download.

### Design notes for future edits

- **Only Russian.** The settings carry a single `{"rus": "ru"}` and the preset points at
  `…/tts/ru/v5_6_ru.pt`. Silero's other families (CIS/Turkic, the older `v4_ru`, the jit
  models) live at other URLs and some need the `torch.jit` path instead of
  `torch.package` — that is where to extend.
- **`v5_6_ru` is the best Russian model** (auto-stress, homographs, questions, emphasis).
  It has exactly five speakers — `aidar`, `baya`, `kseniya`, `eugene`, `xenia` — and does
  **not** accept `speaker='random'` (v3/v4 did).
- **Voice mapping** (`_speaker_for`): a voice file whose stem is a speaker name is used
  as-is; otherwise the gender is read from the directory (`…/adult/male/…` vs
  `…/adult/female/…`, female checked first, deterministic md5 fallback when neither is
  there) and the speaker is handed out round-robin from `speaker_pools` (male
  `aidar`/`eugene`, female `baya`/`kseniya`/`xenia`), cached in `params['voice_speakers']`
  so a narrator keeps one speaker for the whole book.
- Spacing and `[emotion:…]` behaviour are copied from `qwen3tts`: `convert()` uses the
  same `trim_audio()` / `_silence_follows()` / `last_audible` rules (so no pause comes
  from a `[voice:…]` switch), and `fine_tuned_params = {}` means the emotion tag reaches
  this engine only as `params['emotion_gain']`.
- `samplerate: 24000`, not the model's 48 kHz: at 48 kHz the output peaks above 1.0
  (measured 1.043) while the final ffmpeg pass writes `-sample_fmt s16`, so it would
  clip. All three rates produce the same duration (it re-renders, not upsamples).
- The `.pt` source contains `'\^'`, so python emits
  `SyntaxWarning: invalid escape sequence` while compiling it; `load_engine()` swallows
  exactly that category.
- Not in `tts_engines_with_inner_speaker`, `tts_engines_from_coqui` or
  `tts_engines_with_custom_model`: the GUI keeps the XTTS-like voice flow (voices from
  files), because the speaker choice happens inside the engine.
- `apply_tts()` exposes no `speed` and no `ssml` switch (SSML goes through `ssml_text=`),
  but it does take `voice_path=`, which is the extension point if real cloning is wanted.
- **Licence:** the weights are CC BY-NC-SA 4.0 (non-commercial); the package code is MIT.

### How to check

```bash
python -c "import ast; [ast.parse(open(p).read()) for p in ('lib/conf_models.py', 'lib/classes/tts_engines/silero.py', 'lib/classes/tts_engines/presets/silero_presets.py', 'lib/classes/tts_engines/__init__.py')]"
```

A live run downloads ~148 MB from `models.silero.ai` into `models/tts/silero/`; both the
engine smoke test and a headless `--tts_engine silero --device cpu` run pass on torch
2.14.1+cu130.

## Fork change: transformers pinned to 4.57.3

`qwen-tts` does not run on transformers 5.x: 5.x turned `@check_model_inputs()` into
`check_model_inputs(func)`, and removed the `PretrainedConfig` defaults the qwen
configs read (`pad_token_id`). Shimming the decorator only moves the failure one
step further, so the whole runtime is pinned instead.

| Pin | Where | Why |
|-----|-------|-----|
| `transformers==4.57.3` | `device_installer.select_pkg('transformers')` (it is in `device_pkgs`, so the pin rides along on every install call) | the only line `qwen-tts` supports |
| `tokenizers==0.22.2` | resolved automatically as a dependency | 4.57.3 aborts at import above `tokenizers==0.23.0`; the 0.23.0 release does not exist (0.23.0rc0, then 0.23.1…), so 0.22.2 is the newest allowed |
| `huggingface-hub<1.0` (0.36.2) | `device_installer.pip_overrides`, written to a uv `--overrides` file in `pip_cache_dir` | 4.57.3 declares `hub<1.0` (it imports `HTTPError` from `huggingface_hub.utils` and expects hub errors to subclass `requests.HTTPError` — both gone in hub 1.x/2.x), while `gradio==6.29.1` declares `hub>=1.16.0`. The resolver cannot satisfy both, so the override forces the older hub. gradio 6.29.1 imports, starts and runs on 0.36.2 (verified); E2A itself only uses hub via `hf_hub_download` / `snapshot_download` / `get_hf_file_metadata` / `hf_hub_url`, all present in both lines. |

Consequences to keep in mind:

- The transformers 5.x patches in `components/sitecustomize.py` are inert on this pin
  but must stay: they repair torch<2.6 `check_torch_load_is_safe`, the coqui-tts
  `isin_mps_friendly` import and the GPT2InferenceModel all-1s attention_mask.
- `--overrides` is only added to `uv pip install` (see `_uv_pip`), never to `uninstall`.
- Verified end to end on this stack (torch 2.14.1+cu130, CUDA): a headless `qwen3tts`
  run, a headless `xtts` run, and the qwen3tts engine smoke test (5 cases, with the
  `[emotion:…]` gain applied).
- After any transformers or gradio bump re-check both engines: transformers must stay
  below 5 unless `qwen-tts` ships a 5.x-capable release, and gradio must stay on a
  version that runs with `huggingface-hub<1.0`.
