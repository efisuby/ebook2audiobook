import os
from lib.conf import voices_dir
from lib.conf_models import TTS_ENGINES, default_engine_settings

models = {
    # "internal" is the key default_fine_tuned points at, so the GUI/CLI
    # fine-tuned fallback resolves for this engine as well.
    "internal": {
        "lang": "multi",
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['QWEN3TTS']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['QWEN3TTS']]['samplerate']
    },
    "0.6B": {
        "lang": "multi",
        "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['QWEN3TTS']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['QWEN3TTS']]['samplerate']
    }
}
