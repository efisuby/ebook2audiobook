import os
from lib.conf import voices_dir
from lib.conf_models import TTS_ENGINES, default_engine_settings

# Silero's russian v5 checkpoints live on models.silero.ai as torch.package
# archives (not on the Hugging Face hub): the engine downloads the url below
# into tts_dir/silero/ on first use. v5_6_ru is the newest of the family and
# the only one with automatic stress, homographs, question and emphasis
# intonation; it carries the 5 speakers the engine maps cast voices onto.
models = {
    # "internal" is the key default_fine_tuned points at, so the GUI/CLI
    # fine-tuned fallback resolves for this engine as well.
    "internal": {
        "lang": "ru",
        "repo": "https://models.silero.ai/models/tts/ru/v5_6_ru.pt",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['SILERO']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['SILERO']]['samplerate']
    }
}
