from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.lang import legends

import hashlib
import warnings

#sys.stderr = StdoutFilter(sys.stdout)

class Silero(TTSUtils, TTSRegistry, name='silero'):

    # Silero synthesizes with the speakers baked into the checkpoint and cannot
    # clone a reference wav, so the per-character cast voice E2A hands over is
    # mapped onto one of them. The pools are split by the gender E2A already
    # encodes in the voice path (voices/<lang>/adult/<gender>/<Name>.wav), and a
    # voice is assigned round-robin per gender: the first character of a gender
    # gets the first speaker, the next one the next, and only a cast larger than
    # the pool reuses a speaker. 5 speakers of the v5_ru family in total.
    speaker_pools = {
        'male': ('aidar', 'eugene'),
        'female': ('baya', 'kseniya', 'xenia')
    }

    # The checkpoints carry an accentor of their own, but it is older than the
    # standalone silero-stress model, which also solves homographs — and the
    # checkpoints honour explicit '+' marks ('зам+ок' and 'замо+к' synthesize
    # differently), so the newer accentor gets the last word. Every plain text
    # fragment therefore goes through silero-stress before the synthesis; a
    # preset entry can set "stress": False to feed the raw text instead.
    stress_enabled = True

    def __init__(self, session:DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speaker = None
            self.tts_key = self.session['model_cache']
            self.resampler_cache = {}
            self.audio_segments = []
            self.accentor = None
            self.tts_engine = self.session.get('tts_engine')
            self.models = load_engine_presets(self.tts_engine)
            self.params = {
                'voice_speakers': {},
                'speaker_cursor': {gender: 0 for gender in self.speaker_pools}
            }
            # effective language for TTS (target when translating, else source)
            self.language = self.session.get('language')
            self.language_iso1 = self.session.get('language_iso1')
            if self.session.get('translate_enabled'):
                if self.session.get('translate'):
                    self.language = self.session['translate']
                if self.session.get('translate_iso1'):
                    self.language_iso1 = self.session['translate_iso1']
            if self.tts_engine not in default_engine_settings:
                error = legends['error_invalid_tts_engine'].format(engine=self.tts_engine)
                raise ValueError(error)
            engine_langs = default_engine_settings[self.tts_engine].get('languages', {})
            if self.language not in engine_langs:
                error = legends['error_language_not_supported_engine'].format(lang=self.language, engine=self.tts_engine)
                raise ValueError(error)
            fine_tuned = self.session.get('fine_tuned')
            if fine_tuned not in self.models:
                error = legends['error_invalid_fine_tuned'].format(model=fine_tuned, models=list(self.models.keys()))
                raise ValueError(error)
            model_cfg = self.models[fine_tuned]
            for required_key in ('repo', 'samplerate'):
                if required_key not in model_cfg:
                    error = legends['error_fine_tuned_missing_key'].format(model=fine_tuned, key=required_key)
                    raise ValueError(error)
            self.params['samplerate'] = model_cfg['samplerate']
            self.use_stress = bool(model_cfg.get('stress', self.stress_enabled))
            self.model_path = model_cfg['repo']
            # Silero's checkpoints are CPU TorchScript packages: a sentence takes
            # tens of milliseconds, so the GPU is left to the other engines.
            self.device = devices['CPU']['proc']
            # the checkpoints expose no temperature/top_k surface, so the
            # [emotion:...] tag reaches this engine only through the audio gain
            self.fine_tuned_params = {}
            self.engine = self.load_engine()
        except Exception as e:
            error = f'__init__() error: {e}'
            raise ValueError(error)

    def load_engine(self)->Any:
        try:
            msg = legends['msg_loading_tts_model'].format(model=self.tts_key)
            print(msg)
            self.cleanup_memory()
            engine = loaded_tts.get(self.tts_key)
            if not engine:
                try:
                    import torch
                    from torch import package
                except ImportError as e:
                    error = f'load_engine(): torch.package is required for {self.tts_engine}: {e}'
                    raise RuntimeError(error) from e
                model_file = os.path.join(tts_dir, self.tts_engine, os.path.basename(self.model_path))
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                if not os.path.isfile(model_file):
                    try:
                        torch.hub.download_url_to_file(self.model_path, model_file, progress=False)
                    except Exception as e:
                        error = f'load_engine(): cannot download {self.model_path}: {e}'
                        raise RuntimeError(error) from e
                try:
                    with warnings.catch_warnings():
                        # the packaged checkpoint source contains regex escapes that
                        # python flags on compile ('\^'); it is harmless noise
                        warnings.simplefilter('ignore', SyntaxWarning)
                        engine = package.PackageImporter(model_file).load_pickle('tts_models', 'model')
                except Exception as e:
                    error = f'load_engine(): failed to load {model_file}: {e}'
                    raise RuntimeError(error) from e
                loaded_tts[self.tts_key] = engine
            if engine:
                msg = legends['msg_tts_loaded'].format(model=self.tts_key)
                print(msg)
                return engine
            error = 'load_engine(): engine is None'
            raise RuntimeError(error)
        except Exception as e:
            error = f'load_engine() error: {e}'
            raise RuntimeError(error) from e

    def _speaker_for(self, voice:str)->str:
        speakers = self.params['voice_speakers']
        if voice in speakers:
            return speakers[voice]
        available = self._available_speakers()
        builtin = tuple(speaker for pool in self.speaker_pools.values() for speaker in pool)
        stem = Path(voice).stem.lower() if voice else ''
        if stem in builtin and (not available or stem in available):
            # a cast voice named after a builtin speaker selects that speaker
            speakers[voice] = stem
            return stem
        gender = self._voice_gender(voice)
        pool = tuple(speaker for speaker in self.speaker_pools[gender] if not available or speaker in available)
        if not pool:
            pool = available or self.speaker_pools[gender]
        cursor = self.params['speaker_cursor']
        speaker = pool[cursor[gender] % len(pool)]
        cursor[gender] += 1
        speakers[voice] = speaker
        return speaker

    def _available_speakers(self)->tuple:
        # not every checkpoint of the family carries all five voices (v5_4_ru
        # ships four), so the pools are intersected with the real speaker list
        return tuple(getattr(self.engine, 'speakers', None) or ())

    @staticmethod
    def _voice_gender(voice:str)->str:
        # E2A stores the cast voices as voices/<lang>/adult/<gender>/<Name>.wav;
        # anything else (a custom wav, a voice conversion reference) gets a
        # stable gender so that the same voice keeps the same speaker.
        parts = [part.lower() for part in Path(voice or '').parts[:-1]]
        for gender in ('female', 'male'):
            if gender in parts:
                return gender
        return 'female' if hashlib.md5((voice or '').encode('utf-8')).digest()[0] % 2 else 'male'

    @classmethod
    def _silence_follows(cls, sentence_parts:list[str], index:int)->bool:
        # True when the next thing that produces audio after sentence_parts[index] is an
        # explicit [break]/[pause] tag: those insert silence of their own, so the
        # punctuation pause must not be stacked on top of them.
        for nxt in sentence_parts[index + 1:]:
            m = SML_TAG_PATTERN.fullmatch(nxt)
            if m is None:
                # plain text is audible unless it is punctuation-only, which the render
                # loop skips for exactly the same reason (no alphanumeric characters)
                if any(c.isalnum() for c in nxt):
                    return False
                continue
            tag = m.group('tag')
            if tag in ('break', 'pause'):
                return True
            if tag not in ('voice', 'emotion'):
                return False
        return False

    def _load_accentor(self)->Any:
        # silero-stress is pure torch, but it is still an extra package: a missing
        # or broken install loses the accent marks and must not fail the sentence,
        # so the failure is remembered and the text is used as it came in.
        if self.accentor is None:
            try:
                from silero_stress import load_accentor
                self.accentor = load_accentor()
            except Exception as e:
                print(f'{self.tts_engine}: silero-stress unavailable, the text is synthesized without stress marks: {e}')
                self.accentor = False
        return self.accentor or None

    def _accent(self, text:str)->str:
        # accents the fragment ('замок' -> 'зам+ок') so that the checkpoint's
        # generator pronounces it as silero-stress decided; SML tags are never
        # passed here, only the plain text between them.
        if not self.use_stress:
            return text
        accentor = self._load_accentor()
        if accentor is None:
            return text
        try:
            accented = accentor(text)
        except Exception as e:
            self.log_exception(f'{self.__class__.__name__}._accent()', e)
            return text
        return accented or text

    def convert(self, sentence_file:str, sentence:str, **kwargs)->tuple:
        try:
            import torch
            import numpy as np
            from lib.classes.tts_engines.common.audio import trim_audio, is_audio_data_valid
            if self.engine:
                sentence_parts = self._split_sentence_on_sml(sentence)
                self.params['block_voice'] = kwargs.get('block_voice', self.session['voice'])
                if self.params.get('inline_voice'):
                    self.params['current_voice'] = self.params['inline_voice']
                else:
                    self.params['current_voice'], error = self._set_voice(self.params['block_voice'])
                    if self.params['current_voice'] is None and error is not None:
                        return False, error
                self.audio_segments = []
                # the last fragment that carries words (same alnum test as below): it must
                # leave a pause behind it, otherwise two sentences would be glued together
                # once the per-fragment padding is trimmed away
                last_audible = max(
                    (i for i, p in enumerate(sentence_parts) if any(c.isalnum() for c in p.strip())),
                    default=-1
                )
                for index, part in enumerate(sentence_parts):
                    part = part.strip()
                    if not part:
                        continue
                    if SML_TAG_PATTERN.fullmatch(part):
                        success, error = self._convert_sml(part)
                        if not success:
                            return False, error
                        continue
                    if not any(c.isalnum() for c in part):
                        continue
                    else:
                        if part.endswith("'"):
                            part = part[:-1]
                        part = self._accent(part)
                        part_sr = self.params['samplerate']
                        try:
                            with torch.inference_mode():
                                audio_part = self.engine.apply_tts(
                                    text=part,
                                    speaker=self._speaker_for(self.params['current_voice']),
                                    sample_rate=part_sr
                                )
                            if audio_part is not None and is_audio_data_valid(audio_part):
                                if part_sr and part_sr != self.params['samplerate']:
                                    audio_part = self._resample_audiodata(audio_part, part_sr, self.params['samplerate'])
                                part_tensor = self._tensor_type(audio_part)
                                # the checkpoint pads what it generates; inside a multi-voice
                                # sentence those paddings piled up with the [break]/[pause]/
                                # punctuation silences into seconds of dead air.
                                trimmed = trim_audio(part_tensor, self.params['samplerate'])
                                if trimmed.numel() == 0:
                                    trimmed = part_tensor
                                part_tensor = trimmed.unsqueeze(0)
                                if part_tensor.numel() == 0:
                                    error = 'part_tensor not valid'
                                    return False, error
                                emotion_gain = self.params.get('emotion_gain')
                                if emotion_gain and float(emotion_gain) != 1.0:
                                    part_tensor = part_tensor * float(emotion_gain)
                                self.audio_segments.append(part_tensor)
                                # a fragment that does not end on a word gets a pause of its own,
                                # but only when nothing else already pauses for it: stacking the
                                # punctuation pause on top of a [break]/[pause] (or on the last
                                # fragment, whose trailing padding is trimmed away) is what made
                                # the switches sound like long gaps.
                                if not self._silence_follows(sentence_parts, index) and (
                                    index == last_audible
                                    or (not re.search(r'\w$', part, flags=re.UNICODE) and part[-1] != '—')
                                ):
                                    silence_time = int(np.random.uniform(0.3, 0.6) * 100) / 100
                                    self.audio_segments.append(torch.zeros(1, int(self.params['samplerate'] * silence_time)))
                            else:
                                error = 'audio_part not valid'
                                return False, error
                        except ValueError as e:
                            # the checkpoint raises ValueError for a fragment it cannot
                            # pronounce at all -- digits, latin words, stray markup --
                            # and there is simply no audio to make: the fragment is
                            # skipped instead of aborting the whole book
                            snippet = part if len(part) <= 80 else f'{part[:80]}…'
                            print(f'{self.tts_engine}: skipped a fragment the checkpoint cannot pronounce: {snippet!r} ({e})')
                            continue
                        except IndexError as e:
                            self.cleanup_memory()
                            error = f'convert() error at {e} segment: {part}'
                            return False, error
                        except Exception as e:
                            # free the failed part's tensors before returning False;
                            # core.py then unloads the engine via unload_tts_manager().
                            self.cleanup_memory()
                            return False, self.log_exception(f'{self.__class__.__name__}.convert() part loop', e)
                if not self.audio_segments:
                    # every fragment was dropped (a sentence of digits, latin or
                    # punctuation only, or unpronounceable markup): the chapter
                    # combiner needs a file for this sentence index, so a short
                    # silence is written instead of leaving the index missing
                    snippet = sentence if len(sentence) <= 80 else f'{sentence[:80]}…'
                    print(f'{self.tts_engine}: nothing to pronounce in {snippet!r}, writing a short silence')
                    self.audio_segments.append(torch.zeros(1, max(1, int(self.params['samplerate'] * 0.3))))
                if self.audio_segments:
                    segment_tensor = torch.cat(self.audio_segments, dim=-1)
                    if not self.audio_save(sentence_file, segment_tensor, self.params['samplerate']):
                        error = f'audio_save() error: cannot save {sentence_file}'
                        return False, error
                    self.audio_segments = []
                    if not os.path.exists(sentence_file):
                        error = legends['error_cannot_create'].format(file=sentence_file)
                        return False, error
                return True, None
            else:
                error = legends['error_tts_engine_load_failed'].format(engine=self.session['tts_engine'])
                return False, error
        except Exception as e:
            self.cleanup_memory()
            self.audio_segments = []
            return False, self.log_exception(f'{self.__class__.__name__}.convert()',e)

    def create_vtt(self, all_sentences:list)->bool:
        if self._build_vtt_file(all_sentences):
            return True
        return False
