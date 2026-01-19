import os
import threading
import queue
import ctypes
import numpy as np
import languageHandler
from collections import OrderedDict
from nvwave import WavePlayer, AudioPurpose
from logHandler import log
from synthDriverHandler import SynthDriver as BaseSynthDriver, VoiceInfo, synthIndexReached, synthDoneSpeaking
from speech.commands import IndexCommand, VolumeCommand, BreakCommand

try:
    from .helper import SopranoEngine
except ImportError as e:
    log.error(f"Soprano TTS: Could not load helper: {e}")

class _SynthQueueThread(threading.Thread):
    def __init__(self, driver):
        super().__init__()
        self.driver = driver
        self.daemon = True
        self.stop_event = threading.Event()
        self.cancel_event = threading.Event()

    def run(self):
        ctypes.windll.ole32.CoInitialize(None)
        while not self.stop_event.is_set():
            try:
                # Keep a small timeout to remain responsive
                request = self.driver._request_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self.cancel_event.clear()
            text, index = request
            
            try:
                if text and text.strip() and self.driver.tts_engine:
                    for chunk in self.driver.tts_engine.infer_stream(text):
                        if self.cancel_event.is_set():
                            break
                        if chunk is not None and self.driver._player:
                            volume_factor = (self.driver._volume / 100.0)
                            audio_int16 = (np.clip(chunk * volume_factor, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                            self.driver._player.feed(audio_int16)
                
                if index is not None and not self.cancel_event.is_set():
                    synthIndexReached.notify(synth=self.driver, index=index)
                
            except Exception as e:
                log.error(f"Soprano TTS: Error: {e}")
            finally:
                self.driver._request_queue.task_done()
                
                # Check if this was the last part of a sequence
                # We do NOT sync here to prevent blocking the intake of new speak() calls
                if self.driver._request_queue.empty():
                    synthDoneSpeaking.notify(synth=self.driver)
        
        ctypes.windll.ole32.CoUninitialize()

class SynthDriver(BaseSynthDriver):
    name = "soprano"
    description = "Soprano AI TTS (Native ONNX)"

    @classmethod
    def check(cls):
        return True

    supportedCommands = frozenset([IndexCommand, VolumeCommand, BreakCommand])
    supportedNotifications = frozenset([synthIndexReached, synthDoneSpeaking])
    supportedSettings = (BaseSynthDriver.VolumeSetting(),)

    def __init__(self):
        self._volume = 100
        self.tts_engine = None
        self._player = None
        self._request_queue = queue.Queue()
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        
        threading.Thread(target=self._initialize_async, daemon=True).start()
        self._worker_thread = _SynthQueueThread(self)
        self._worker_thread.start()

    @property
    def language(self):
        """Sets the voice to English by default as the model currently only supports English."""
        return "en"

    def languageIsSupported(self, lang):
        """Ensures NVDA does not stop speaking when language changes occur in windows."""
        return True

    def _initialize_async(self):
        ctypes.windll.ole32.CoInitialize(None)
        try:
            self.tts_engine = SopranoEngine(self.model_dir)
            self._player = WavePlayer(channels=1, samplesPerSec=32000, bitsPerSample=16, purpose=AudioPurpose.SPEECH)
            log.info("Soprano TTS: Initialized.")
        except Exception as e:
            log.error(f"Soprano TTS: Load failed: {e}")

    def speak(self, speechSequence):
        text_parts = []
        last_index = None
        
        for item in speechSequence:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, IndexCommand):
                # When an index comes in, we must ensure the text before it is queued
                if text_parts:
                    self._request_queue.put(("".join(text_parts), None))
                    text_parts = []
                last_index = item.index
        
        combined_text = "".join(text_parts).strip()
        if combined_text or last_index is not None:
            # We send the text block as NVDA intended. 
            # Internal splitting is removed here to prevent "Read All" from breaking.
            self._request_queue.put((combined_text, last_index))

    def cancel(self):
        self._worker_thread.cancel_event.set()
        if self._player:
            self._player.stop()
        while not self._request_queue.empty():
            try: self._request_queue.get_nowait()
            except queue.Empty: break

    def terminate(self):
        self._worker_thread.stop_event.set()
        if self._player:
            self._player.close()

    def _get_volume(self): return self._volume
    def _set_volume(self, value): self._volume = value
    def _get_availableVoices(self): return OrderedDict([("default", VoiceInfo("default", "Soprano AI"))])
    def _get_voice(self): return "default"
    def _set_voice(self, value): pass
