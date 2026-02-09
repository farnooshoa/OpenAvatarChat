from pathlib import Path
from loguru import logger
import torch
import numpy as np

from chat_engine.common.handler_base import (
    HandlerBase,
    HandlerBaseInfo,
    HandlerDetail
)
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_engine_config_data import (
    ChatEngineConfigModel,
    HandlerBaseConfigModel
)
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType


# -----------------------------
# Per-session context
# -----------------------------

class AdaptiveVADContext(HandlerContext):
    def __init__(self):
        super().__init__()
        self.noise_floor = 0.0
        self.speech_history = []
        self.is_speaking = False
        self.speech_start_sample = 0
        self.speech_end_sample = 0
        self.buffer = []


# -----------------------------
# Handler
# -----------------------------

class AdaptiveVADHandler(HandlerBase):

    def __init__(self):
        super().__init__()
        self.model = None

    # ---- Required metadata ----

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="adaptive_vad",
            config_model=HandlerBaseConfigModel,
            load_priority=10,
        )

    # ---- Global init (once) ----

    def load(self,
             engine_config: ChatEngineConfigModel,
             handler_config: HandlerBaseConfigModel | None = None):

        cfg = handler_config.dict() if handler_config else {}

        # Configuration
        self.speaking_threshold = cfg.get('speaking_threshold', 0.5)
        self.start_delay = cfg.get('start_delay', 2048)
        self.end_delay = cfg.get('end_delay', 2048)
        self.buffer_look_back = cfg.get('buffer_look_back', 1024)
        self.speech_padding = cfg.get('speech_padding', 512)

        # Enhanced features
        self.adaptive_threshold = cfg.get('adaptive_threshold', True)
        self.noise_gate_db = cfg.get('noise_gate_db', -40)
        self.dynamic_sensitivity = cfg.get('dynamic_sensitivity', True)

        # Adaptive parameters
        self.adaptation_rate = cfg.get('adaptation_rate', 0.01)

        # Load Silero VAD model (once)
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.model.eval()
            logger.info('Silero VAD model loaded successfully')
        except Exception as e:
            logger.error(f'Failed to load Silero VAD: {e}')
            raise

    # ---- Session lifecycle ----

    def create_context(self,
                       session_context: SessionContext,
                       handler_config: HandlerBaseConfigModel | None = None
                       ) -> HandlerContext:
        return AdaptiveVADContext()

    def start_context(self,
                      session_context: SessionContext,
                      handler_context: HandlerContext):
        pass

    def destroy_context(self, context: HandlerContext):
        context.buffer.clear()
        context.speech_history.clear()

    # ---- Data contract ----

    def get_handler_detail(self,
                           session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        return HandlerDetail(
            inputs={
                ChatDataType.AUDIO: None
            },
            outputs={}
        )

    # ---- Core logic ----

    def handle(self,
               context: AdaptiveVADContext,
               inputs: ChatData,
               output_definitions):

        audio_chunk = inputs.data
        sample_rate = getattr(inputs, "sample_rate", 16000)

        if audio_chunk is None or len(audio_chunk) == 0:
            return None

        # Audio level
        audio_level = float(np.abs(audio_chunk).mean())

        # Update noise floor
        if self.adaptive_threshold:
            context.noise_floor = (
                (1 - self.adaptation_rate) * context.noise_floor
                + self.adaptation_rate * audio_level
            )

        # Noise gate
        noise_gate_threshold = 10 ** (self.noise_gate_db / 20)
        if audio_level < noise_gate_threshold:
            return None

        # Adaptive threshold
        if self.adaptive_threshold:
            adjusted_threshold = self.speaking_threshold + (context.noise_floor * 0.3)
            adjusted_threshold = max(0.3, min(0.8, adjusted_threshold))
        else:
            adjusted_threshold = self.speaking_threshold

        # Run Silero VAD
        try:
            audio_tensor = torch.from_numpy(audio_chunk).float()
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, sample_rate).item()
        except Exception as e:
            logger.error(f'VAD inference error: {e}')
            return None

        # Dynamic sensitivity
        if self.dynamic_sensitivity:
            context.speech_history.append(speech_prob)
            if len(context.speech_history) > 10:
                context.speech_history.pop(0)

            recent_speech = sum(p > 0.5 for p in context.speech_history) > 3
            if recent_speech:
                adjusted_threshold *= 0.9

        is_speech = speech_prob > adjusted_threshold
        context.is_speaking = is_speech

        logger.debug(
            f'VAD prob={speech_prob:.3f} '
            f'threshold={adjusted_threshold:.3f} '
            f'noise_floor={context.noise_floor:.3f} '
            f'speech={is_speech}'
        )

        return None
