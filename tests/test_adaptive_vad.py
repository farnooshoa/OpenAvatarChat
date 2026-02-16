# test_adaptive_vad.py
import numpy as np
from src.handlers.vad.enhanced.vad_handler_adaptive import AdaptiveVADHandler

config = {
    'min_silence_duration_ms': 500,
    'max_silence_duration_ms': 2000,
    'adaptive_timeout': True,
    'sample_rate': 16000,
}

vad = AdaptiveVADHandler(config)

# Test with long utterance (should capture completely)
print("Test: Long utterance with pauses")
# ... test code ...

# Check statistics
stats = vad.get_statistics()
print(f"Statistics: {stats}")