import os
import sys
import numpy as np
import onnxruntime as ort
from logHandler import log

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
LIBS_PATH = os.path.join(ADDON_ROOT, "libs")

if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

try:
    from tokenizers import Tokenizer
except ImportError as e:
    log.error(f"Soprano: Could not load tokenizers. Error: {e}")

def stable_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class SopranoEngine:
    def __init__(self, model_dir):
        backbone_path = os.path.join(model_dir, "soprano_backbone_kv_int8.onnx")
        decoder_path = os.path.join(model_dir, "soprano_decoder_int8.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.backbone = ort.InferenceSession(backbone_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.decoder = ort.InferenceSession(decoder_path, sess_options=opts, providers=['CPUExecutionProvider'])
        tokenizer_json = os.path.join(model_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_json)
        self.TOKEN_SIZE = 2048
        self.RECEPTIVE_FIELD = 4
        self.CHUNK_SIZE = 12 

    def sample_top_p(self, logits, threshold=0.95, temperature=0.7):
        logits = logits / temperature
        probs = stable_softmax(logits)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        idx_to_remove = cumulative_probs > threshold
        idx_to_remove[1:] = idx_to_remove[:-1].copy()
        idx_to_remove[0] = False
        sorted_probs[idx_to_remove] = 0
        if np.sum(sorted_probs) > 0:
            sorted_probs /= np.sum(sorted_probs)
        else:
            sorted_probs = np.ones(len(sorted_probs)) / len(sorted_probs)
        return np.random.choice(sorted_indices, p=sorted_probs)

    def infer_stream(self, text):
        prompt = f"[STOP][TEXT]{text.strip()}[START]"
        encoding = self.tokenizer.encode(prompt)
        curr_input_ids = np.array([encoding.ids], dtype=np.int64)
        past_kv = {f"past_key_values.{i}.{t}": np.zeros((1, 1, 0, 128), dtype=np.float32) 
                   for i in range(17) for t in ["key", "value"]}
        h_buffer = []
        step = 0
        max_tokens = 512 

        while step < max_tokens:
            past_len = past_kv["past_key_values.0.key"].shape[2]
            pos_ids = np.array([[past_len]], dtype=np.int64) if step > 0 else np.arange(curr_input_ids.shape[1], dtype=np.int64).reshape(1, -1)
            attn_mask = np.ones((1, past_len + curr_input_ids.shape[1]), dtype=np.int64)
            outs = self.backbone.run(None, {"input_ids": curr_input_ids, "attention_mask": attn_mask, "position_ids": pos_ids, **past_kv})
            for j in range(17):
                past_kv[f"past_key_values.{j}.key"] = outs[1 + j*2]
                past_kv[f"past_key_values.{j}.value"] = outs[2 + j*2]
            h_buffer.append(outs[-1][:, -1:, :])
            next_token_id = self.sample_top_p(outs[0][0, -1, :])
            curr_input_ids = np.array([[next_token_id]], dtype=np.int64)
            is_eos = (next_token_id == 3) or (step == max_tokens - 1)
            if len(h_buffer) >= (self.RECEPTIVE_FIELD + self.CHUNK_SIZE) or is_eos:
                if not h_buffer: break
                batch_h = np.concatenate(h_buffer, axis=1)
                inp = np.transpose(batch_h, (0, 2, 1)).astype(np.float32)
                audio = self.decoder.run(None, {self.decoder.get_inputs()[0].name: inp})[0].squeeze()
                if is_eos:
                    offset = len(h_buffer) * self.TOKEN_SIZE
                    yield audio[-int(offset):]
                    break
                else:
                    start_idx = (self.RECEPTIVE_FIELD + self.CHUNK_SIZE - 1) * self.TOKEN_SIZE
                    end_idx = (self.RECEPTIVE_FIELD - 1) * self.TOKEN_SIZE
                    yield audio[-int(start_idx):-int(end_idx)]
                    h_buffer = h_buffer[-self.RECEPTIVE_FIELD:]
            step += 1