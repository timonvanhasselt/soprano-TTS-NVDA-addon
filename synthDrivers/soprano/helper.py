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
        
        # Parameters from the web version
        self.TOKEN_SIZE = 2048
        self.RECEPTIVE_FIELD = 4
        self.CHUNK_SIZE = 8 
        self.temperature = 0.3
        self.top_k = 50
        self.top_p = 0.95
        self.repetition_penalty = 1.2

    def sample(self, logits, seen_tokens):
        # Create a copy to avoid modifying the original logits
        logits = logits.copy()
        
        # 1. Repetition Penalty (as in JS: s < 0 ? s * penalty : s / penalty)
        if self.repetition_penalty != 1.0:
            for token_id in seen_tokens:
                if logits[token_id] < 0:
                    logits[token_id] *= self.repetition_penalty
                else:
                    logits[token_id] /= self.repetition_penalty

        # 2. Temperature scaling
        logits = logits / self.temperature

        # 3. Top-K filtering
        if self.top_k > 0:
            indices_to_remove = logits < np.sort(logits)[-self.top_k]
            logits[indices_to_remove] = -float('Inf')

        # 4. Top-P (Nucleus) filtering
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = stable_softmax(sorted_logits)
        cumulative_probs = np.cumsum(probs)

        # Remove tokens that exceed the cumulative threshold
        idx_to_remove = cumulative_probs > self.top_p
        # Shift mask so we keep at least one token (the first token that crosses the threshold)
        idx_to_remove[1:] = idx_to_remove[:-1].copy()
        idx_to_remove[0] = False

        masked_indices = sorted_indices[idx_to_remove]
        logits[masked_indices] = -float('Inf')

        # 5. Final sampling
        final_probs = stable_softmax(logits)
        return np.random.choice(len(final_probs), p=final_probs)

    def infer_stream(self, text):
        prompt = f"[STOP][TEXT]{text.strip()}[START]"
        encoding = self.tokenizer.encode(prompt)
        curr_input_ids = np.array([encoding.ids], dtype=np.int64)
        
        # Track seen tokens for repetition penalty (including prompt tokens)
        seen_tokens = set(encoding.ids)
        
        past_kv = {f"past_key_values.{i}.{t}": np.zeros((1, 1, 0, 128), dtype=np.float32) 
                   for i in range(17) for t in ["key", "value"]}
        h_buffer = []
        step = 0
        max_tokens = 512 

        while step < max_tokens:
            past_len = past_kv["past_key_values.0.key"].shape[2]
            pos_ids = np.array([[past_len]], dtype=np.int64) if step > 0 else np.arange(curr_input_ids.shape[1], dtype=np.int64).reshape(1, -1)
            attn_mask = np.ones((1, past_len + curr_input_ids.shape[1]), dtype=np.int64)
            
            outs = self.backbone.run(None, {
                "input_ids": curr_input_ids, 
                "attention_mask": attn_mask, 
                "position_ids": pos_ids, 
                **past_kv
            })
            
            for j in range(17):
                past_kv[f"past_key_values.{j}.key"] = outs[1 + j*2]
                past_kv[f"past_key_values.{j}.value"] = outs[2 + j*2]
            
            h_buffer.append(outs[-1][:, -1:, :])
            
            # Use the new sample method with parameters from the web version
            next_token_id = self.sample(outs[0][0, -1, :], seen_tokens)
            
            # Update seen tokens
            seen_tokens.add(int(next_token_id))
            
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
