import os
import sys
import inspect

import torch
import torch.onnx

from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT as OriginalGPT, GPTConfig as OriginalGPTConfig

class LilleConfig(PretrainedConfig):
    model_type = "lille-130m"
    def __init__(
        self,
        block_size: int = 512,
        vocab_size: int = 32768,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        n_kv_heads: int | None = 4,
        rope_theta: float = 10000.0,
        **kwargs
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.n_kv_heads = n_kv_heads
        self.rope_theta = rope_theta
        self.tie_word_embeddings = True
        super().__init__(**kwargs)

class LilleForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LilleConfig
    
    @property
    def main_input_name(self):
        return "input_ids"

    def __init__(self, config: LilleConfig):
        super().__init__(config)
        hf_config_dict = config.to_dict()
        if 'n_layer' in hf_config_dict:
            hf_config_dict['n_layers'] = hf_config_dict.pop('n_layer')
        if 'n_head' in hf_config_dict:
            hf_config_dict['n_heads'] = hf_config_dict.pop('n_head')
        
        expected_keys = inspect.signature(OriginalGPTConfig).parameters.keys()
        filtered_args = {key: hf_config_dict[key] for key in expected_keys if key in hf_config_dict}
        original_config = OriginalGPTConfig(**filtered_args)
        
        self.transformer = OriginalGPT(original_config)
        self.post_init()
        
        if self.config.tie_word_embeddings:
            self._tied_weights_keys = ["transformer.lm_head.weight"]
        
    def get_input_embeddings(self):
        return self.transformer.tok_embeddings
        
    def set_input_embeddings(self, new_embeddings):
        self.transformer.tok_embeddings = new_embeddings
        
    def get_output_embeddings(self):
        return self.transformer.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        past_key_values: list[torch.Tensor] | None = None,
        use_cache: bool | None = None,
        **kwargs
    ):
        past_kv_cache_for_model = None
        if past_key_values is not None:
            past_kv_cache_for_model = past_key_values
        
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            
            if past_kv_cache_for_model is not None and len(past_kv_cache_for_model) > 0:
                past_seq_len = past_kv_cache_for_model[0][0].shape[2]
                current_seq_len = input_ids.shape[1]
                total_seq_len = past_seq_len + current_seq_len
                
                if attention_mask.shape[1] < total_seq_len:
                    batch_size = attention_mask.shape[0]
                    cached_mask = torch.ones((batch_size, past_seq_len), 
                                            dtype=attention_mask.dtype, 
                                            device=attention_mask.device)
                    attention_mask = torch.cat([cached_mask, attention_mask], dim=1)
        
        outputs = self.transformer(
            input_ids,
            past_kv_cache=past_kv_cache_for_model,
            use_cache=use_cache,
            attn_mask=attention_mask
        )
        
        if use_cache:
            logits, present_kv_cache = outputs
            new_past_kv = tuple(present_kv_cache) if present_kv_cache else None
        else:
            logits = outputs[0]
            new_past_kv = None
        
        if torch.onnx.is_in_onnx_export():
            if new_past_kv is not None:
                flat_present = [t for pair in new_past_kv for t in pair]
                return (logits, *flat_present)
            else:
                return (logits,)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=new_past_kv
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
        
        return model_inputs
