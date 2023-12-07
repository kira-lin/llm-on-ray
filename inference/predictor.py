import re
import torch

def max_input_len(input_text_length):
    if input_text_length <= 128:
        return 128
    elif input_text_length <= 512:
        return 512
    elif input_text_length <= 2048:
        return 2048
    else:
        print("Max support length is 4096")
        return 4096

class Predictor:
    def tokenize_inputs(self, text):
        if self.device.type == "hpu":
            input_tokens_no_pad = self.tokenizer(text, return_tensors="pt")
            input_token_len = input_tokens_no_pad.input_ids.shape[-1]
            input_tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=max_input_len(input_token_len),
            )
        else:
            input_tokens = self.tokenizer(
                text, return_tensors="pt", padding=True
            )
            input_token_len = input_tokens.input_ids.shape[-1]
        input_ids = input_tokens.input_ids.to(device=self.device)
        return input_ids, input_token_len

    def configure_tokenizer(self, model_name):
        model = self.model
        tokenizer = self.tokenizer
        if re.search("llama", model.config.architectures[0], re.IGNORECASE):
            # unwind broken decapoda-research config
            model.generation_config.pad_token_id = 0
            model.generation_config.bos_token_id = 1
            model.generation_config.eos_token_id = 2

        if (
            hasattr(model.generation_config, "pad_token_id")
            and model.generation_config.pad_token_id is not None
            and not "chatglm" in model_name
        ):
            tokenizer.pad_token_id = model.generation_config.pad_token_id
        if (
            hasattr(model.generation_config, "eos_token_id")
            and model.generation_config.eos_token_id is not None
            and not "chatglm" in model_name
        ):
            tokenizer.eos_token_id = model.generation_config.eos_token_id
        if (
            hasattr(model.generation_config, "bos_token_id")
            and model.generation_config.bos_token_id is not None
        ):
            tokenizer.bos_token_id = model.generation_config.bos_token_id

        if tokenizer.pad_token_id is None:
            model.generation_config.pad_token_id = (
                tokenizer.pad_token_id
            ) = tokenizer.eos_token_id

        if model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        
        if not model.config.is_encoder_decoder:
            tokenizer.padding_side = "left"

        if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    def generate(self, inputs, **config):
        pass

    def streaming_generate(self, inputs, streamer, **config):
        pass