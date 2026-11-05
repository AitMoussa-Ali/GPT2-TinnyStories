
from transformers import GPT2TokenizerFast


class GPT2Tokenizer:
    
    def __init__(self, pretrained_model: str = "gpt2"):

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            pretrained_model
        )

        self.vocab_size = self.tokenizer.vocab_size
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def encode_story(self, text: str) -> list[int]:
        token_ids = self.encode(text)
        token_ids.append(self.eos_token_id)
        return token_ids
