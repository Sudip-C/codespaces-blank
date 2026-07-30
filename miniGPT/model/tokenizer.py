import tiktoken


class GPTTokenizer:

    def __init__(self):

        self.tokenizer = tiktoken.get_encoding(
            "gpt2"
        )

    def encode(self, text: str):

        return self.tokenizer.encode(
            text,
            allowed_special={
                "<|endoftext|>"
            }
        )

    def decode(self, token_ids):

        return self.tokenizer.decode(
            token_ids
        )

    @property
    def vocab_size(self):

        return self.tokenizer.n_vocab
    @property
    def eos_token_id(self):
        return self.tokenizer.eot_token