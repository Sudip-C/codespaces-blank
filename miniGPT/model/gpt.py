import torch
import torch.nn as nn

from model.embedding import GPTEmbedding
from model.transformer import TransformerBlock


class GPTModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.embedding = GPTEmbedding(
            config
        )

        self.transformer_blocks = nn.ModuleList(

            [
                TransformerBlock(config)

                for _ in range(
                    config.n_layers
                )
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.d_model
        )

        self.output_head = nn.Linear(

            config.d_model,

            config.vocab_size,

            bias=False
        )

    def forward(self, input_ids):

        x = self.embedding(
            input_ids
        )

        for block in self.transformer_blocks:

            x = block(x)

        x = self.final_layer_norm(
            x
        )

        logits = self.output_head(
            x
        )

        return logits
    @torch.no_grad()
    def generate(

        self,

        input_ids,

        max_new_tokens=150,

        temperature=0.8,

        top_k=40

    ):

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the last max_seq_len tokens
            input_cond = input_ids[:, -self.config.max_seq_len:]

            logits = self(input_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            # Top-k sampling
            values, indices = torch.topk(

                logits,

                k=top_k

            )

            probs = torch.softmax(

                values,

                dim=-1

            )

            next_token = torch.multinomial(

                probs,

                num_samples=1

            )

            next_token = torch.gather(

                indices,

                -1,

                next_token

            )

            input_ids = torch.cat(

                [

                    input_ids,

                    next_token

                ],

                dim=1

            )

        return input_ids