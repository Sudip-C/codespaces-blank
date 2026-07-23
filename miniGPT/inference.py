import torch

from config import GPTConfig

from model.gpt import GPTModel

from model.tokenizer import GPTTokenizer


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50
):

    device = next(
        model.parameters()
    ).device

    # Convert prompt to token IDs

    token_ids = tokenizer.encode(
        prompt
    )

    input_ids = torch.tensor(

        [token_ids],

        dtype=torch.long,

        device=device
    )

    model.eval()

    with torch.no_grad():

        for _ in range(
            max_new_tokens
        ):

            # Keep only the latest tokens

            input_ids_cond = input_ids[

                :,

                -model.config.max_seq_len:
            ]

            # Get predictions

            logits = model(
                input_ids_cond
            )

            # Get the last token prediction

            logits = logits[:, -1, :]

            # Temperature

            logits = logits / temperature

            # Top-k filtering

            if top_k is not None:

                top_values, _ = torch.topk(

                    logits,

                    min(
                        top_k,
                        logits.size(-1)
                    )
                )

                minimum_value = top_values[
                    :, -1
                ]

                logits[
                    logits < minimum_value
                ] = float("-inf")

            # Convert to probabilities

            probabilities = torch.softmax(

                logits,

                dim=-1
            )

            # Sample next token

            next_token = torch.multinomial(

                probabilities,

                num_samples=1
            )

            # Add token

            input_ids = torch.cat(

                [
                    input_ids,
                    next_token
                ],

                dim=1
            )

    # Decode

    generated_text = tokenizer.decode(

        input_ids[0].tolist()
    )

    return generated_text


def main():

    config = GPTConfig()

    tokenizer = GPTTokenizer()

    model = GPTModel(
        config
    )
    
    model.to(
        config.device
    )
    # Load best_model

    model.load_state_dict(

        torch.load(

            "best_model.pt",

            map_location=config.device
        )
    )
    model.eval()
    

    prompt = input(
        "\nEnter a prompt: "
    )

    generated_text = generate_text(

        model=model,

        tokenizer=tokenizer,

        prompt=prompt,

        max_new_tokens=50,

        temperature=1,

        top_k=50
    )

    print(
        "\nGenerated text:\n"
    )

    print(
        generated_text
    )


if __name__ == "__main__":

    main()