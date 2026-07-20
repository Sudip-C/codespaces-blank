from model.tokenizer import GPTTokenizer


def main():

    tokenizer = GPTTokenizer()

    text = "Hello, I am building an AI model.this will be the first /"

    token_ids = tokenizer.encode(
        text
    )

    decoded_text = tokenizer.decode(
        token_ids
    )

    print("Original text:")
    print(text)

    print("\nToken IDs:")
    print(token_ids)

    print("\nDecoded text:")
    print(decoded_text)

    print("\nVocabulary size:")
    print(tokenizer.vocab_size)


if __name__ == "__main__":

    main()