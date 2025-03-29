from transformers import pipeline
import torch
import gc


def complete(messages: list, model_size: str = "7B") -> str:
    """
    Queries a Mistral model to generate a response to a conversation.

    Args:
        messages (list): A list of dictionaries, each containing a role and content key.
        model_size (str): The size of the Mistral model to use. Defaults to "7B".
    """
    if model_size == "7B":
        model = "mistralai/Mistral-7B-Instruct-v0.3"
    elif model_size == "24B":
        model = "mistralai/Mistral-Small-24B-Instruct-250"
    else:
        raise ValueError("model_size must be either '7B' or '24B'")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipeline("text-generation", model=model, device=DEVICE)
    gc.collect()
    torch.cuda.empty_cache()
    out = pipe(messages, max_new_tokens=1000)
    print(out)
    return out[0]["generated_text"][2]['content']


if __name__ == '__main__':
    print(
        complete(
            [{"role":"system", "content": "You are a helpful assistent"}, 
             {"role": "user", "content": "What are the lyrics to the song 'Yellow Submarine' by The Beatles?"}]))
