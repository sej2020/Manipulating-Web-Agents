from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc


def complete(messages: list, model_name: str) -> str:
    """
    Queries a Mistral model to generate a response to a conversation.

    Args:
        messages (list): A list of dictionaries, each containing a role and content key.
        model (str): The name of the model to use. Options: ["mistral-7B", "mistral-24B", "llama2", "llama3"]
    """
    match model_name:
        case "mistral-7B":
            model_full_name = "mistralai/Mistral-7B-Instruct-v0.3"
        case "mistral-24B":
            model_full_name = "mistralai/Mistral-Small-24B-Instruct-250"
        case "llama2":
            model_full_name = "meta-llama/Llama-2-7b-chat-hf"
        case "llama3":
            model_full_name = "meta-llama/Llama-3.1-8B-Instruct"
        case _:
            raise ValueError(f"Model {model_name} not supported. Supported models are: ['mistral-7B', 'mistral-24B', 'llama2', 'llama3']")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_full_name, torch_dtype=getattr(torch, "float16")).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_full_name)

    gc.collect()
    torch.cuda.empty_cache()

    input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    output = model.generate(input, do_sample=False, max_new_tokens=512)

    generation = tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]

    return generation


if __name__ == '__main__':
    print(
        complete(
            [{"role":"system", "content": "You are a helpful assistent"}, 
             {"role": "user", "content": "What are the lyrics to the song 'Yellow Submarine' by The Beatles?"}],
             "llama3"
             ))
