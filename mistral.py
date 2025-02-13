from transformers import pipeline
import torch
import gc
pipe = pipeline("text-generation", model="mistralai/Mistral-Small-24B-Instruct-2501")

def complete(messages):
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    return pipe(messages, max_new_tokens=1000)[0]["generated_text"][2]['content']

if __name__ == '__main__':
    print(
        complete(
            [{"role":"system", "content": "You are a helpful assistent"}, 
             {"role": "user", "content": "What are the lyrics to the song 'Yellow Submarine' by The Beatles?"}]))
