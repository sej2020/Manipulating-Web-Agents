import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.attack.nanogpp.gpp import run, GPPConfig

from src.attack.utils.promptify import promptify_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_name", type=str, help="Name of the JSON file in the data directory to find a trigger for.")
    parser.add_argument("--model", choices=["mistral-7B", "mistral-24B", "llama2", "llama3"], default="mistral-7B", help="The model to use for generation.")
    parser.add_argument("--target", type=str, default="```click(\"41\")```", help="The desired output when triggered.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type to use for the model.")
    args = parser.parse_args()
    return args


def find_trigger():
    """
    Uses nanoGPP to find a trigger that will cause the specified model to generate the target output when given the website and
    goal data provided in the specified JSON file. Saves the trigger to a JSON file in the triggers directory.
    """

    args = parse_args()
    match args.model:
        case "mistral-7B":
            model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        case "mistral-24B":
            model_name = "mistralai/Mistral-Small-24B-Instruct-250"
        case "llama2":
            model_name = "meta-llama/Llama-2-7b-chat-hf"
        case "llama3":
            model_name = "meta-llama/Llama-3.1-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, args.dtype)).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(f"data/{args.json_name}.json") as f:
        obs_dict = json.load(f)

    sys_content, user_content = promptify_json(obs_dict)

    messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content}
        ]

    # could be customizable from the command line in the future
    config = GPPConfig(
        num_steps=1000,
        search_width=128,
        topk=64,
        early_stop=True,
        verbosity="INFO",
        batch_size=4,
        add_space_before_target = True if args.model == "llama2" else False,
    )

    result = run(
        model,
        tokenizer,
        messages,
        args.target,
        config,
    )

    print("~~ BEST TRIGGER: ~~")
    print(result.best_string)

    test_message = messages.copy()
    test_message[-1]["content"] = test_message[-1]["content"].replace("{optim_str}", result.best_string)

    input = tokenizer.apply_chat_template(test_message, add_generation_prompt=True, return_tensors="pt").to(args.device)
    output = model.generate(input, do_sample=False, max_new_tokens=512)

    print(f"Prompt:\n{test_message[-1]['content']}\n")
    print(f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]}")


if __name__ == "__main__":
    find_trigger()