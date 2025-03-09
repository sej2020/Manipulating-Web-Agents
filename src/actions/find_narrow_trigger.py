import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nanogcg
from nanogcg import GCGConfig

from src.utils.data_processing import promptify_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_name", type=str, help="Name of the JSON file in the data directory to find a trigger for.")
    parser.add_argument("--model", choices=["mistral-7B", "mistral-24B"], default="mistral-7B", help="The model to use for generation.")
    parser.add_argument("--target", type=str, default="```click('41')```", help="The desired output when triggered.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    return args


def find_trigger():
    """
    Uses nanoGCG to find a trigger that will cause the specified model to generate the target output when given the website and
    goal data provided in the specified JSON file. Saves the trigger to a JSON file in the triggers directory.
    """

    args = parse_args()
    if args.model == "mistral-7B":
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model == "mistral-24B":
        model_name = "mistralai/Mistral-Small-24B-Instruct-250"

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
    config = GCGConfig(
        num_steps=500,
        search_width=64,
        topk=64,
        seed=42,
        batch_size=8,
        verbosity="WARNING"
    )

    result = nanogcg.run(
        model,
        tokenizer,
        messages,
        args.target,
        config,
    )

    print("~~ BEST TRIGGER: ~~")
    print(result.best_string)

    messages[-1]["content"] = messages[-1]["content"] + " " + result.best_string

    input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(args.device)
    output = model.generate(input, do_sample=False, max_new_tokens=512)

    print(f"Prompt:\n{messages[-1]['content']}\n")
    print(f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]}")

    with open(f"triggers/{args.json_name}_trigger.json", "w") as f:
        trig_dict = {
            "json_name": args.json_name,
            "model": args.model,
            "target": args.target,
            "trigger": result.best_string,
        }
        json.dump(trig_dict, f, indent=4)


if __name__ == "__main__":
    find_trigger()