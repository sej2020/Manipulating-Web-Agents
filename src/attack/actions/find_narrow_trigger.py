import argparse
import json
import os
import math
import torch
import pathlib
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.attack.nanogcg_plus.gcg_plus import run, GCGConfig

from src.attack.utils.promptify import promptify_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--json", type=str, help="File and path for the JSON file to find a trigger for.", default="")
    parser.add_argument("--target", type=str, default="```click(\"41\")```", help="The desired output when triggered.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    
    # Config for TWTI expr
    parser.add_argument("--trigger_length", type=int, required=True)
    parser.add_argument("--include_target", required=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--loss_fn", choices=["cw", "mm", "ce"])
    parser.add_argument("--search_width", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)

    # Defaults are probably good
    parser.add_argument("--model", choices=["mistral-7B", "mistral-24B", "llama2", "llama3"], default="llama3", help="The model to use for generation.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type to use for the model.")
    args = parser.parse_args()
    
    return args


def find_trigger():
    """
    Uses nanogcg_plus to find a trigger that will cause the specified model to generate the target output when given the website and
    goal data provided in the specified JSON file. Saves the trigger to a JSON file in the triggers directory.
    """

    args = parse_args()
    print("~~ARGS~~:\n", args.__dict__, flush=True)

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

    json_path = pathlib.Path(args.json)
    with open(json_path) as f:
        obs_dict = json.load(f)

    sys_content, user_content = promptify_json(obs_dict)

    messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content}
        ]


    if args.include_target:
        n_target_tokens = len(tokenizer.tokenize(args.target))
        total_xs = args.trigger_length - n_target_tokens
        starting_str = math.floor(total_xs/2) * "x " + args.target + math.ceil(total_xs/2) * "x "
    else:
        starting_str = args.trigger_length * "x "
    
    match args.loss_fn:
        case "cw":
            use_mm = False
            use_cw = True
        case "mm":
            use_mm = True
            use_cw = False
        case "ce":
            use_mm = False
            use_cw = False

    config = GCGConfig(
        num_steps=300,
        optim_str_init=starting_str,
        search_width=args.search_width,
        batch_size=4,
        topk=args.top_k,
        use_mellowmax=use_mm,
        use_cw_loss=use_cw,
        early_stop=True,
        verbosity="INFO",
        add_space_before_target = True if args.model == "llama2" else False,
    )

    result = run(
        model,
        tokenizer,
        messages,
        args.target,
        config,
    )

    print("~~ BEST TRIGGER: ~~", flush=True)
    print(result.best_string, flush=True)

    test_message = messages.copy()
    test_message[-1]["content"] = test_message[-1]["content"].replace("{optim_str}", result.best_string)

    input = tokenizer.apply_chat_template(test_message, add_generation_prompt=True, return_tensors="pt").to(args.device)
    output = model.generate(input, do_sample=False, max_new_tokens=512)

    print(f"Prompt:\n{test_message[-1]['content']}\n", flush=True)
    print(f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]}", flush=True)

    os.makedirs("triggers_twti", exist_ok=True)
    with open(f"triggers_twti/{json_path.stem}_{datetime.datetime.now()}.json", "w") as f:
        trig_dict = {
            "json_path": str(json_path),
            "model": args.model,
            "target": args.target,
            "trigger": result.best_string,
            "time_to_find": result.time_to_find_s,
            "num_steps": result.num_steps,
            "args": args.__dict__,
        }
        json.dump(trig_dict, f, indent=4)


if __name__ == "__main__":
    find_trigger()