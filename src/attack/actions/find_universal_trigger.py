import argparse
import json
import yaml
import torch
import pathlib
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from attack.nanogcg_plus.gcg_plus import run, GCGConfig

from src.attack.utils.promptify import promptify_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup_yaml", type=str, required=True, help="Path to the setup YAML file, which should have a list of JSON files to use and their corresponding targets.")
    parser.add_argument("--model", choices=["mistral-7B", "mistral-24B", "llama2", "llama3"], default="llama3", help="The model to use for generation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type to use for the model.")
    parser.add_argument("--optim_str_init", type=str, default="x x x x x x x x x x x x x x x x x x x x ")
    args = parser.parse_args()
    return args


def find_trigger():
    """
    Uses nanogcg_plus to find a trigger that will cause the specified model to generate the target output when given the website and
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

    # Load the setup YAML file
    setup_yaml_path = pathlib.Path(args.setup_yaml)
    with open(setup_yaml_path, "r") as f:
        setup = yaml.safe_load(f)
    
    jsons = setup["jsons"]
    targets = setup["targets"]
    test_jsons = setup["test_jsons"]
    test_targets = setup["test_targets"]

    messages = []
    for json_file in jsons:
        with open(pathlib.Path(json_file)) as f:
            obs_dict = json.load(f)
        obs_dict_copy = obs_dict.copy()
        for goal in obs_dict["goal_object"]:
            obs_dict_copy["goal_object"] = [goal]
            sys_content, user_content = promptify_json(obs_dict_copy)
            message=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content}
                ]
            messages.append(message)

    assert len(messages) == len(targets), "Number of messages and targets must be the same."

    test_messages = []
    for json_file in test_jsons:
        with open(pathlib.Path(json_file)) as f:
            obs_dict = json.load(f)
        obs_dict_copy = obs_dict.copy()
        for goal in obs_dict["goal_object"]:
            obs_dict_copy["goal_object"] = [goal]
            sys_content, user_content = promptify_json(obs_dict_copy)
            message=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content}
                ]
        test_messages.append(message)

    assert len(test_messages) == len(test_targets), "Number of test messages and targets must be the same."

    config = GCGConfig(
        num_steps=2000,
        optim_str_init=args.optim_str_init,
        universal=True,
        early_stop=True,
        verbosity="INFO",
        batch_size=1,
        use_cw_loss=False,
        add_space_before_target = True if args.model == "llama2" else False,
    )

    result = run(
        model,
        tokenizer,
        messages,
        targets,
        config,
    )

    # class Result:
    #     def __init__(self, best_string):
    #         self.best_string = best_string
    # result = Result("I am a test trigger.")            

    print("~~ BEST TRIGGER: ~~", flush=True)
    print(result.best_string, flush=True)

    for message in messages:
        message[-1]["content"] = message[-1]["content"].replace("{optim_str}", result.best_string)

        input = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").to(args.device)
        output = model.generate(input, do_sample=False, max_new_tokens=512)

        print(f"Prompt:\n{message[-1]['content']}\n", flush=True)
        print(f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]}", flush=True)


    print("~~ TESTING TRIGGER: ~~", flush=True)

    for test_message in test_messages:
        test_message[-1]["content"] = test_message[-1]["content"].replace("{optim_str}", result.best_string)

        input = tokenizer.apply_chat_template(test_message, add_generation_prompt=True, return_tensors="pt").to(args.device)
        output = model.generate(input, do_sample=False, max_new_tokens=512)

        print(f"Prompt:\n{test_message[-1]['content']}\n", flush=True)
        print(f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]}", flush=True)


    trigger_identifier = pathlib.Path(jsons[0]).stem
    with open(f"triggers/{trigger_identifier}_trigger_uni_{datetime.datetime.now()}.json", "w") as f:
        trig_dict = {
            "jsons": jsons,
            "model": args.model,
            "targets": targets,
            "test_jsons": test_jsons,
            "test_targets": test_targets,
            "trigger": result.best_string,
        }
        json.dump(trig_dict, f, indent=4)


if __name__ == "__main__":
    find_trigger()