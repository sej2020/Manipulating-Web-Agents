import argparse
import json
from src.model.agent import DemoAgentArgs
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model",
        default="mistral-7B",
        choices=["mistral-7B", "mistral-24B", "gpt-4o-mini"],
        help="Which LLM to use for the agent.",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://sj110.pages.iu.edu/attack_demo.html",
        help="Starting URL for the environment.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=False,
        help="Use HTML in the agent's observation space.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in the agent's observation space.",
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=False,
        help="Run the environment in headless mode.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Navigate to the contact page",
        help="Goal of the task. Necessary if headless is True.",
    )
    parser.add_argument(
        "--trigger_json",
        type=str,
        default="",
        help="The name of the JSON file to find a trigger to attack the LLM on this specific task. If not provided, there will be no attack.",
    )

    return parser.parse_args()


def main():
    """
    Runs an experiment where an LLM agent interacts with a web environment to complete a task. If a trigger json is provided,
    then an adversarial attack is performed on the LLM web navigation agent.
    """

    args = parse_args()

    if args.trigger_json:
        with open(f"triggers/{args.trigger_json}.json") as f:
            trigger_dict = json.load(f)
            trigger = trigger_dict["trigger"]
    else:
        trigger = None
    
    # setting up agent config
    agent_args = DemoAgentArgs(
        model_name=args.model,
        chat_mode=False,
        demo_mode="default",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=False,
        trigger=trigger,
    )
    # setting up environment config
    env_args = EnvArgs(
        task_name="openended",
        task_seed=None,
        max_steps=100,
        headless=args.headless,  # keep the browser open
        viewport={"width": 800, "height": 680},  # can be played with if needed

    )

    if not args.headless:
        agent_args.chat_mode = True
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}
    else:
        env_args.task_kwargs = {"start_url": args.start_url, "goal": args.goal}

    # setting up the experiment
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )

    # running and logging results
    exp_args.prepare("./results")
    exp_args.run()

    # loading and printing results
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
