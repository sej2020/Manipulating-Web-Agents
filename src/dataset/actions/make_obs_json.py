import argparse
from src.attack.model.agent import DemoAgentArgs
from browsergym.experiments import EnvArgs, ExpArgs


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
        "--url",
        type=str,
        default="https://sj110.pages.iu.edu/travel_ad_demo.html",
        help="Starting URL for the environment.",
    )
    parser.add_argument("--json_filepath", type=str, default="", help="Path to the JSON file where observation data will be saved.")

    return parser.parse_args()


def main():
    args = parse_args()

    agent_args = DemoAgentArgs(
        model_name="llama3",
        chat_mode=False,
        demo_mode="default",
        use_html=False,
        use_axtree=True,
        use_screenshot=False,
        trigger=None,
        save_obs_and_kill_file=args.json_filepath,
    )
    # setting up environment config
    env_args = EnvArgs(
        task_name="openended",
        task_seed=None,
        max_steps=10,
        headless=False,  # keep the browser open
        viewport={"width": 800, "height": 680},  # can be played with if needed
    )

    agent_args.chat_mode = True
    env_args.wait_for_user_message = True
    env_args.task_kwargs = {"start_url": args.url}
    # env_args.task_kwargs = {"start_url": args.url, "goal": ""}

    # setting up the experiment
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )

    # running and logging results
    exp_args.prepare("./results")
    exp_args.run()


if __name__ == "__main__":
    main()
