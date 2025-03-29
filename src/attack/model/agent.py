import base64
import dataclasses
import io
import logging
from dotenv import load_dotenv
import os

import numpy as np
import openai
from PIL import Image

from src.attack.model.mistral import complete

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AbstractAgentArgs, Agent
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", api_key)  # Debugging: Ensure it's set
openai.api_key = api_key

logger = logging.getLogger(__name__)


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class DemoAgent(Agent):
    """A basic agent using OpenAI API, to demonstrate BrowserGym's functionalities."""

    def obs_preprocessor(self, obs: dict) -> dict:

        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def __init__(
        self,
        model_name: str,
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
        trigger: str = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.trigger = trigger

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        if self.model_name != "mistral":
            self.openai_client = openai.OpenAI()

        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],  # define a subset of the action space
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode=demo_mode,  # add visual effects
        )

        self.action_history = []


    def get_action(self, obs: dict) -> tuple[str, dict]:

        system_msgs = []
        user_msgs = []

        if self.chat_mode:
            print("?"*100)
            system_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, to which the user gives you instructions and to which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
                }
            )
            # append chat messages
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Chat Messages
""",
                }
            )
            for msg in obs["chat_messages"]:
                if msg["role"] in ("user", "assistant", "infeasible"):
                    user_msgs.append(
                        {
                            "type": "text",
                            "text": f"""\
- [{msg['role']}] {msg['message']}
""",
                        }
                    )
                elif msg["role"] == "user_image":
                    user_msgs.append({"type": "image_url", "image_url": msg["message"]})
                else:
                    raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")

        else:
            assert obs["goal_object"], "The goal is missing."
            system_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Instructions

Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
                }
            )
            # append goal
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Goal
""",
                }
            )
            # goal_object is directly presented as a list of openai-style messages
            user_msgs.extend(obs["goal_object"])

        # append url of all open tabs
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Currently open tabs
""",
            }
        )
        for page_index, (page_url, page_title) in enumerate(
            zip(obs["open_pages_urls"], obs["open_pages_titles"])
        ):
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}
  Title: {page_title}
  URL: {page_url}
""",
                }
            )

        # append page AXTree (if asked)
        if self.use_axtree:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Current page Accessibility Tree

{obs["axtree_txt"]}

""",
                }
            )
        # append page HTML (if asked)
        if self.use_html:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Current page DOM

{obs["pruned_html"]}

""",
                }
            )

        # append page screenshot (if asked)
        if self.use_screenshot:
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
# Current page Screenshot
""",
                }
            )
            user_msgs.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_jpg_base64_url(obs["screenshot"]),
                        "detail": "auto",
                    },  # Literal["low", "high", "auto"] = "auto"
                }
            )

        # append action space description
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

""",
            }
        )

        # append past actions (and last error message) if any
        if self.action_history:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# History of past actions
""",
                }
            )
            user_msgs.extend(
                [
                    {
                        "type": "text",
                        "text": f"""\

{action}
""",
                    }
                    for action in self.action_history
                ]
            )

            if obs["last_action_error"]:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
# Error message from last action

{obs["last_action_error"]}

""",
                    }
                )

        # ask for the next action
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Next action

You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action. If you believe you are done with the task, please produce a noop.
""",
            }
        )

        prompt_text_strings = []
        for message in system_msgs + user_msgs:
            match message["type"]:
                case "text":
                    prompt_text_strings.append(message["text"])
                case "image_url":
                    image_url = message["image_url"]
                    if isinstance(message["image_url"], dict):
                        image_url = image_url["url"]
                    if image_url.startswith("data:image"):
                        prompt_text_strings.append(
                            "image_url: " + image_url[:30] + "... (truncated)"
                        )
                    else:
                        prompt_text_strings.append("image_url: " + image_url)
                case _:
                    raise ValueError(
                        f"Unknown message type {repr(message['type'])} in the task goal."
                    )
        full_prompt_txt = "\n".join(prompt_text_strings)
        logger.info(full_prompt_txt)
        

        # Send prompt to model based on model name

        if "gpt" in self.model_name:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msgs},
                    {"role": "user", "content": user_msgs},
                ],
            )
            action = response.choices[0].message.content

        elif "mistral" in self.model_name:
            if self.model_name == "mistral-7B":
                model_size = "7B"
            elif self.model_name == "mistral-24B":
                model_size = "24B"
            else:
                raise ValueError(f"Unknown model name {self.model_name}")

            sys_content = '\n'.join([s['text'] for s in system_msgs])
            user_content = '\n'.join([u['text'] for u in user_msgs])
            if self.trigger:
                user_content.replace("Hidden", self.trigger)
            response = complete(
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                model_size=model_size
            )
            action = response

        else:
            raise ValueError(f"Unknown model name {self.model_name}")

        self.action_history.append(action)

        if "I'm done" in action or "noop()" in action:
            return None, {}

        return action, {}


@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    """
    This class is meant to store the arguments that define the agent.

    By isolating them in a dataclass, this ensures serialization without storing
    internal states of the agent.
    """

    model_name: str = "gpt-4o-mini"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False
    trigger: str = None

    def make_agent(self):
        return DemoAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            trigger=self.trigger,
        )
