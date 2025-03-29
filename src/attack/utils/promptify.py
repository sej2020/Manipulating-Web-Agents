from browsergym.core.action.highlevel import HighLevelActionSet

def promptify_json(obs_json: dict) -> tuple:
    """
    Converts json dictionary objects of website observations into the message format required by an LLM model.

    Args:
        obs_json (dict): The observation dictionary of a website

    Returns:
        sys_content (str): The system message content
        user_content (str): The user message content
    """

    action_set = HighLevelActionSet(
        subsets=["chat", "tab", "nav", "bid", "infeas"],
        strict=False,
        multiaction=False,
        demo_mode=False,
    )
    system_msgs = []
    user_msgs = []

    assert obs_json["goal_object"], "The goal is missing."
    system_msgs.append(
        {"type": "text", "text": f"""\
# Instructions

Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
        }
    )
    # append goal
    user_msgs.append(
        {"type": "text", "text": f"""\
# Goal
""",
        }
    )
    # goal_object is directly presented as a list of openai-style messages
    user_msgs.extend([{'type': 'text', 'text': goal} for goal in obs_json["goal_object"]]) 

    # append url of all open tabs
    user_msgs.append(
        {"type": "text", "text": f"""\
# Currently open tabs
""",
        }
    )
    for page_index, (page_url, page_title) in enumerate(
        zip(obs_json["open_pages_urls"], obs_json["open_pages_titles"])
    ):
        user_msgs.append(
            {"type": "text", "text": f"""\
Tab {page_index}" (active tab)"
Title: {page_title}
URL: {page_url}
""",
            }
        )

    # append page AXTree (if asked)
    user_msgs.append(
        {"type": "text", "text": f"""\
# Current page Accessibility Tree

{obs_json["axtree_txt"]}

""",
            }
        )

    # append action space description
    user_msgs.append(
        {"type": "text", "text": f"""\
# Action Space

{action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

""",
        }
    )

    # ask for the next action
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Next action

You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action. If you think you have completed the task, please simply say "I'm done."
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
    print(full_prompt_txt)

    sys_content = '\n'.join([s['text'] for s in system_msgs])
    user_content = '\n'.join([u['text'] for u in user_msgs])

    # TO BE USED LIKE THIS
    # messages=[
    #         {"role": "system", "content": sys_content},
    #         {"role": "user", "content": user_content}
    #     ]
    
    return sys_content, user_content
    # ready to be fed directly into the model's complete function