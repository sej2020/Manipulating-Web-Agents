import openai
import json
import os
import pathlib
from browsergym.core.action.highlevel import HighLevelActionSet

def filter_websites():
    # remove banned websites
    # balance classes
    json_files = list(pathlib.Path('data').rglob('*.json'))
    print(len(json_files))
    sites= {}
    for file in json_files:
        json_data = json.load(open(file, 'r'))
        if 'blocked' in json_data['axtree_txt'] or 'banned' in json_data['axtree_txt'] or 'restricted' in json_data['axtree_txt'] or 'forbidden' in json_data['axtree_txt']:
            print('blocked')
            os.remove(file)
            continue
        elif '404' in json_data['open_pages_titles'][0] or '403' in json_data['open_pages_titles'][0]:
            print('blocked')
            os.remove(file)
            continue

        site = json_data['open_pages_urls'][0]
        site = site.split('/')[2]
        if site not in sites:
            sites[site] = 1
        else:
            sites[site] += 1

        if sites[site] > len(json_files) / 50:
            os.remove(file)
            continue

    return 'Done!'


def batch_goal_prompt(ax_tree):
    prompt = f"""
    Based on the following webpage accessibility tree, please provide a web navigation goal for a user to achieve.
    Examples of goals: "Find the contact information of the author", "Locate the search bar", "Navigate to the homepage of the blog", "Purchase a dress from the online store", etc.
    Here is the accessibility tree:

    {ax_tree}
    """
    return prompt


def promptify_json(obs_json):

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
    user_msgs.extend(obs_json["goal_object"])

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
Tab {page_index}{" (active tab)" if page_index == obs_json["active_page_index"] else ""}
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
    # query mistral
    messages=[
            {"role": "system", "content": '\n'.join([s['text'] for s in system_msgs])},
            {"role": "user", "content": '\n'.join([u['text'] for u in user_msgs])},
        ]
    return messages
    # ready to be fed directly into the model's complete function

# jsonl format for the batch api
# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

if __name__ == '__main__':
    # filter_websites()
    # json_files = list(pathlib.Path('data').rglob('*.json'))
    # batch1_jsonl = []
    # batch2_jsonl = []
    # for file in json_files[:len(json_files)//2]:
    #     json_data = json.load(open(file, 'r'))
    #     batch1_jsonl.append(
    #         {
    #             "custom_id": file.stem,
    #             "method": "POST",
    #             "url": "/v1/chat/completions",
    #             "body": {
    #                 "model": "gpt-4o-mini",
    #                 "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": f"{batch_goal_prompt(json_data['axtree_txt'])}"}],
    #                 "max_tokens": 200
    #             }
    #         }
    #     )
    # for file in json_files[len(json_files)//2:]:
    #     json_data = json.load(open(file, 'r'))
    #     batch2_jsonl.append(
    #         {
    #             "custom_id": file.stem,
    #             "method": "POST",
    #             "url": "/v1/chat/completions",
    #             "body": {
    #                 "model": "gpt-4o-mini",
    #                 "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": f"{batch_goal_prompt(json_data['axtree_txt'])}"}],
    #                 "max_tokens": 200
    #             }
    #         }
    #     )
    # with open('goal_prompt_batch1.jsonl', 'w') as f:
    #     for item in batch1_jsonl:
    #         f.write(json.dumps(item))
    #         f.write('\n')
    # with open('goal_prompt_batch2.jsonl', 'w') as f:
    #     for item in batch2_jsonl:
    #         f.write(json.dumps(item))
    #         f.write('\n')
    # print('Done!')
    
    # # submitting the batch job
    # from openai import OpenAI
    # client = OpenAI()

    # for f_name in ["goal_prompt_batch1.jsonl", "goal_prompt_batch2.jsonl"]:
    #     batch_input_file = client.files.create(
    #         file=open(f_name, "rb"),
    #         purpose="batch"
    #     )

    #     print(batch_input_file)
    #     print(batch_input_file.id)

    #     batch_input_file_id = batch_input_file.id
    #     batch = client.batches.create(
    #         input_file_id=batch_input_file_id,
    #         endpoint="/v1/chat/completions",
    #         completion_window="24h",
    #         metadata={
    #             "description": f"batch web nav goal job for {f_name}"
    #         }
    #     )
    #     print(batch)

    # checking the status of the batch job
    # curl https://api.openai.com/v1/batches -H "Authorization: Bearer $OPENAI_API_KEY"
    # batch_67bdfb010a388190a3f0dd63a6ad4c36
    # batch_67bdfb0ca5348190ba947033054fa950
    from openai import OpenAI
    client = OpenAI()

    batch = client.batches.retrieve("batch_67bdfb0ca5348190ba947033054fa950")
    print(batch)

