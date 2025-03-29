import json
import os
import pathlib
from openai import OpenAI


def filter_websites() -> str:
    """
    After websites observation dictionaries are created, this function filters out banned websites and removes many 
    instances of the same website to balance classes.

    Returns:
        out (str): A message indicating that the function has completed.
    """
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


def batch_goal_prompt(ax_tree: dict) -> str:
    """
    Formats the accessibility tree into a prompt for gpt to create a web navigation goal.

    Args:
        ax_tree (dict): The accessibility tree of the website

    Returns:
        out (str): The prompt asking gpt to create a web navigation goal
    """
    prompt = f"""
    Based on the following webpage accessibility tree, please provide a web navigation goal for a user to achieve.
    Examples of goals: "Find the contact information of the author", "Locate the search bar", "Navigate to the homepage of the blog", "Purchase a dress from the online store", etc.
    Here is the accessibility tree:

    {ax_tree}
    """
    return prompt


def prepare_batch_files(json_file_list: list, n_splits: int = 2) -> str:
    """
    The OpenAI API has a limit on the number of requests that can be made in a single batch job. This function splits the
    list of json files into n_splits batches to be processed separately.

    Args:
        json_file_list (list): A list of json files to be processed
        n_splits (int, optional): The number of batches to split the json files into. Defaults to 2.

    Returns:
        out (str): A message indicating that the function has completed.
    """

    for i in range(n_splits):
        batch_jsonl = []
        split_size = len(json_file_list)//n_splits
        for file in json_file_list[i*split_size : (i+1)*split_size]:
            json_data = json.load(open(file, 'r'))
            batch_jsonl.append(
                {
                    "custom_id": file.stem,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": f"{batch_goal_prompt(json_data['axtree_txt'])}"}],
                        "max_tokens": 200
                    }
                }
            )
        with open(f'data/goal_prompt_batch{i}.jsonl', 'w') as f:
            for item in batch_jsonl:
                f.write(json.dumps(item))
                f.write('\n')

    return 'Done!'


def submit_goal_object_batch(n_splits: int = 2) -> str:
    """
    This function is to be run after get_website_data() has been used to generate a suitable amount of website data.
    The json observations it produces will be in need of cleaning and are without goal objects. This function will 
    filter websites that are blocked, rebalance the site classes, and submit a batch job to generate goal objects for
    the remaining websites.

    Args:
        n_splits (int, optional): The number of batches to split the json files into. Defaults to 2.

    Returns:
        ids (list): A list of the ids of the batch jobs submitted
    """
    filter_websites()
    json_files = list(pathlib.Path('data').rglob('*.json'))
    prepare_batch_files(json_files, n_splits)

    # submitting the batch job
    client = OpenAI()

    ids = []
    for i in range(n_splits):
        f_name = f"data/goal_prompt_batch{i}.jsonl"
        batch_input_file = client.files.create(
            file=open(f_name, "rb"),
            purpose="batch"
        )

        print(batch_input_file)
        print(batch_input_file.id)

        batch_input_file_id = batch_input_file.id
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"batch web nav goal job for {f_name}"
            }
        )
        print(batch)
        ids.append(batch.id)
    
    return ids
       

def get_goal_object_batch(ids: list) -> None:
    """
    If the batch jobs are completed, this writes goal_objects to corresponding json files

    Note: If you have lost the ids of the batch jobs, you can retrieve them by running the following command in the terminal:
        `curl https://api.openai.com/v1/batches -H "Authorization: Bearer $OPENAI_API_KEY"`

    Args:
        ids (list): A list of the ids of the batch jobs submitted

    Returns:
        None
    """
    client = OpenAI()

    for idx, batch_id in enumerate(ids):
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            print(batch)
            output_file = client.files.content(batch.output_file_id)
            for line in output_file.iter_lines():
                response_dict = json.loads(line)
                goal_object = parse_batch_reponse(response_dict)
                json_name = response_dict['custom_id']
                with open(f"data/{json_name}.json", 'r') as f:
                    data_dict = json.load(f)
                data_dict['goal_object'] = [goal_object]
                with open(f"data/{json_name}.json", 'w') as f:
                    json.dump(data_dict, f, indent=4)
            
        else:
            print(f"Batch job {batch_id} not completed yet: status is {batch.status}")


def parse_batch_reponse(response_dict: dict) -> str:
    """
    Extracts the goal object from each response in the batch job

    Args:
        response_dict (dict): The response dictionary from the batch job

    Returns:
        out (str): The goal object extracted from the response
    """
    response_txt = response_dict['response']['body']['choices'][0]['message']['content']
    if ':' in response_txt:
        goal_raw = response_txt.split(':')[1]
        if "." in goal_raw:
            goal_raw = goal_raw.split('.')[0]
    elif 'accessibility tree' in response_txt:
        goal_raw = response_txt.split("\"")[1]
    else:
        goal_raw = response_txt
    
    goal = goal_raw.strip().replace('\n', ' ').replace('\"', '').replace('*', '')

    return goal


