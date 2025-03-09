from openai import OpenAI
import pathlib
import json


def batch_annotate_prompt(ax_tree: dict) -> str:
    prompt = f"""
    Your task is to categorize this webpage in a few different ways. Based on the accessibility tree of a webpage, provide the following information:
    1. the type of webpage (e.g. video, article, login page, shopping page, etc.)
    2. the complexity of the webpage from 1-10 (e.g. 1 for a simple text paragraph up to 10 for a crazy complicated interactive visualization software)
    3. the genre of the webpage (e.g. entertainment, education, news, sports)
    4. the topic of the webpage (e.g. basketball, cooking, WW2, etc.)

    Here is the accessibility tree:
    {ax_tree[:20_000]}

    Please provide your answers in a numbered list, with each item on a new line. PLEASE ONLY USE ONE WORD FOR EACH ANSWER. For example:
    1. article
    2. 4
    3. news
    4. Brexit
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
                        "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": f"{batch_annotate_prompt(json_data['axtree_txt'])}"}],
                        "max_tokens": 200
                    }
                }
            )
        with open(f'anno_prompt_batch_missing.jsonl', 'w') as f:
            for item in batch_jsonl:
                f.write(json.dumps(item))
                f.write('\n')

    return 'Done!'


def submit_annotate_batch(n_splits: int = 2) -> str:
    """
    Args:
        n_splits (int, optional): The number of batches to split the json files into. Defaults to 2.

    Returns:
        ids (list): A list of the ids of the batch jobs submitted
    """
    # check if redo_files.json exists, if so, redo those files
    if pathlib.Path('redo_files.json').exists():
        with open('redo_files.json', 'r') as f:
            need_to_redo = json.load(f)
        json_files = [f"data/{k}.json" for k in need_to_redo.keys()]
        json_files = [pathlib.Path(f) for f in json_files]
    elif pathlib.Path('missing_files').exists():
        with open('missing_files.json', 'r') as f:
            missing_files = json.load(f)
        json_files = [pathlib.Path(f) for f in missing_files['missing_files']]
    else:
        json_files = list(pathlib.Path('data').rglob('*.json'))
    breakpoint()
    prepare_batch_files(json_files, n_splits)

    # submitting the batch job
    client = OpenAI()

    ids = []
    for i in range(n_splits):
        f_name = f"anno_prompt_batch_missing.jsonl"
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
                "description": f"batch web annotation job for {f_name}"
            }
        )
        print(batch)
        ids.append(batch.id)
    
    return ids
       

def get_anno_batch(ids: list) -> None:
    """
    Note: If you have lost the ids of the batch jobs, you can retrieve them by running the following command in the terminal:
        `curl https://api.openai.com/v1/batches -H "Authorization: Bearer $OPENAI_API_KEY"`

    Args:
        ids (list): A list of the ids of the batch jobs submitted

    Returns:
        None
    """
    client = OpenAI()

    need_to_redo = {}
    for idx, batch_id in enumerate(ids):
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            print(batch)
            output_file = client.files.content(batch.output_file_id)

            for line in output_file.iter_lines():

                response_dict = json.loads(line)

                anno_dict = parse_batch_reponse(response_dict)
                json_name = response_dict['custom_id']
                print(json_name)

                redoing = False
                for k,v in anno_dict.items():
                    if len(v.split(' ')) > 3:
                        need_to_redo[json_name] = anno_dict
                        redoing = True
                        break
                if redoing:
                    continue

                with open(f"data/{json_name}.json", 'r') as f:
                    data_dict = json.load(f)

                for k,v in anno_dict.items():
                    data_dict[k] = v

                with open(f"data/{json_name}.json", 'w') as f:
                    json.dump(data_dict, f, indent=4)
  
        else:
            print(f"Batch job {batch_id} not completed yet: status is {batch.status}")
    
    print(need_to_redo)
    breakpoint()
    with open('redo_files.json', 'w') as f:
        json.dump(need_to_redo, f, indent=4)


def parse_batch_reponse(response_dict: dict) -> dict:
    """
    TBD
    """
    response_txt = response_dict['response']['body']['choices'][0]['message']['content']
    anno_dict = {
        'type': response_txt.split('1.')[1].split('\n')[0].strip(),
        'complexity': response_txt.split('2.')[1].split('\n')[0].strip(),
        'genre': response_txt.split('3.')[1].split('\n')[0].strip(),
        'topic': response_txt.split('4.')[1].split('\n')[0].strip()
    }

    return anno_dict


if __name__ == '__main__':
    # ids = submit_annotate_batch(1)
    # with open('batch_ids.txt', 'w') as f:
    #     f.write(str(ids))
    ids = ['batch_67c4f5eb24408190850cab248d01be45']
    get_anno_batch(ids)