"""
Note: This script probably has some bugs, it hasn't been tested yet, only the component parts.
"""

import multiprocessing as mp
import argparse

from src.dataset.utils.website_data import get_website_data
from src.dataset.utils.data_processing import submit_goal_object_batch, get_goal_object_batch

parser = argparse.ArgumentParser(description='Prepare web navigation dataset')

parser.add_argument(
    'instruction', choices=['get_webs', 'set_goals', 'get_goals'], 
    help = """Which step to execute in preparing the web navigation dataset. Each step needs to be performed once, 
    in the order get_webs -> set_goals -> get_goals"""
)
parser.add_argument(
    '--n_batch_splits', type=int, default=2,
    help = """
    The OpenAI batch API has a limit on file-size that is exceeded for large datasets. 
    A good rule of thumb is to increase n_batch splits by 1 for every 5000 websites in your dataset.
    """
)
parser.add_argument(
    '--batch_ids', type=str, nargs='+',
    help = """
    The batch ids of the goal objects to retrieve. Wait 24 hrs after submitting the goal objects to OpenAI before running this step.
    """
)

args = parser.parse_args()

match args.instruction:
    case 'get_webs':
        # run main on 4 processes
        with mp.Pool(4) as pool:
            pool.map(get_website_data, range(4))
    case 'set_goals':
        ids = submit_goal_object_batch(args.n_batch_splits)
        print("Save your batch ids for retrieval the batches later. Your batch ids are: ")
        print(ids)
    case 'get_goals':
        get_goal_object_batch(args.batch_ids)
    case _:
        raise ValueError("Invalid instruction")