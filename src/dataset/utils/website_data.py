import requests
import random
import time
import json
import openai
import os
import playwright.sync_api
from dotenv import load_dotenv

from browsergym.core import _get_global_playwright
from browsergym.core.env import BrowserEnv, Chat
from browsergym.core.task import OpenEndedTask
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from src.dataset.utils.data_assets import prompt_example_options, prompt_phrasing_options, init_script

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", api_key)  # Debugging: Ensure it's set
openai.api_key = api_key


class DownloaderEnv(BrowserEnv):
    """
    Patchwork of BrowserGym code for initializing a playwright browser and downloading website data in the format suitable for 
    later BrowserGym use.
    """
    metadata = {"render_modes": None}

    def __init__(self, start_url: str) -> None:
        """
        Starts up the playwright chromium browser.

        Args:
            start_url (str): The url of the website to download data from.
        """
        self.start_url = start_url
        super().__init__(OpenEndedTask, {'start_url': start_url})

        # use the global Playwright instance
        pw: playwright.sync_api.Playwright = _get_global_playwright()

        # important: change playwright's test id attribute from "data-testid" to "bid"
        pw.selectors.set_test_id_attribute("bid")

        # create a new browser
        self.browser = pw.chromium.launch(
            headless=self.headless,
            **self.pw_chromium_kwargs,
        )
        self.context = self.browser.new_context(
            **self.pw_context_kwargs,
        )
        self.context.expose_binding(
            "browsergym_page_activated", lambda source: self._activate_page_from_js(source["page"])
        )
        self.context.add_init_script(
            init_script,
        )
        self.chat = Chat(
            headless=self.headless,
            chat_size=(500, 500, 800),
        )

        self.chat.add_message(
            role="assistant",
            msg="Hi! I am your UI assistant, I can perform web tasks for you. What can I help you with?",
        )
        self.goal_object = []
        self.last_action = ""
        self.last_action_error = ""


    def obs_from_url(self, url: str) -> dict:
        """
        Parses webpage to produce observation dictionary

        Args:
            url (str): The url of the webpage to parse

        Returns:
            obs (dict): The observation for the BrowserGym environment
        """
        self.start_time = time.time()
        self.start_url = url
        self.page = self.context.new_page()
        self.page.goto(self.start_url, timeout=10000)
        self._wait_dom_loaded()
        self._active_page_check()
        time.sleep(10)
        obs = self._get_obs()
        obs = self.obs_preprocessor(obs)
        self.page.close()
        return obs
    
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Preprocesses the observation dictionary

        Args:
            obs (dict): The observation dictionary 
        
        Returns:
            obs (dict): The observation dictionary with some preprocessing done
        """
        return {
            "chat_messages": obs["chat_messages"],
            # "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            # "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }


def random_query_list(n_queries: int = 100) -> str:
    """
    Have gpt generate a list of random internet search queries using a random prompt and examples

    Args:
        n_queries (int, optional): Number of queries to generate. Defaults to 100.

    Returns:
        out (str): The gpt response containing the list of queries
    """
    client = openai.OpenAI()
    prompt_phrase = random.choice(prompt_phrasing_options)
    prompt_examples = random.sample(prompt_example_options, 3)
    random_letter = random.choice('abcdefghijklmnopqrstuvwxyz')
    prompt = f"""
    {prompt_phrase}. Please provide a numbered list of {n_queries} examples. For example:
    1. {prompt_examples[0]}
    2. {prompt_examples[1]}
    3. {prompt_examples[2]}
    Please avoid using the letter {random_letter} in your examples, and be specific!
    """
    print(prompt)
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
        temperature=0.99,
            )
    return response.choices[0].message.content


def parse_query_list(gpt_response: str) -> list:
    """
    Parses the gpt response containing the list of queries into a list of queries.

    Args:
        gpt_response (str): The gpt response containing the list of queries

    Returns:
        out (list): A list of queries
    """
    print(gpt_response.split('\n'))
    return [q.split('. ')[1] for q in gpt_response.split('\n')[1:] if any(char.isdigit() for char in q)]


def fetch_sites_from_google(query: str) -> str:
    """
    Fetches a random site url from google search results for a given query.

    Args:
        query (str): The search query

    Returns:
        out (str): The url of the random site fetched from google
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    cx = os.getenv('GOOGLE_CX')
    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'
    response = requests.get(url)
    return random.choice([item['link'] for item in response.json().get('items', [])])


def get_website_data(delay: int = 0) -> None:
    """
    Generates a dataset of json files containing observation dictionaries for random websites.
    - Step 1: Have chatpgt generate many random short queries
    - Step 2: Use Google Custom Search API to fetch a site url for each query
    - Step 3: Use playwright to generate the observation dictionary for each site
    - Step 4: Save these observation files as json in a dataset directory

    Args:
        delay (int, optional): Delay in minutes before starting, useful for running multiple instances. Defaults to 0.

    Returns:
        None
    """
    time.sleep(delay*60)
    dl = DownloaderEnv('https://www.google.com')
    random_queries = parse_query_list(random_query_list())
    for rq in random_queries[-5:0:-1]:
        print(rq, flush=True)
        try:
            random_site = fetch_sites_from_google(rq)
            print(random_site, flush=True)
            obs = dl.obs_from_url(random_site)
            dl.page.close()
            json.dump(obs, open(f"data/{(obs['open_pages_titles'][0][:15]).replace(' ','_')}.json", 'w'), indent=4)
        except Exception as e:
            # very unreliable process, so we need to catch exceptions and move on
            print(e, flush=True)
            dl.page.close()
            del dl
            time.sleep(5)
            dl = DownloaderEnv('https://www.google.com')
            continue



if __name__ == "__main__":

    # if you want to pull down a single website's data
    url = "https://www.instagram.com/accounts/login/"
    website_name = 'zzz_instagram_demo'

    dl = DownloaderEnv(url)
    obs = dl.obs_from_url(url)
    dl.page.close()
    json.dump(obs, open(f"data/{website_name}.json", 'w'), indent=4)
