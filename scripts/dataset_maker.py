import requests
import random
from dotenv import load_dotenv
import os
from browsergym.core.env import BrowserEnv, Chat
from browsergym.core.task import OpenEndedTask
import playwright.sync_api
from browsergym.core import _get_global_playwright
from browsergym.core.constants import BROWSERGYM_ID_ATTRIBUTE
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
import time
import json
import openai
import random
import multiprocessing as mp
import pathlib

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", api_key)  # Debugging: Ensure it's set
openai.api_key = api_key

prompt_example_options = ['Beef Stroganoff recipe',
    '2022 World Cup schedule',
    'New shoes',
    'What Harry Potter house am I in quiz',
    'Paper mache volcano',
    'Guy sprays himself with pepper spray',
    'Top 10 NFL quarterbacks',
    'One Direction songs',
    'Nietzsche quotes',
    'Faroese Islands',
    'Wikipedia',
    'Forbes',
    'Fortnite',
    '80s trivia',
    'Porcfest',
    'Greek God Family Tree',
    'Washing machine repair',
    ]

prompt_phrasing_options = [
    'Give me a list of random things you can find on the internet',
    'Generate a list of niche hobbies/interests/groups that people have',
    'Guess people/places/things/companies/ideas that I like',
    'What are some random things people search for online?',
    'What are the best web pages that people visit?',
    'Cool websites I can show my friends',
    'What are some interesting things to do online?',
    'What is on the internet?',
    'What do I think about when Im bored',
    'Esoteric pages on the internet',
    'Examples of proper nouns',
    ]

def query_list(n_queries: int = 100):
    # ask gpt to generate a list of random search queries
    # high temperature to get more diverse results
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


def parse_query_list(query_list: str):
    print(query_list.split('\n'))
    return [q.split('. ')[1] for q in query_list.split('\n')[1:] if any(char.isdigit() for char in q)]


def fetch_sites_from_google(query):
    api_key = os.getenv('GOOGLE_API_KEY')
    cx = os.getenv('GOOGLE_CX')
    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'
    response = requests.get(url)
    return random.choice([item['link'] for item in response.json().get('items', [])])

def obs_preprocessor(obs: dict) -> dict:

    return {
        "chat_messages": obs["chat_messages"],
        # "screenshot": obs["screenshot"], # might need to be added back later
        "goal_object": obs["goal_object"],
        "last_action": obs["last_action"],
        "last_action_error": obs["last_action_error"],
        "open_pages_urls": obs["open_pages_urls"],
        "open_pages_titles": obs["open_pages_titles"],
        # "active_page_index": obs["active_page_index"], # might need to be added back later
        "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
        # "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
    }

class DownloaderEnv(BrowserEnv):

    # gym metadata
    metadata = {"render_modes": None}

    def __init__(
        self,
        start_url: str,
    ):
        self.start_url = start_url
        super().__init__(OpenEndedTask, {'start_url': start_url})

        # use the global Playwright instance
        pw: playwright.sync_api.Playwright = _get_global_playwright()
        # important: change playwright's test id attribute from "data-testid" to "bid"
        pw.selectors.set_test_id_attribute(BROWSERGYM_ID_ATTRIBUTE)

        # create a new browser
        self.browser = pw.chromium.launch(
            headless=self.headless,
            # will raise an Exception if above args are overriden
            **self.pw_chromium_kwargs,
        )

        self.context = self.browser.new_context(
            # will raise an Exception if above args are overriden
            **self.pw_context_kwargs,
        )

        self.context.expose_binding(
            "browsergym_page_activated", lambda source: self._activate_page_from_js(source["page"])
        )
        self.context.add_init_script(
            r"""
window.browsergym_page_activated();
window.addEventListener("focus", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("focusin", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("load", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("pageshow", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mousemove", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mouseup", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mousedown", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("wheel", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("keyup", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("keydown", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("input", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("touchstart", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("touchend", () => {window.browsergym_page_activated();}, {capture: true});
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
        window.browsergym_page_activated();
    }
}, {capture: true});
"""
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

    def obs_from_url(self, url):
        self.start_time = time.time()
        self.start_url = url
        self.page = self.context.new_page()
        self.page.goto(self.start_url, timeout=5000)
        self._wait_dom_loaded()
        self._active_page_check()
        obs = self._get_obs()
        obs = obs_preprocessor(obs)
        self.page.close()
        return obs




if __name__ == '__main__':
    # Step 1: Have chatpgt generate 10,000 random short queries
    # Step 2: Use Google Custom Search API to fetch top site url for each query
    # Step 3: Write the urls to a file
    # Step 4: Retrieve the urls and use playwright to generate the observation dictionary for each site
    ## - inherit from BrowserEnv
    ## - implement a new method to setup the playwright browser and then visit each site
    ## - put that through the get_obs framework
    # Step 5: Save these observation files as json in a dataset directory
    # Step 6: Post-process by removing banned websites and balancing classes
    # Step 7: Use chatgpt create goals related to the sites

    # steps 1-5
    def main(i):
        time.sleep(i*60)
        dl = DownloaderEnv('https://www.google.com')
        random_queries = parse_query_list(query_list())
        for rq in random_queries[-5:0:-1]:
            print(rq, flush=True)
            try:
                random_site = fetch_sites_from_google(rq)
                print(random_site, flush=True)
                obs = dl.obs_from_url(random_site)
                dl.page.close()
                json.dump(obs, open(f"data/{(obs['open_pages_titles'][0][:15]).replace(' ','_')}.json", 'w'), indent=4)
            except Exception as e:
                print(e, flush=True)
                dl.page.close()
                del dl
                time.sleep(5)
                dl = DownloaderEnv('https://www.google.com')
                continue

    # run main on 4 processes
    with mp.Pool(4) as pool:
        pool.map(main, range(4))
    