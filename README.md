# Prompt Injection Attacks on Web Navigation Agents

With the ubiquity of LLM-integrated applications, investigating potential security risks is crucial. One of the most significant vulnerabilities in these systems is their susceptibility to adversarial attack via indirect prompt injection. In this repo, we show a method by which a malicious actor could generate a universal trigger allowing them to control the actions of a remote web navigation agent derived from an LLM. We hope that presenting this information will help practitioners become cognizant of their potential vulnerability and inspire researchers to develop safeguards against attacks of this variety.

### Setup

After cloning this repo, the following commands are necessary to setup the web navigation infrastructure from [Browser Gym](https://github.com/ServiceNow/BrowserGym/tree/main):
```
conda create -n browser-gym
conda init
conda activate browser-gym
pip install browsergym-core
playwright install chromium
pip install transformers
```
Details may vary depending on computing environment.

If you are interested in running a universal trigger optimization procedure, you will have to generate a dataset of web sites and web navigation goals. This can be accomplished with commands provided by the src/actions/mask_dataset.py file. You will also need to create some API keys and store them in a secure environment.

1. Run `pip install python-dotenv` and then `touch .env`
2. Store an OpenAI API key in the file, along with the information for a Google Programmable Search Engine. Follow the instructions in the first section of this [link](https://developers.google.com/custom-search/v1/introduction) to setup your search engine. The syntax of your '.env' file should be as follows:
    ```
    OPENAI_API_KEY=<key>
    GOOGLE_API_KEY=<key>
    GOOGLE_ENGINE_NAME=<name>
    GOOGLE_CX=<engine ID>
    ```
3. Run `python -m src.dataset.actions.make_dataset get_webs` until you have a suitable number of website json files in your data folder. For reference, we amassed 10k for our purposes.
4. Run `python -m src.dataset.actions.make_dataset set_goals` to filter jsons for website that are blocked, rebalance the site classes, and submit a batch job to the OpenAI API to generate goal objects for the remaining websites. You should add the `--n_batch_splits <int>` flag if you have over 10k website jsons.
5. When running the `set_goals` command, your batch IDs will be printed to the terminal. Run `python -m src.dataset.actions.make_dataset get_goals --batch_ids <batch ID 1> <batch ID 2> ...` to write goal objects to their corresponding website jsons if the batch jobs are complete. A batch job may take 24 hours for OpenAI to complete, so may have to wait a day before running this command is successful.


### Running Demo 1

Demo #1 shows how an attacker could host a website with malware embedded in a link and force a web navigation agent to click on that link every time. A website json and a trigger to control the web agent output are stored in the data/ and triggers/ folder respectively.

#### Control
Running `python -m src.attack.actions.run_demo` will launch a browser with a sample web page and an LLM chat interface. The LLM chat will ask you to provide a goal for the web navigation agent. Write a task like "Navigate to the contact page" in the window. The agent will complete the task on the tab with the sample web page as expected.

#### Attack
Run `python -m src.attack.actions.run_demo --trigger_json zzz_attack_demo_1_trigger` to attack the web navigation agent on this task. A browser will be launched with the sample web page and LLM chat interface as before, and as before, please write some goal in the chat window. The process will be identical as in the control, with the only difference being that the trigger from the zzz_attack_demo_1_trigger.json file will be inserted into a hidden link on the website. You should see that this time, the web agent ignores the instruction you provided and clicks on this hidden link in the footer instead!

... more demos to come!


### Optimizing your own adversarial trigger

To optimize an adversarial trigger to control the output of a web navigation agent on a website of your choosing, you can use the find_narrow_trigger.py file. Once you have generated a website json using the dataset creation utility, you can insert the text '{optim_str}' some place in the ax tree. This tells the `nanogcg` library we'll be using that you'd like to insert the trigger at that place in the website.

1. Run `pip install nanogcg`
2. Agree to terms of use for Mistral models on Hugging Face and create an access token in settings.
3. Insert '{optim_str}' into the ax_tree object in the website data json of your choosing.
4. Run `python -m src.attack.actions.find_narrow_trigger`. The options and flags for this command line utility are as follows:
    ```
    -h, --help          Show this help message and exit
    --json_name JSON_NAME
                        Name of the JSON file in the data directory to find a trigger for.
    --model {mistral-7B,mistral-24B}
                        The model to use for generation.
    --target TARGET     The desired output when triggered.
    --device DEVICE     Device to run the model on.
    --dtype DTYPE       Data type to use for the model.
    ```
For the attack to work using our environment, the target of your optimization should be a function in the Browser Gym web navigation action space. Refer to the [action_space.txt](https://github.com/sej2020/LLM-Honeypots/blob/main/src/attack/utils/action_space.txt) file.

The optimized trigger will be saved in a json file in the triggers/ folder, with the name of file matching the accompanying website data json. You can use this trigger to run your own demo using the `python -m src.attack.actions.run_demo` utility with appropriate flags.
