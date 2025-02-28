
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

init_script = r"""
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