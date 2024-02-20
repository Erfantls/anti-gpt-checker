import json
from openai import OpenAI

from config import OPENAI_API_KEY, DEFAULT_EMAIL_GENERATION_PROMPT_NAME, OPENAI_GPT3_5_MODEL_NAME

# Replace YOUR_API_KEY with your actual OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_email(subject:str, language_code: str, engine:str):
    # Determine the language based on the language code
    match language_code:
        case 'en':
            language = "English"
        case 'pl':
            language = "Polish"
        case _:
            raise ValueError(f"Language {language_code} is not supported")

    prompt_text = _load_prompt_from_json_and_format(DEFAULT_EMAIL_GENERATION_PROMPT_NAME, subject, language)
    try:
        response = client.chat.completions.create(
            model=engine,
            messages=[
                {"role": "user", "content": prompt_text},
                #{"role": "user", "content": subject}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def _load_prompt_from_json_and_format(prompt_name: str, subject: str, language: str) -> str:
    with open('data/email_prompts.json', 'r', encoding='utf-8') as file:
        prompts = json.load(file)
        prompt_template = prompts[f"{language}_{prompt_name}"]
        return prompt_template.format(subject=subject)

