import json
from typing import List, Optional
import re

from openai import OpenAI

from config import OPENAI_API_KEY, DEFAULT_EMAIL_GENERATION_PROMPT_NAME, OPENAI_GPT3_5_MODEL_NAME
from models.email import EmailTone, EmailInfoForGPT

# Replace YOUR_API_KEY with your actual OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_email_based_on_og_email(email_subject: str, email_text: str, language_code: str, engine: str = OPENAI_GPT3_5_MODEL_NAME) -> Optional[str]:
    basic_info = get_basic_info_for_generation(email_text, language_code, engine)
    return generate_email(email_subject, basic_info, language_code, engine)

def generate_email(subject:str, email_info_for_gpt: EmailInfoForGPT, language_code: str, engine:str = OPENAI_GPT3_5_MODEL_NAME) -> Optional[str]:
    _validate_lang_code(language_code)

    system_prompt_text = _load_prompt_from_json(DEFAULT_EMAIL_GENERATION_PROMPT_NAME, language_code)
    user_prompt_text = email_info_for_gpt.to_prompt(subject, language_code)

    messages = [
        {"role": "system",
         "content": system_prompt_text},
        {"role": "user",
         "content": user_prompt_text}
    ]
    response_txt = _request_to_openai_api(messages, engine)

    if_finished = _ensure_finished(response_txt)
    no_placeholders = _ensure_no_placeholders(response_txt)
    if not if_finished or not no_placeholders:
        response_txt = fix_generated_email(response_txt, messages, language_code, if_finished, no_placeholders, engine)
        if isinstance(response_txt, tuple):
            return (response_txt[0], response_txt[1].replace("===FINISHED===", "").strip())

    return response_txt.replace("===FINISHED===", "").strip()


def fix_generated_email(generated_email_text: str,
                        earlier_messages: List[dict],
                        language_code: str,
                        is_finished: bool,
                        no_placeholders: bool,
                        engine: str = OPENAI_GPT3_5_MODEL_NAME) -> Optional[str]:
    _validate_lang_code(language_code)
    fix_prompt = _load_prompt_from_json("fix_template", language_code)
    if not is_finished:
        fix_prompt += f"\n{_load_prompt_from_json('fix_finished_tag', language_code)}"
    if not no_placeholders:
        fix_prompt += f"\n{_load_prompt_from_json('fix_placeholders', language_code)}"
    messages = earlier_messages + [
        {"role": "assistant",
         "content": generated_email_text},
        {"role": "user",
         "content": fix_prompt}
    ]
    response_txt = _request_to_openai_api(messages, engine)
    if not _ensure_finished(response_txt):
        print("RESPONSE NOT FINISHED")
        print(response_txt)
        raise ValueError("The response is not finished")
    if not _ensure_no_placeholders(response_txt):
        # print("RESPONSE WITH PLACEHOLDERS")
        # print(response_txt)
        # raise ValueError("The response contains placeholders")
        return (False, response_txt)
    return response_txt


def get_basic_info_for_generation(email_text: str, language_code: str, engine: str = OPENAI_GPT3_5_MODEL_NAME) -> EmailInfoForGPT:
    _validate_lang_code(language_code)
    messages = [
        {"role": "system",
         "content": _load_prompt_from_json("get_info", language_code)},
        {"role": "user", "content": email_text}
    ]
    response_txt = _request_to_openai_api(messages, engine)
    summary = None
    tone = None
    length = len(email_text.split(" "))

    is_finished = _ensure_finished(response_txt)
    if not is_finished:
        messages.append({"role": "assistant", "content": response_txt})
        fix_is_finished_prompt = f"{_load_prompt_from_json('fix_template', language_code)}\n" \
                                 f"{_load_prompt_from_json('fix_finished_tag', language_code)}"
        messages.append({"role": "user", "content": fix_is_finished_prompt})
        response_txt = _request_to_openai_api(messages, engine)
        if not _ensure_finished(response_txt):
            raise ValueError("The response is not finished")


    summary_started = False
    tone_started = False
    # Iterate through each line to extract summary and tone
    for line in response_txt.split('\n'):
        if line == '==SUMMARY==':
            summary_started = True
            summary = ''
        elif line == '==TONE==':
            summary_started = False
            tone_started = True
        elif line == '==FINISHED==':
            tone_started = False
        elif summary_started:
            summary += line.strip() + ' '
        elif tone_started and tone is None:
            try:
                tone = EmailTone[line.strip().upper()]
            except KeyError:
                raise ValueError(f"Invalid tone: {line.strip().upper()}")

    if summary is None or tone is None:
        print(response_txt)
        raise ValueError("Summary or tone not found in the response")

    return EmailInfoForGPT(summary=summary.strip(), length=length, tone=tone)





def _request_to_openai_api(messages: List[dict], engine:str = OPENAI_GPT3_5_MODEL_NAME) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=engine,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise ConnectionError(f"Error while connecting to OpenAI API: {e}")


def _load_prompt_from_json(prompt_name: str, lang_code: str) -> str:
    with open('../data/email_prompts.json', 'r', encoding='utf-8') as file:
        prompts = json.load(file)
        return prompts[f"{lang_code}_{prompt_name}"]


def _validate_lang_code(language_code):
    # Determine the language based on the language code
    match language_code:
        case 'en':
            language = "English"
        case 'pl':
            language = "Polish"
        case _:
            raise ValueError(f"Language {language_code} is not supported")


def _ensure_finished(response_txt: str) -> bool:
    if response_txt.strip().split('\n')[-1] != '===FINISHED===':
        return False
    return True

def _ensure_no_placeholders(text) -> bool:
    pattern = r"\[[^\]]*\]"
    return not bool(re.search(pattern, text))