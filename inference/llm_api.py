from typing import List
from openai import OpenAI
from utils.common_util import *
import re
import time

def get_llm_response(messages: List[str], opt):
    # Only pass api_key and base_url if they are not empty
    client_kwargs = {}
    if opt.base_url:
        client_kwargs['base_url'] = opt.base_url
        client_kwargs['api_key'] = opt.api_key or 'dummy'  # vLLM often allows any key
    elif opt.api_key:
        client_kwargs['api_key'] = opt.api_key
    client = OpenAI(**client_kwargs)
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": messages[i]} for i in range(len(messages))]
    create_kwargs = {"messages": messages, "max_tokens": 4096}
    # When using vLLM, server model id may differ from --model; use --vllm_model if set
    model_for_api = getattr(opt, "vllm_model", None) or getattr(opt, "model", None)
    if model_for_api:
        create_kwargs["model"] = model_for_api
    # vLLM / local server: optional sampling (match local GLM behavior)
    if getattr(opt, "temperature", None) is not None:
        create_kwargs["temperature"] = opt.temperature
    if getattr(opt, "top_p", None) is not None:
        create_kwargs["top_p"] = opt.top_p
    chat_completion = client.chat.completions.create(**create_kwargs)
    content = chat_completion.choices[0].message.content
    return content if content is not None else ""

def get_mlm_response(messages, image_file, opt):
    client_kwargs = {}
    if opt.base_url:
        client_kwargs['base_url'] = opt.base_url
        client_kwargs['api_key'] = opt.api_key or 'dummy'
    elif opt.api_key:
        client_kwargs['api_key'] = opt.api_key
    client = OpenAI(**client_kwargs)
    image = encode_image(image_file)
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ]
    create_kwargs = {"messages": messages, "temperature": 0, "max_tokens": 4096}
    model_for_api = getattr(opt, "vllm_model", None) or getattr(opt, "model", None)
    if model_for_api:
        create_kwargs["model"] = model_for_api
    chat_completion = client.chat.completions.create(**create_kwargs)
    content = chat_completion.choices[0].message.content
    return content if content is not None else ""

def get_mlm_response_multi(messages, image_file, opt):
    client_kwargs = {}
    if opt.base_url:
        client_kwargs['base_url'] = opt.base_url
        client_kwargs['api_key'] = opt.api_key or 'dummy'
    elif opt.api_key:
        client_kwargs['api_key'] = opt.api_key
    client = OpenAI(**client_kwargs)
    image = encode_image(image_file)
    if len(messages) ==1 : messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages[0]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }]
    else: messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages[0]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content":  messages[1]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages[2]}
                ]
            }
        ]
    model_for_api = getattr(opt, "vllm_model", None) or getattr(opt, "model", None)
    create_kwargs = {"messages": messages, "temperature": 0, "max_tokens": 4096}
    if model_for_api:
        create_kwargs["model"] = model_for_api
    chat_completion = client.chat.completions.create(**create_kwargs)
    content = chat_completion.choices[0].message.content
    return content if content is not None else ""

def get_final_answer(messages, answer_format, opt, sleep_time=5, max_retry=1):
    """Align with llm_local.get_final_answer: append response, retry note on last msg, return '' when exhausted."""
    retry = 0
    while retry < max_retry:
        response = get_llm_response(messages, opt)
        if response is None:
            response = ""
        messages.append(response)
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
            return final_answer
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            return final_answer
        else:
            retry += 1
            print("No 'Final Answer' found, requesting again...")
            messages[len(messages) - 1] += '\nNote: Please check your output format. The ouput format should be: [Final Answer]: "' + answer_format
            time.sleep(sleep_time)
        if retry == max_retry:
            return ""

def get_final_answer_mlm(messages, answer_format, image_file, opt, sleep_time=5, max_retry=1):
    """Align with local: normalize None to '', return '' when retry exhausted (so query gets 0 score, not skipped)."""
    retry = 0
    while retry < max_retry:
        response = get_mlm_response(messages, image_file, opt)
        if response is None:
            response = ""
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
            return final_answer
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            return final_answer
        else:
            retry += 1
            print("No 'Final Answer' found, requesting again...")
            messages += '\n Note: Please check your output format. You do not need to do much explaining, just give the final answer in the given format: "[Final Answer]: '+ answer_format
            time.sleep(sleep_time)
        if retry == max_retry:
            return ""


