from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlavaForConditionalGeneration, \
    BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from typing import List
from utils.common_util import *
import time
import re
import os
import base64
import io

from PIL import Image
import requests
import torch
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration
from vllm import LLM, SamplingParams

try:
    from transformers import Glm4vForConditionalGeneration

    _GLM4V_AVAILABLE = True
except ImportError:
    _GLM4V_AVAILABLE = False

try:
    from transformers import Glm4vMoeForConditionalGeneration

    _GLM4V_MOE_AVAILABLE = True
except ImportError:
    _GLM4V_MOE_AVAILABLE = False

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model

from typing import Any

device = 'cuda'


def load_model(opt):
    config = AutoConfig.from_pretrained(opt.model_dir)
    model_type = config.model_type.lower()

    tokenizer = AutoTokenizer.from_pretrained(opt.model_dir, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        opt.model_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        offload_folder="offload_dir",
        offload_state_dict=True,
    )
    return tokenizer, model


def load_llama_vl_model(opt):
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     opt.model_dir,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    model = LLM(model=opt.model_dir, tensor_parallel_size=4, dtype="bfloat16", enable_prefix_caching=True,
                enable_chunked_prefill=True)
    # tokenizer = AutoTokenizer.from_pretrained(opt.model_dir)
    return model


def load_llava_model(opt):
    model_path = opt.model_dir
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
    )
    return model, image_processor


def load_qwen_vl_model(opt):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        opt.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(opt.model_dir)
    return model, processor


def load_glm4v_model(opt):
    config = AutoConfig.from_pretrained(opt.model_dir)
    model_type = (getattr(config, "model_type", "") or "").lower()

    if model_type == "glm4v_moe":
        if not _GLM4V_MOE_AVAILABLE:
            raise ImportError(
                "GLM-4.6V-FP8 (glm4v_moe) requires transformers with Glm4vMoeForConditionalGeneration. "
                "Upgrade with: pip install -U transformers"
            )
        model = Glm4vMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=opt.model_dir,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        if not _GLM4V_AVAILABLE:
            raise ImportError(
                "GLM-4.6V requires transformers>=5.0.0rc0 with Glm4vForConditionalGeneration. "
                "Install with: pip install 'transformers>=5.0.0rc0'"
            )
        model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=opt.model_dir,
            torch_dtype="auto",
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(opt.model_dir)
    return model, processor


def get_glm4v_response(messages, image_file, model, processor):
    """GLM-4.6V multimodal inference (image + text). Used for MLM and MIX."""
    if image_file is not None:
        if image_file.startswith(("http://", "https://")):
            url = image_file
        else:
            path = os.path.abspath(image_file)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"GLM image not found: {path}")
            buf = io.BytesIO()
            Image.open(path).convert("RGB").save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        content = [
            {"type": "image", "url": url},
            {"type": "text", "text": messages},
        ]
    else:
        content = [{"type": "text", "text": messages}]
    chat = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    else:
        inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            top_p=0.6,
            top_k=2,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    seq_len = inputs["input_ids"].shape[1]
    output_text = processor.decode(
        generated_ids[0][seq_len:], skip_special_tokens=False
    )
    return output_text.strip()


def _glm4v_content(item):
    """Ensure content is list of {type, text} dicts for GLM apply_chat_template."""
    if isinstance(item, list) and item and isinstance(item[0], dict) and "type" in item[0]:
        return item
    s = item if isinstance(item, str) else str(item)
    return [{"type": "text", "text": s}]


def _strip_glm_special_tokens(text: str) -> str:
    """Remove GLM-4 special tokens like <|end_of_box|><|user|> from model output."""
    if not text or not isinstance(text, str):
        return text
    s = re.sub(r"<\|[^|]*\|>", "", text)
    return s.strip()


def _normalize_glm_final_answer(text: str) -> str:
    """Normalize GLM final answer: strip special tokens and remove label prefixes (e.g. Year_1955 -> 1955)."""
    s = _strip_glm_special_tokens(text)
    if not s:
        return s
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        q = re.sub(r"^\w+_", "", p).strip()
        if q:
            out.append(q)
    return ", ".join(out) if out else s


def get_glm4v_text_response(messages: List[str], model, processor):
    """GLM-4.6V text-only inference. Used for LLM task."""
    chat = []
    for i, msg in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        chat.append({"role": role, "content": _glm4v_content(msg)})
    inputs = processor.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    else:
        inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            top_p=0.6,
            top_k=2,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    seq_len = inputs["input_ids"].shape[1]
    output_text = processor.decode(
        generated_ids[0][seq_len:], skip_special_tokens=False
    )
    return output_text.strip()


def get_llm_response(messages: List[str], tokenizer, model):
    device = next(model.parameters()).device
    model.eval()
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": messages[i]} for i in range(len(messages))]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_input = tokenizer([text], return_tensors='pt').to(device)
    device = next(model.parameters()).device
    model_input = model_input.to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=4096,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.6,
        use_cache=True
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def get_llama_vl_response(messages, image_file, model, tokenizer):
    prompt = "USER: <image>\n" + messages + "\nASSISTANT"
    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }
    sampling_params = SamplingParams(temperature=0, max_tokens=32000, stop_token_ids=[tokenizer.eos_token_id])
    outputs = model.generate(inputs, sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def get_llava_response(prompt, image_file, tokenizer, model, image_processor, opt):
    model_path = opt.model_dir
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0.2,
        "top_p": None,
        "num_beams": 1,
        'max_new_tokens': 4096
    })()
    response = eval_model(args, tokenizer, model, image_processor)
    return response


def get_qwen_vl_response(messages, image_file, model, processor):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": messages},
                {
                    "type": "image",
                }
            ],
        }
    ]
    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text


def get_tablellm_response(messages: List[str], tokenizer, model):
    model.eval()
    prompt = messages[0]

    model_input = tokenizer([prompt], return_tensors='pt').to(device)
    device = next(model.parameters()).device
    model_input = model_input.to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=1024,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.001
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response


def get_tablellama_answer(messages: List[str], tokenizer, model, query):
    model.eval()
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input_seg}\n\n### Question:\n{question}\n\n### Response:"
        ),
    }
    latex_file_path = os.path.abspath(f'../data/latex')

    question_prompt = PROMPT_DICT["prompt_input"].format(instruction=messages[0], input_seg="",
                                                         question=query['Question'])
    input_part1 = len(tokenizer.encode(question_prompt)) + 100
    table_content = read_file(f'{latex_file_path}/{query["FileName"]}.txt')
    truncated_table_tokens = tokenizer.encode(table_content, max_length=8192 - input_part1, truncation=True,
                                              add_special_tokens=False)
    truncated_table_content = tokenizer.decode(truncated_table_tokens, skip_special_tokens=True)
    prompt = PROMPT_DICT["prompt_input"].format(instruction=messages[0], input_seg=truncated_table_content,
                                                question=query['Question'])
    model_input = tokenizer(prompt, return_tensors='pt').to(device)
    device = next(model.parameters()).device
    model_input = model_input.to(device)
    output = model.generate(
        **model_input,
        max_new_tokens=1024
    )
    out = tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    response = out.split(prompt)[1].strip().split("</s>")[0]
    return response


def get_final_answer(messages, answer_format, tokenizer, model, sleep_time=5, max_retry=1):
    retry = 0
    while retry < max_retry:
        response = get_llm_response(messages, tokenizer, model)
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
            messages[
                len(messages) - 1] += '\nNote: Please check your output format. You should follow the given format: "[Final Answer]: '
            time.sleep(sleep_time)
        if retry == max_retry: return ""


def get_glm4v_text_final_answer(messages, answer_format, model, processor, sleep_time=5, max_retry=1):
    """GLM-4.6V text-only final-answer loop for LLM task."""
    retry = 0
    while retry < max_retry:
        response = get_glm4v_text_response(messages, model, processor)
        messages.append(response)
        if "[Final Answer]:" in response:
            out = response.split("[Final Answer]:")[-1].strip()
            return _normalize_glm_final_answer(out)
        if "Final Answer:" in response:
            out = response.split("Final Answer:")[-1].strip()
            return _normalize_glm_final_answer(out)
        retry += 1
        print("No 'Final Answer' found, requesting again...")
        messages[-1] += '\nNote: Please check your output format. You should follow the given format: "[Final Answer]: '
        time.sleep(sleep_time)
    return ""


def get_multimodal_final_answer(messages, image_file, answer_format, tokenizer, model, image_processor, opt,
                                sleep_time=5, max_retry=1):
    retry = 0
    while retry < max_retry:
        config = AutoConfig.from_pretrained(opt.model_dir)
        model_type = config.model_type.lower()

        response: Any = ""

        if model_type == 'llama3_2_vl':
            response = get_llama_vl_response(messages, image_file, model, tokenizer)
        elif model_type == 'llava':
            response = get_llava_response(messages, image_file, tokenizer, model, image_processor, opt)
        elif model_type == 'qwen2_vl':
            response = get_qwen_vl_response(messages, image_file, model, image_processor)
        elif model_type in ('glm4v', 'glm4v_moe'):
            response = get_glm4v_response(messages, image_file, model, image_processor)
        else:
            raise ValueError(
                f"The model-specific loading script has not yet been configured; please consult the model's documentation.")

        # response = get_llava_response(messages, image_file, tokenizer, model, image_processor, opt)
        # 判断响应中是否包含 "Final Answer"
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
            if model_type in ("glm4v", "glm4v_moe"):
                final_answer = _normalize_glm_final_answer(final_answer)
            return final_answer
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            if model_type in ("glm4v", "glm4v_moe"):
                final_answer = _normalize_glm_final_answer(final_answer)
            return final_answer
        else:
            retry += 1
            print("No 'Final Answer' found, requesting again...")
            messages = messages + '\n Note: Please check your output format. You do not need to do much explaining, just give the final answer in the given format: "[Final Answer]: ' + answer_format
            time.sleep(sleep_time)
        if retry == max_retry:
            return _normalize_glm_final_answer(response) if model_type in ("glm4v", "glm4v_moe") else response


def get_final_answer_tablellama(messages, answer_format, tokenizer, model, query, sleep_time=5, max_retry=5):
    retry = 0
    while retry < max_retry:
        response = get_tablellama_answer(messages, tokenizer, model, query)
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
            messages.append(
                'Note: Please check your output format. You do not need to do much explaining, just give the final answer in the given format: "[Final Answer]: ' + answer_format)
            time.sleep(sleep_time)
        if retry == max_retry: return response