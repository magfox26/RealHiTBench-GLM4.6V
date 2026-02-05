import sys
import os

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)
# Fallback: if utils not under _base, use cwd (run from project root)
if not os.path.isdir(os.path.join(_base, "utils")):
    _cwd = os.getcwd()
    if os.path.isdir(os.path.join(_cwd, "utils")) and _cwd not in sys.path:
        sys.path.insert(0, _cwd)

import concurrent.futures
import datetime
import json
import ast
import argparse
import pandas as pd
from tqdm import tqdm
import base64
import time
from answer_prompt_llm import Answer_Prompt, Model_First_Response, User_Prompt
from llm_local import *
from llm_local import _normalize_glm_final_answer
from llm_api import get_final_answer as get_final_answer_api
# GPT evaluation removed - no OpenAI API access needed
# from gpt_eval_prompt import Eval_Prompt
# from gpt_eval import get_eval_score
from transformers import AutoConfig, AutoTokenizer
from metrics.qa_metrics import QAMetric
from utils.common_util import *
from utils.chart_process import *
from utils.chart_metric_util import *
from matplotlib import pyplot as plt
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_extensions = {
    "latex": "txt",
    "markdown": "md",
    "csv": "csv",
    "html": "html",
    "json": "json",
}


@timeout(15)
def execute(c):
    exec(c)


@timeout(20)
def exec_and_get_y_reference(answer_code, chart_type):
    """与 Qwen3-VL 一致：空判断 python_code == \"\"，异常只打印 e，无 _format_exec_error / traceback。"""
    ecr_1 = False
    python_code, eval_code = build_eval_code(answer_code, chart_type)
    print("Code:", python_code)
    if not (python_code and python_code.strip()):
        return "", False
    try:
        python_code = surround_pycode_with_main(python_code)
        execute(python_code)
        plt.close("all")
        ecr_1 = True
    except Exception as e:
        print("Python Error:", e)
        ecr_1 = False
        return "", False
    if ecr_1:
        pass
    try:
        from io import StringIO
        output = StringIO()
        stdout = sys.stdout
        try:
            sys.stdout = output
            chart_eval_code = surround_pycode_with_main(eval_code)
            execute(chart_eval_code)
        except Exception as e:
            print("Eval Error:", e)
            return "", True
        finally:
            sys.stdout = stdout
        output_value = output.getvalue()
        print("OUTPUT VALUE: ", output_value)
    except Exception as e:
        print("Eval Error:", e)
        output_value = ''

    if output_value != '':
        parsed_prediction = output_value.strip()
    else:
        parsed_prediction = ''
    plt.close('all')
    return parsed_prediction, ecr_1


def build_messages(query, answer_format, opt):
    file_format = opt.format
    _dr = getattr(opt, 'data_root', None)
    file_path = os.path.join(_dr, file_format) if _dr else os.path.abspath(f'../data/{file_format}')
    question_type = query['QuestionType']
    if question_type == 'Data Analysis':
        PROMPT_FORMAT = Answer_Prompt[query['SubQType']].format(format=opt.format)
    else:
        PROMPT_FORMAT = Answer_Prompt[question_type].format(format=opt.format)
    messages = [PROMPT_FORMAT]
    messages.append(Model_First_Response)
    prompt = ""
    prompt = User_Prompt.format_map({
        'Question': query['Question'],
        'Table': read_file(f'{file_path}/{query["FileName"]}.{file_extensions[opt.format]}'),
        "Answer_format": answer_format
    })
    messages.append(prompt)
    return messages


def build_messages_truncated(query, answer_format, tokenizer, opt):
    file_format = opt.format
    _dr = getattr(opt, 'data_root', None)
    file_path = os.path.join(_dr, file_format) if _dr else os.path.abspath(f'../data/{file_format}')
    question_type = query['QuestionType']
    if question_type == 'Data Analysis':
        PROMPT_FORMAT = Answer_Prompt[query['SubQType']].format(format=opt.format)
    else:
        PROMPT_FORMAT = Answer_Prompt[question_type].format(format=opt.format)
    messages = [PROMPT_FORMAT]
    messages.append(Model_First_Response)
    enc = tokenizer(PROMPT_FORMAT + Model_First_Response + query['Question'] +
                    "Emphasize: you need to make sure your final answer is formatted in this way: [Final Answer]:" + answer_format,
                    return_tensors='pt')
    input_part1 = enc.input_ids.shape[1] + 512
    prompt = ""
    table_content = read_file(f'{file_path}/{query["FileName"]}.{file_extensions[opt.format]}')
    max_table_tokens = opt.max_input - input_part1
    if max_table_tokens <= 0:
        max_table_tokens = 1024
    truncated_table_tokens = tokenizer.encode(table_content, max_length=max_table_tokens, truncation=True,
                                              add_special_tokens=False)
    was_truncated = len(truncated_table_tokens) >= max_table_tokens
    if was_truncated:
        print(f"[LLM] query id={query.get('id')} table TRUNCATED: max_input={opt.max_input}, table limited to {max_table_tokens} tokens.")
    truncated_table_content = tokenizer.decode(truncated_table_tokens, skip_special_tokens=True)
    prompt = User_Prompt.format_map({
        'Question': query['Question'],
        'Table': truncated_table_content,
        "Answer_format": answer_format
    })
    messages.append(prompt)
    return messages, was_truncated


def get_answer_format(query):
    answer_format = ""
    if query['SubQType'] == 'Exploratory Analysis':
        answer_format = "CorrelationRelation, CorrelationCoefficient"
    elif query['QuestionType'] == 'Visualization':
        answer_format = "import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show()"
    else:
        answer_format = "AnswerName1, AnswerName2..."
    return answer_format


def gen_solution(opt):
    start_time = datetime.datetime.now()

    _data_root = os.path.abspath(opt.data_dir) if opt.data_dir else os.path.abspath("../data")
    opt.data_root = _data_root
    with open(os.path.join(_data_root, 'QA_final.json'), 'r') as fp:
        dataset = json.load(fp)
        querys = dataset['queries']
    if opt.max_queries > 0:
        querys = querys[: opt.max_queries]
    # Optional: run only a specific task type (e.g. only Visualization for debugging)
    if getattr(opt, 'task_filter', None) and str(opt.task_filter).strip().lower() not in ('', 'all'):
        _task = str(opt.task_filter).strip()
        _type_map = {
            'visualization': 'Visualization',
            'fact checking': 'Fact Checking',
            'numerical reasoning': 'Numerical Reasoning',
            'data analysis': 'Data Analysis',
            'structure comprehending': 'Structure Comprehending',
        }
        _task = _type_map.get(_task.lower(), _task)
        querys = [q for q in querys if q['QuestionType'] == _task]
        print(f"Task filter: only running '{_task}' ({len(querys)} queries)")

    output_file_path = os.path.abspath(f'../result')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
        os.chmod(output_file_path, 0o777)

    output_file_path = f'{output_file_path}/open_source/{opt.model}'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
        os.chmod(output_file_path, 0o777)

    file_path = os.path.join(_data_root, opt.format)
    table_file_path = os.path.join(_data_root, "tables")

    all_eval_results = []
    truncated_queries = []  # list of {"id", "format"} for prompts that were truncated
    config = AutoConfig.from_pretrained(opt.model_dir)
    model_type = config.model_type.lower()
    use_vllm = bool(getattr(opt, 'base_url', None))

    if use_vllm:
        # vLLM/OpenAI API: load only tokenizer for truncation, no local model
        tokenizer = AutoTokenizer.from_pretrained(opt.model_dir, use_fast=False)
        model, processor = None, None
        use_glm4v = model_type in ('glm4v', 'glm4v_moe')

        def _get_final_answer(messages, answer_format):
            raw = get_final_answer_api(messages, answer_format, opt)
            return _normalize_glm_final_answer(raw) if use_glm4v else raw
    else:
        if model_type in ('glm4v', 'glm4v_moe'):
            model, processor = load_glm4v_model(opt)
            tokenizer = processor.tokenizer
            use_glm4v = True
        else:
            tokenizer, model = load_model(opt)
            processor = None
            use_glm4v = False

        def _get_final_answer(messages, answer_format):
            if use_glm4v:
                return get_glm4v_text_final_answer(messages, answer_format, model, processor)
            return get_final_answer(messages, answer_format, tokenizer, model)

    for query in tqdm(querys):
        try:
            print("----------------------------- Current query: {} --------------------------".format(query['id']))
            question_type = query['QuestionType']
            answer_format = get_answer_format(query)
            messages, truncated = build_messages_truncated(query, answer_format, tokenizer, opt)
            if truncated:
                truncated_queries.append({"id": query["id"], "format": opt.format})
            metric_scores = {}
            if question_type == 'Visualization':
                response = _get_final_answer(messages, answer_format)
                reference = query['ProcessedAnswer']
                chart_type = query['SubQType'].split()[0]
                full_xlsx_path = os.path.join(table_file_path, query['FileName'] + ".xlsx")
                python_code = re.sub(r"'[^']*\.xlsx'", "'" + full_xlsx_path.replace("\\", "/") + "'", response)
                python_code = re.sub(r'"[^"]*\.xlsx"', '"' + full_xlsx_path.replace("\\", "/") + '"', python_code)
                python_code = python_code.replace("table.xlsx", full_xlsx_path)
                prediction, ecr_1 = exec_and_get_y_reference(python_code, chart_type)
                metric_scores['ECR'] = ecr_1
                if prediction != '':
                    try:
                        prediction = ast.literal_eval(prediction)
                        reference = ast.literal_eval(reference)
                        if chart_type == 'PieChart':
                            metric_scores['Pass'] = compute_pie_chart_metric(reference, prediction)
                        else:
                            metric_scores['Pass'] = compute_general_chart_metric(reference, prediction)
                    except Exception as e:
                        metric_scores['Pass'] = False
                else:
                    metric_scores['Pass'] = False
            else:

                reference = query['FinalAnswer']
                if question_type == 'Data Analysis':
                    response = _get_final_answer(messages, answer_format)
                    prediction = response
                    # GPT evaluation removed - no OpenAI API access needed
                    metric_scores = qa_metric.compute([reference], [prediction])
                    metric_scores['GPT_EVAL'] = ''
                elif question_type == 'Structure Comprehending':
                    # QA_final 中 SC 的 FinalAnswer 为空，改为第一遍 vs 第二遍（一致性），分数才有意义
                    messages, tr1 = build_messages_truncated(query, answer_format, tokenizer, opt)
                    if tr1:
                        truncated_queries.append({"id": query["id"], "format": opt.format})
                    response_first = _get_final_answer(messages, answer_format)  # first pass（原表）
                    query["FileName"] = query["FileName"] + "_swap"
                    messages, tr2 = build_messages_truncated(query, answer_format, tokenizer, opt)
                    if tr2:
                        truncated_queries.append({"id": query["id"], "format": opt.format})
                    response = _get_final_answer(messages, answer_format)  # second pass（_swap 表）
                    prediction = response
                    metric_scores = qa_metric.compute([response_first], [prediction])  # 第一遍 vs 第二遍（一致性）
                else:
                    response = _get_final_answer(messages, answer_format)
                    prediction = response
                    metric_scores = qa_metric.compute([reference], [prediction])

            eval_result = {
                'Id': query['id'],
                'QuestionType': query['QuestionType'],
                'Model_Answer': response,
                'Correct_Answer': query['FinalAnswer'],
                'F1': metric_scores.get('F1', ''),
                'EM': metric_scores.get('EM', ''),
                'ROUGE-L': metric_scores.get('ROUGE-L', ''),
                'SacreBLEU': metric_scores.get('SacreBLEU', ''),
                'GPT_EVAL': metric_scores.get('GPT_EVAL', ''),
                'ECR': metric_scores.get('ECR', ''),
                'Pass': metric_scores.get('Pass', '')
            }
            print(eval_result)
            all_eval_results.append(eval_result)
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()

    print(all_eval_results)
    print(output_file_path)
    end_time = datetime.datetime.now()
    print(f"Total time taken: {end_time - start_time}")

    df = pd.DataFrame(all_eval_results)
    df.to_csv(f'{output_file_path}/{opt.model}_text_{opt.format}_{end_time}.csv', index=False)

    if truncated_queries:
        trunc_df = pd.DataFrame(truncated_queries)
        trunc_path = os.path.join(output_file_path, f"truncated_llm_{opt.format}_{end_time}.csv")
        trunc_df.to_csv(trunc_path, index=False)
        print(f"[LLM] Saved {len(truncated_queries)} truncated query ids to {trunc_path}")


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")

    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--model_dir', type=str, default='', help='input file path')
    parser.add_argument('--max_input', type=int, default=128000, help='max input tokens (default 128k for GLM-4.6V). Truncate table if longer to avoid OOM.')
    parser.add_argument('--api_key', type=str, default="", help='the api key of model')
    parser.add_argument('--base_url', type=str, default="", help='OpenAI-compatible API base URL (e.g. http://localhost:8000/v1 for vLLM); if set, use API instead of loading model locally')
    parser.add_argument('--vllm_model', type=str, default="", help='model id sent to the API when using --base_url (default: use --model). If you get 404, set this to the "id" from: curl BASE_URL/models')
    parser.add_argument('--format', type=str, default='latex', choices=['csv', 'html', 'json', 'latex', 'markdown'],
                        help='table format (data/{format}/)')
    parser.add_argument('--data_dir', type=str, default='',
                        help='data root (default: ../data). e.g. /ltstorage/home/liu/RealHiTBench/data')
    parser.add_argument('--max_queries', type=int, default=0, help='max queries to run (0 = all). use 1 for quick test')
    parser.add_argument('--task_filter', type=str, default='',
                        help='Run only this task type. Leave empty or "all" for all tasks. '
                             'e.g. Visualization (only chart tasks), Fact Checking, Data Analysis, etc.')
    opt = parser.parse_args()

    if not opt.model_dir:
        parser.error('--model_dir is required (path to the model, used for tokenizer/config even with --base_url)')
    if opt.base_url and not opt.model and not getattr(opt, 'vllm_model', ''):
        parser.error('when using --base_url, set either --model or --vllm_model (vLLM model id from curl BASE_URL/models)')
    if opt.base_url and not opt.data_dir:
        parser.error('--data_dir is required')
    if opt.base_url and not opt.model and getattr(opt, 'vllm_model', ''):
        opt.model = opt.vllm_model  # use for output path

    return opt


if __name__ == '__main__':
    opt = parse_option()
    print(opt)
    qa_metric = QAMetric()

    gen_solution(opt)
