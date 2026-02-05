---
license: cc-by-nc-4.0
task_categories:
- table-question-answering
language:
- en
---

# Dataset Card of RealHiTBench

<p align="left">
    <a href="https://arxiv.org/abs/2506.13405">📖Paper</a>  <a href="https://github.com/cspzyy/RealHiTBench">⌨️Code</a>
</p>



## Dataset Summary

**RealHiTBench** is a challenging benchmark designed to evaluate the ability of large language models (LLMs) and multimodal LLMs to understand and reason over complex, real-world **hierarchical tables**. It features diverse question types and input formats—including *LaTeX*, *HTML*, and *PNG*—across 24 domains, with 708 tables and 3,752 QA pairs. Unlike existing datasets that focus on flat structures, RealHiTBench includes rich structural complexity such as nested sub-tables and multi-level headers, making it a comprehensive resource for advancing table understanding in both text and visual modalities.



## Data Introduction

Our dataset consists of two major parts, **hierarchical tables** and corresponding **Question-Answer (QA) pairs**.

- **Tables**: We collect complex structural tables from various sources and convert them into multiple file formats (stored in corresponding folders).
- **QA pairs**: We designed diverse question types to thoroughly evaluate LLMs' understanding of hierarchical tables across tasks. When generating reference answers, we used prompt engineering to document the Chain-of-Thought process for annotation. To minimize bias, we also conducted final regularization and meticulous manual checks on the annotated QA pairs.

To better align with real-world applications, the data are collected from multiple open public platforms. More details about the data seen in the paper.

### QA Fields

| Field           | Type       | Description                                                  |
| --------------- | ---------- | ------------------------------------------------------------ |
| id              | int        | Identifier for each QA                                       |
| FileName        | string     | File name of the corresponding table                         |
| CompStrucCata   | string     | Complex structure catagory of the table                      |
| Source          | string     | Collected source of the table                                |
| Question        | string     | Content of the question                                      |
| QuestionType    | string     | Type of the question                                         |
| SubQType        | string     | Subtype of the question                                      |
| COT             | list[json] | Chain-of-Thought from question to answer during the annotation process |
| FinalAnswer     | string     | Result answer from annotation                                |
| ProcessedAnswer | string     | Available answer after processing                            |

### QA Examples

```json
{
            "id": 974,
            "FileName": "economy-table67",
            "CompStrucCata": "SingleRowClassified",
            "Source": "ExcelGuru",
            "Question": "What are the components listed in the “Employer's Contribution” section of the table?",
            "QuestionType": "Fact Checking",
            "SubQType": "Value-Matching",
            "COT": [
                {
                    "step": "First, locate the “Employer's Contribution” section in the table."
                },
                {
                    "step": "Identify all the rows under this section to list the components."
                },
                {
                    "step": "Extract the names of all the components mentioned in this section."
                }
            ],
            "FinalAnswer": "Gratuity, Employer's Provident Fund Contribution, Employer's ESI Contribution, Medical Insurance, Performance Incentive",
            "ProcessedAnswer": "Gratuity, Employer's Provident Fund Contribution, Employer's ESI Contribution, Medical Insurance, Performance Incentive"
        }
```

## Something to Clarify

- RealHiTBench can conduct a comprehensive and challenging evaluation of your model's ability to understand hierarchical tables, whether it is a language or a multimodal model.
- We recommend using *LaTeX*-formatted tables as text-based inputs in experiments due to their comprehensive presentation. Anyway, we also provide other textual formats for your potential use.
- Due to the model's input length restrictions, we've separated some overly large tables and their QA from the original data and haven't included them in the evaluation for now. Similarly, we've retained them in the dataset for your potential use.

## Citation
If you find RealHiTBench is useful in your work, please consider citing the paper:
```bibtext
@misc{wu2025realhitbenchcomprehensiverealistichierarchical,
      title={RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis}, 
      author={Pengzuo Wu and Yuhang Yang and Guangcheng Zhu and Chao Ye and Hong Gu and Xu Lu and Ruixuan Xiao and Bowen Bao and Yijing He and Liangyu Zha and Wentao Ye and Junbo Zhao and Haobo Wang},
      year={2025},
      eprint={2506.13405},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.13405}, 
}
```
