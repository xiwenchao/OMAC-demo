# OMAC: A Broad Optimization Framework for LLM-Based Multi-Agent Collaboration

## Overview

In this project, we define five optimization points for a multi-agent collaboration framework from both functional and structural perspectives. The five optimization points are:

1. **Fun-1**: Optimize existing candidate agents. E.g., optimize the instruction prompts and/or context examples of these agents.

2. **Fun-2**: Optimize the design and construction of new agents to participate in the collaboration process.

3. **Str-1**: Optimize an LLM-powered controller to choose the candidate agents from all existing agents before collaboration.

4. **Str-2**: Optimize an LLM-powered controller to dynamically select agents for participation and collaboration during current step.

5. **Str-3**: Optimize an LLM-powered controller to determine how agents communicate with each other during collaboration process.

The code and data are provided as a demon application on general reasoning tasks. The paper is in submission, and we will release the whole code and data after the paper is accepted. The demon experiments are conducted on the [MMLU](https://github.com/hendrycks/test) dataset.

## Structure of the folder

- `code`: Code of OMAC.
  - `MMLU`: Code of OMAC on MMLU dataset.
- `data`: data.
  - `MMLU`: MMLU dataset.

## Installation

```bash
conda create -n OMAC python=3.9
conda activate OMAC
cd code
pip install -r requirements.txt
```

## Run Experiments

1. Prepare an OpenAI API key and set it in the environment variable `OPENAI_API_KEY`, or set it in following Python scripts.

    ```python
    code/MMLU/run_evol.py
    ```

2. Run OMAC on different tasks by running the following scripts under the corresponding folder.

    ```python
    python -u run_evol.py
    ```

## Configuration

1. Most configurations are set in `run_evol.py`. You can modify the parameters in the script to adjust the settings for your experiments. The most important one is `prompt_roles_list`, which defines the seqence of optimization dimensions. For example, in code generation task, corresponding dimensions for Fun-1 are `'Economist', 'Doctor', 'Lawyer', 'Mathematician', 'Psychologist', 'Programmer', 'Historian'`. For Fun-2, Str-1, Str-2, Str-3, the dimensions are `'construct-role'`, `'pre-rank'`, `'rank'`, and `'structure'` respectively. The order of the dimensions is important, as it determines the sequence of optimization points. For example, if you set `prompt_roles_list = [['Economist', 'rank']]`, the optimization will be performed in the order of `Economist -> rank`, which means the existing agent `Economist` will be optimized first, and then the controller for `rank` will be optimized by keeping the optimized `Economist`.

2. All default intruction prompts for existing agents and controllers, as well as the intruction prompts for our Semantic Initializer and Contrastive Comparator are given in `prompt_lib.py`. You can view and modify the prompts in this file to adjust the behavior of the agents and controllers.

## Acknowledgments
 
This project incorporates code from the [DyLAN package](https://github.com/SALT-NLP/DyLAN), originally licensed under the MIT License.

Significant modifications have been made to adapt it to OMAC's goals and architecture.
