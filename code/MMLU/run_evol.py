import multiprocessing
from prompt_iteration.prompt import Prompt
from utils_evo import agent_collaboration, testify

import os
import time

import numpy as np
import json
import argparse
import datetime
from utils import *
import logging
import gc


# Prompt params:
initialize_num = 3  # 5

# data_dir = '../exp_data/'
dataset_name = 'mmlu'
evol_model = "gpt-3.5-turbo-1106"


FD_max_iter = 3  # 6
FD_min_improvement = 0.01  # 0.01
FD_tolerance_iter = 2  # 4
FD_score_threshold = 1  # 3

Iter_num = [1]


openai.api_key = ''

# ROLES="['Economist', 'Doctor', 'Lawyer', 'Mathematician', 'Psychologist', 'Programmer', 'Historian']"

if __name__ == '__main__':

    # specify the log file path
    # log_dir = '../../exp_data/{}/'.format(dataset_name)
    # os.makedirs(log_dir, exist_ok=True)
    # log_file = os.path.join(log_dir, 'log.txt')
    log_file = 'exp_log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO)

    prompt_roles_list = [["Mathematician"]]  # "rank", "construct-role", "pre-rank", "structure"

    for iter_num, prompt_roles in enumerate(prompt_roles_list):
        prompt_names = '-'.join(prompt_roles)

        DIR_NAME_TRAIN='./results/mmlu_{}_35_1106_train'.format(prompt_names)
        DIR_NAME_TEST='./results/mmlu_{}_35_1106_test'.format(prompt_names)
        # delete these two directories if they exist
        if os.path.exists(DIR_NAME_TRAIN):
            os.system('rm -r {}'.format(DIR_NAME_TRAIN))
        if os.path.exists(DIR_NAME_TEST):
            os.system('rm -r {}'.format(DIR_NAME_TEST))

        os.makedirs(DIR_NAME_TRAIN, exist_ok=True)
        os.makedirs(DIR_NAME_TEST, exist_ok=True)

        for iter in range(Iter_num[iter_num]):
            print('---------------Iteration {}----------------'.format(iter))
            # Run the prompt iteration process
            for prompt_role in prompt_roles:

                print('---------------Init Prompts----------------')
                prompt_all = Prompt(evol_model, dataset_name)
                prompt_all.initialize(prompt_role, initialize_num)
                agent_collaboration(prompt_all, prompt_role, DIR_NAME_TRAIN, iter)
                prompt_all.renew_prompts()


                print('---------------Feedback Mutation ----------------')

                tolerance = 0
                max_score = max(prompt_all.id2score.values())
                for t in range(FD_max_iter):
                    # get two prompt ids, the first one is randomly choosen from the first score_threshold prompts with highest scores, the second one is randomly choosen from the first score_threshold prompts with the lowest score_threshold scores. The information is stored in the prompt.id2score dictionary
                    prompt_pair = prompt_all.FD_parent_selection(FD_score_threshold + t)  #  + t
                    prompt_all.FD_mutation(prompt_pair)
                    agent_collaboration(prompt_all, prompt_role, DIR_NAME_TRAIN, iter)
                    prompt_all.renew_prompts()

                    if max(prompt_all.id2score.values()) - max_score < FD_min_improvement:
                        tolerance += 1
                    max_score = max(prompt_all.id2score.values())
                    if tolerance >= FD_tolerance_iter:
                        break
                    gc.collect()


                print('---------------Testify the Optimal Prompt----------------')

                # get the optimal prompt id
                sorted_id2score = sorted(prompt_all.id2score.items(), key=lambda x: x[1], reverse=True)
                optimal_prompt_id = sorted_id2score[0][0]      

                print("optimal_prompt_id", optimal_prompt_id)
                
                testify(prompt_all, prompt_role, optimal_prompt_id, DIR_NAME_TEST, iter)
                
                prompt_all.optimized_prompts[prompt_role] = [prompt_all.role_names[optimal_prompt_id], prompt_all.id2prompt[optimal_prompt_id]]

                gc.collect()            
