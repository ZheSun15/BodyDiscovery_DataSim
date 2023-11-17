import os
import ast
import json
import openai
import argparse
import time
from datetime import datetime, timedelta
import numpy as np
import re
import csv
import data_simulate

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

# Load your API key from an environment variable or secret management service
# openai.api_key = "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"
# openai.api_key = "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"
openai.api_key = "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"

# model_list = openai.Model.list()
# for model in model_list['data']:
#     if model['id'][:3] == 'gpt':
#         print(model['id'])



def isint(response):
    try:
        int(response)
        return True
    except:
        return False


def find_list(string):

    try:
        pattern = r"\[.*?\]"
        matches = re.findall(string, pattern)
        return matches
    except:
        return False


def is_all_numeric(my_list):
    for item in my_list:
        if not isinstance(item, (int, float)):  # 使用 isinstance 检查元素是否为 int 或 float 类型
            return False
    return True


def state2prompt(obj_states, scene):
    global request_prompt
    N = obj_states[0].N
    prompt_head = "The states of the objects are as follows. Each row stands for an object and each column stands for a feature.\n"
    state_prompt = ""
    for i in range(len(obj_states)):
        state2str = np.array2string(obj_states[i].state, separator=',')
        state_prompt += state2str + "\n"
    state_prompt = state_prompt.replace('[', '').replace(']', '').replace(' ', '')
    prompt = prompt_head + state_prompt + request_prompt.format(N * len(scene))
    return prompt


def call_GPT(model, prompt):
    try:
        response = openai.ChatCompletion.create(model=model,  # model="gpt-3.5-turbo-16k",
                                                messages=[{
                                                    "role": "user",
                                                    "content": prompt
                                                }],
                                                max_tokens=50, #10240,
                                                # temperature=args.t,
                                                # n=1,
                                                # frequency_penalty=0,
                                                # presence_penalty=0.05
                                                )

        print(response, '\n')
        response_content = response['choices'][0]['message']['content']
        # response_content = response_content.replace('\\', '').replace('\n', '').replace('\"', '"').replace('，', ',')
        # response_content = ast.literal_eval(response_content)
        # with open(f"{p_str}_{day_flag}_{args.t}_{args.f}.json", "w") as file:
        #     json.dump(response_content, file, indent=4)
        return response_content
    except Exception as e:
        print(str(e))
        return False


def GPT_experiment(scene):
    test_model = "gpt-4-0613"
    # test_model = "gpt-3.5-turbo-16k"

    # set hyper-parameters
    Q = 10
    N = 20
    K_dict = {
        "2d_position": 2,
        "3d_position": 3,
        "light": 1,
        "3d_rotation": 1
    }
    noise_dict = {
        "2d_position": 1,
        "3d_position": 1,
        "light": 0,
        "3d_rotation": 0.1
    }
    max_try_step = 20
    fail_flag = False

    # introduce experiment to GPT
    global system_prompt
    system_prompt = system_prompt.format(N * len(scene), Q, scene)
    response = False
    try_step = 0
    while (not response) and (try_step < max_try_step):
        response = call_GPT(test_model, system_prompt)
        try_step += 1
    if try_step >= max_try_step:
        fail_flag = True
        print("Error: GPT failed to response to the setting.")
        return np.zeros((1, 5))

    # do single agent seperately
    if "3d_rotation" in scene:
        pass

    else:
        # init states
        K = 0
        for _scene in scene:
            K += K_dict[_scene]
        init_states = np.zeros((len(scene), N, K))
        K_start = 0
        for i, _scene in enumerate(scene):
            K_end = K_start + K_dict[_scene]
            init_states[i, :, K_start:K_end] = data_simulate.state_init(_scene, N, K_dict[_scene])
            K_start = K_end
        obj_states = [data_simulate.State(init_state=_state) for _state in init_states]  # scene_num * (N * K)
        # init APIs
        API_pool, APIs_idonly = data_simulate.API_init(Q=Q, N=N, scene_list=scene)

        # start experiment
        continue_flag = True
        T = 0
        last_prompt = system_prompt
        while continue_flag:
            T += 1

            # parse response
            if "give up" in response:
                continue_flag = False
                fail_flag = True
                break
            elif find_list(response):
                response = find_list(response)[0]
                # str to list
                response = response.replace('\n', '').replace('\\', '').replace('\"', '"').replace('，', ',')
                response = ast.literal_eval(response)
                fail_flag = not is_all_numeric(response)
                continue_flag = False
                break
            elif isint(response):
                # call api
                this_API_id = int(response)  # equal to API_list's this element
                if this_API_id == 0:  # call no api
                    continue
                else:
                    this_API_id -= 1
                APIs_ids_to_call = APIs_idonly[this_API_id, :]  # (1, scene_num)
                K_start = 0
                for scene_id, _scene in enumerate(scene):
                    K_end = K_start + K_dict[_scene]
                    _api = API_pool[_scene][APIs_ids_to_call[scene_id]]
                    noise_range = noise_dict[_scene]
                    obj_states[scene_id] = data_simulate.update_state(_api, obj_states[scene_id], _scene, K_start=K_start, K_end=K_end, env_flag=True, noise_range=noise_range)
                    K_start = K_end
                # send states to gpt and get new response
                prompt = state2prompt(obj_states, scene)
                response = False
                try_step = 0
                while (not response) and (try_step < max_try_step):
                    response = call_GPT(test_model, prompt)
                    try_step += 1
                if try_step >= max_try_step:
                    fail_flag = True
                    break
                # update last prompt
                last_prompt = prompt
            else:
                print("Error: response format error.")
                # regenerate last prompt
                response = False
                try_step = 0
                while (not response) and (try_step < max_try_step):
                    response = call_GPT(test_model, last_prompt)
                    try_step += 1
                if try_step >= max_try_step:
                    fail_flag = True
                    break

    # evaluation
    
    if fail_flag:
        print("Error: GPT failed to response to the setting.")
        return np.zeros((1, 5))
    elif "give up" in response:
        return np.zeros((1, 5))
    else:
        # prepare all body index
        gt_list = set()
        for scene_id, _scene in enumerate(scene):
            for q in range(Q):
                api = API_pool[_scene][APIs_idonly[q, scene_id]]
                new_obj_id = (scene_id * N) + np.array(api.object_ids)
                gt_list |= set(new_obj_id)
        F_gt = set(range(N * len(scene))) - gt_list
        pred_F = set(range(N * len(scene))) - set(response)
        TP_list = set(response) & gt_list
        TF_list = set(response) & F_gt
        FP_list = pred_F & F_gt
        FN_list = gt_list - set(response)

        acc = (len(TP_list) + len(FP_list)) / (N * len(scene))
        recall = len(TP_list)/len(gt_list)
        if len(response) == 0:
            precision = 0
        else:
            precision = len(TP_list)/len(response)
        spec = len(FP_list) / len(F_gt)
        F1 = 2*len(TP_list) / (2*len(TP_list) + len(FP_list) + len(FN_list))
        
        return np.array([acc, recall, precision, spec, F1]).reshape((1, 5))



def baseline():
    scenes = [
        # ["3d_rotation"],  # 单智能体
        ["2d_position"],
        # ["3d_position"],
        # ["light"],
        # ["2d_position", "light"],
        # ["3d_position", "light"],
        # ["2d_position", "3d_position"],
        # ["2d_position", "3d_position", "light"]
    ]

    for scene in scenes:
        round = 10
        metrics = np.zeros((round, 5))
        for r in range(round):
            metrics[r, :] = GPT_experiment(scene)
        
        # save metrics to csv
        with open("./results/gpt/{0}_metrics.csv".format(scene), "w") as f:
            writer = csv.writer(f)
            writer.writerows(metrics)

# 全局变量
system_prompt = open("./prompts/system_setting.txt", "r").read()
request_prompt = open("./prompts/request.txt", "r").read()
         
if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(7)    
    # 读取单智能体设定
    # franka 机械臂 joints, 0=x, 1=y, 2=z
    franka = json.load(open("./utils/franka.json", "r"))

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--t', type=float, default=1.115)
    # parser.add_argument('--f', type=str, default = '')  # comment or flag
    # args = parser.parse_args()
    # baseline(args)
    # time.sleep(100)
    
    baseline()

