import numpy as np
import json
import data_simulate

# --- preparation ---
# 读取单智能体设定
# franka 机械臂 joints, 0=x, 1=y, 2=z
franka = json.load(open("./utils/franka.json", "r"))
# 设置随机种子
np.random.seed(7)


# --- noise function ---
def env(object_state, K_start, K_end, scene_label, intensity, api_min=-0.5, api_max=0.5, agent_range=None):
    if "3d_rotation" in scene_label:
        if agent_range == None:
            agent_range = franka
        N_joints = agent_range.keys().__len__()
        for joint_idx in range(N_joints):
            _key = list(agent_range.keys())[joint_idx]
            _max = agent_range[_key]["range"][1]
            _min = agent_range[_key]["range"][0]
            _delta = (np.random.rand(1)-0.5) * (_max - _min) * intensity
            object_state.state[joint_idx] += _delta
    elif "position" in scene_label:
        env_deltas = (np.random.rand(object_state.N, K_end-K_start)-0.5) * (api_max - api_min) * intensity
        object_state.update(delta_state=env_deltas, K_start=K_start, K_end=K_end)
    elif "light" in scene_label:
        # generate env noise
        # env_affect_num = np.random.randint(low=1, high=object_state.N)
        env_affect_num = min(int(object_state.N * intensity), object_state.N)
        affect_list = np.random.choice(a=object_state.N, size=env_affect_num, replace=False)
        env_deltas = np.random.choice(a=[-1,1], size=(env_affect_num, K_end-K_start))
        for idx in range(env_affect_num):
            object_state.state[affect_list[idx]] += env_deltas[idx]
        # check validation, cut those more than 1
        object_state.state[np.where(object_state.state > 0)] = 1
        object_state.state[np.where(object_state.state < 0)] = 0
    
    return object_state


def other_agent(object_state, K_start, K_end, scene_label, other_api, intensity):
    if "3d_rotation" in scene_label:
        # 写在data_simulate中了，不便拆分
        print("skip 3d rotation...")
        pass
    else:
        if np.random.rand(1) < intensity:  # 以一定概率产生其他智能体干扰
            object_state.call_api(api=other_api, K_start=K_start, K_end=K_end)
            if "light" in scene_label:
                # check validation, cut those more than 1
                object_state.state[np.where(object_state.state > 0)] = 1
                object_state.state[np.where(object_state.state < 0)] = 0

    return object_state


def sense_flaw(object_state, K_start, K_end, scene_label, intensity, agent_range=None):
    if "3d_rotation" in scene_label:
        if agent_range == None:
            agent_range = franka
        N_joints = agent_range.keys().__len__()
        for joint_idx in range(N_joints):
            _max = object_state.state[joint_idx]
            _delta = (np.random.rand(1)-0.5) * np.abs(_max) * intensity
            object_state.state[joint_idx] += _delta
    elif "position" in scene_label:
        _max = np.max(object_state.state[:, K_start:K_end])
        sense_deltas = (np.random.rand(object_state.N, K_end-K_start)-0.5) * np.abs(_max) * intensity
        object_state.update(delta_state=sense_deltas, K_start=K_start, K_end=K_end)
    elif "light" in scene_label:
        # generate sense noise
        sense_affect_num = int(object_state.N * intensity)
        affect_list = np.random.choice(a=object_state.N, size=sense_affect_num, replace=False)
        sense_deltas = np.random.choice(a=[-1,1], size=(sense_affect_num, K_end-K_start))
        for idx in range(sense_affect_num):
            object_state.state[affect_list[idx]] += sense_deltas[idx]
        # check validation, cut those more than 1
        object_state.state[np.where(object_state.state > 0)] = 1
        object_state.state[np.where(object_state.state < 0)] = 0
    
    return object_state


def generate_other_api(scene_label, N, Q, K, T, api_min, api_max, all_body, other_rate):
    # init other agent apis
    if "3d_rotation" in scene_label:
        # 写在data_simulate中了，不便拆分
        print("skip 3d rotation...")
        pass
    else:
        other_body = list(set(range(N)) - set(all_body))
        other_num = max(int(len(other_body)*other_rate), 1)
        other_num = min(other_num, len(other_body))
        Q_num = max(int(Q*other_rate), 1)
        if "position" in scene_label:
            other_APIs, _ = data_simulate.generate_api(Q=Q_num, N_list=other_body, O=other_num, K=K, 
                                                    api_min=api_min*other_rate, api_max=api_max*other_rate, int_flag=False)  # len = Q
            other_APIlist = data_simulate.API_list_generate(len(other_APIs), T)
        elif "light" in scene_label:
            other_APIs, _ = data_simulate.generate_api(Q=Q_num, N_list=other_body, O=other_num, K=K, 
                                                    api_min=-1, api_max=1, int_flag=True)  # len = Q
            other_APIlist = data_simulate.API_list_generate(len(other_APIs), T)

    pass

    return other_APIs, other_APIlist