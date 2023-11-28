# generate simulated positions

import numpy as np
import pandas as pd
import os
import json
import glob
import csv
import matplotlib.pyplot as plt
import random
import seaborn as sns

# description:
# Q: How to test the accuracy of the algorithm?
# A: 每个场景测试test_num次。每次测试，随机生成N个物体的位置，每个物体有K个特征，每个特征有T次尝试机会。

class Feature():
    def __init__(self, feature_ids, feature_discript, feature_value):
        self.ids = feature_ids
        self.discription = feature_discript
        self.value = feature_value

    def __str__(self):
        return "feature_id: %s, feature_type: %s, feature_value: %s" % (self.feature_id, self.feature_type, self.feature_value)
    

class State():
    def __init__(self, init_state):
        self.state = init_state
        self.N = init_state.shape[0]
        self.K = init_state.shape[1]
    
    def update(self, delta_state, K_start=0, K_end=-1):
        if K_end == -1:
            K_end = self.K
        self.state[:, K_start:K_end] += delta_state
    
    def call_api(self, api, K_start=0, K_end=-1):
        if K_end == -1:
            K_end = self.K
        for id in range(len(api.object_ids)):
            self.state[api.object_ids[id], K_start:K_end] += api.deltas[id, :]


class API():
    def __init__(self, object_ids, feature_num):
        self.object_ids = object_ids
        self.feature_num = feature_num
        self.deltas = np.zeros((len(object_ids), feature_num))

    def update(self, deltas):
        self.deltas = deltas
    
    def to_dict(self):
        return {"object_ids": self.object_ids,
                "feature_num": self.feature_num,
                "deltas": self.deltas.tolist()
        }


def generate_api(Q, N, O=1, K=3, min=-1, max=1, int_flag=False):
    # Q: number of APIs
    # K: number of features
    # O: number of objects per API
    # return: list of APIs: [api 0, api 1, ..., api q], 
    #         api 1 = [(object_id, dx,dy,dz), (object_id, dx,dy,dz)],
    #         api q = [(object_id, dx,dt,dz)]
    # 其中，api 0 = 不调用api
    APIs = []
    APIs.append(API(object_ids=[0], feature_num=K))
    # sample the object list
    if O == 1:
        body_gt_num = O
    else:
        body_gt_num = np.random.randint(low=1, high=int(N/2))
    # body_gt_list starts from 0, max = N-1
    body_gt_list = np.random.choice(a=N, size=body_gt_num, replace=False)
    for q in range(Q):
        # sample the object list
        if O == 1:
            object_num = O
        else:
            object_num = np.random.randint(low=1, high=body_gt_num+1)
        object_ids = np.random.choice(a=body_gt_list, size=object_num, replace=False)
        if not isinstance(object_ids, list):
            object_ids = object_ids.tolist()
        api = API(object_ids=object_ids, feature_num=K)
        if int_flag:
            deltas = np.random.choice(a=[-1,1], size=(object_num, K))
        else:
            deltas = np.random.rand(object_num, K) - 0.5  # [-0.5, 0.5)
            # add bias to meet min&max
            deltas = deltas * (max - min)
        api.update(deltas=deltas)
        APIs.append(api)
    return APIs


def generate_rotation_api(Q, N, agent_range, O=1, K=3):

    # 其中，api 0 = 不调用api
    APIs = []
    APIs.append(API(object_ids=[0], feature_num=K))
    # body_gt_list starts from 0, max = N-1
    for q in range(Q):
        # sample the object list
        if O == 1:
            object_num = O
        else:
            object_num = np.random.randint(low=1, high=N+1)
        object_ids = np.random.choice(a=N, size=object_num, replace=False)
        if not isinstance(object_ids, list):
            object_ids = object_ids.tolist()
        api = API(object_ids=object_ids, feature_num=K)
        # 为每个object生成一个随机的旋转角度
        deltas = np.zeros((object_num, K))
        for i in range(object_ids.__len__()):
            key = list(agent_range.keys())[object_ids[i]]
            max = agent_range[key]["range"][1]
            min = agent_range[key]["range"][0]
            _delta = np.random.rand(1, K) - 0.5  # [-0.5, 0.5)
            _delta = _delta * (max - min)
            deltas[i, :] = _delta
        # add bias to meet min&max
        api.update(deltas=deltas)
        APIs.append(api)
    return APIs


def API_list_generate(Q, T):
    # 计算每份的长度
    length = T // Q
    # 计算余数，用于处理不能整除的情况
    remainder = T % Q
    # 生成分组
    groups = [length] * (Q - remainder) + [length + 1] * remainder
    random.shuffle(groups)

    APIlist = []
    for q in range(Q):
        APIlist += [q] * groups[q]
    
    random.shuffle(APIlist)

    return APIlist


def check_rotation_state(states, agent_range):
    N_joints = agent_range.keys().__len__()
    for i in range(N_joints):
        _key = list(agent_range.keys())[i]
        _max = agent_range[_key]["range"][1]
        _min = agent_range[_key]["range"][0]
        if states[i] > _max:
            states[i] = _max
        elif states[i] < _min:
            states[i] = _min
    return states


def generate_3d_position(save_path, APIs, APIlist, N=20, K_start=0, K_end=3, T=1000, min=-1, max=1):
    # N: number of objects
    # K: number of features
    # A: API list. Q = lenth(A)
    # T: number of time steps
    born_area_min = -50
    born_area_max = 50

    # calculate overall body
    all_body = set()
    for api in APIs[1:]:
        all_body = set(api.object_ids) | all_body
    all_body = list(all_body)
    # prepare columns.
    K = K_end - K_start
    columns = []
    for t in range(T+1):
        columns.append("V%s" % str(t))
    indexes = []
    for n in range(N):
        indexes.append("O%s" % str(n+1))

    dfs = []
    init_states = np.zeros((N, K))
    for k in range(K_start, K_end):
        dfs.append(pd.DataFrame(columns=columns, index=indexes))
        # initialize
        init_states[:, k] = np.random.rand(N) * (born_area_max - born_area_min) + born_area_min
        dfs[k].loc[:, "V0"] = init_states[:, k].copy()
    obj_state = State(init_state=init_states)

    for t in range(1,T+1):
        # generate N objects' features
        env_deltas = (np.random.rand(N, K)-0.5) * (max - min)
        obj_state.update(delta_state=env_deltas)

        # generate N objects' APIs
        # APIs = np.random.choice(a=N, size=N)
        # API = np.random.randint(low=0, high=len(APIs))
        api_id = APIlist[t-1]

        # call APIs once every time step
        obj_state.call_api(api=APIs[api_id])

        # save to dataframe
        for k in range(K_start, K_end):
            dfs[k].loc[:, "V%s" % str(t)] = obj_state.state[:, k].copy()
    
    # print(dfs[0])

    print("save to %s" % save_path)
    for k in range(K_start, K_end):
        _path = save_path + "/feature_{0}.xlsx".format(str(k+1))
        dfs[k].to_excel(_path, sheet_name="Sheet1")
    
    # save csv for xiaoxiao
    for k in range(K_start, K_end):
        _path = save_path + "/feature_{0}.csv".format(str(k+1))
        dfs[k].to_csv(_path, index=False) #, sheet_name="feature_{0}".format(str(k+1)))


    
    # save ground truth
    ran = [[N],
           APIlist, #.tolist(),
           [K],
           [len(APIs)-1]
    ]
    list_data = []
    for k in range(K):
        _obj_ids_per_feature = []
        for api in APIs:
            _obj_ids_per_feature.append(api.object_ids)
        list_data.append(_obj_ids_per_feature[1:])  # ignore first api
    api_gt = []
    for api in APIs[1:]:  # ignore first api
        api_gt.append(api.to_dict())
    api_gt.append(all_body)
    json.dump(ran, open(save_path + "/ran.json", "w"), indent=4)
    json.dump(list_data, open(save_path + "/list_data.json", "w"), indent=4)
    json.dump(api_gt, open(save_path + "/api_gt.json", "w"), indent=4)


def generate_light(save_path, APIs, APIlist, N=20, K_start=0, K_end=1, T=100, mirror_flag=False, min=-1, max=1):

    # N: number of objects
    # K: number of features
    # A: API list. Q = lenth(A)
    # T: number of time steps
    born_area_min = -50
    born_area_max = 50

    # calculate overall body
    all_body = set()
    for api in APIs[1:]:
        all_body = set(api.object_ids) | all_body
    all_body = list(all_body)
    # prepare columns.
    K = K_end - K_start
    columns = []
    for t in range(T+1):
        columns.append("V%s" % str(t))
    indexes = []
    for n in range(N):
        indexes.append("O%s" % str(n+1))
    
    # pick lights in the mirror
    N_total = N
    if mirror_flag:
        mirror_light_num = np.random.randint(low=0, high=len(all_body))
        mirror_id_list = np.random.choice(a=all_body, size=mirror_light_num, replace=False)
        N_total = N + mirror_light_num

    dfs = []
    born_spots = np.random.rand(N_total, 2) * (born_area_max - born_area_min) + born_area_min
    init_states = np.zeros((N_total, K))
    for k in range(K_start, K_end):
        dfs.append(pd.DataFrame(columns=columns, index=indexes))
        # initialize
        init_states[:N, k] = np.random.randint(low=0, high=2, size=(N))
        dfs[k].loc[:N, "V0"] = init_states[:N, k].copy()
    if mirror_flag:
        N_id = N
        for mirrored_id in mirror_id_list:
            init_states[N_id, k] = init_states[mirrored_id, k]
            N_id += 1
    obj_state = State(init_state=init_states)

    for t in range(1,T+1):
        # generate env noise
        env_affect_num = np.random.randint(low=1, high=N)
        affect_list = np.random.choice(a=N, size=env_affect_num, replace=False)
        env_deltas = np.random.choice(a=[-1,1], size=(env_affect_num, K))
        for idx in range(env_affect_num):
            obj_state.state[affect_list[idx]] += env_deltas[idx]
        # check validation, cut those more than 1
        obj_state.state[np.where(obj_state.state > 0)] = 1
        obj_state.state[np.where(obj_state.state < 0)] = 0

        # call APIs once every time step
        api_id = APIlist[t-1]
        obj_state.call_api(api=APIs[api_id])

        # check validation, cut those more than 1
        obj_state.state[np.where(obj_state.state > 0)] = 1
        obj_state.state[np.where(obj_state.state < 0)] = 0

        # if mirror_flag, update mirror lights
        N_id = N
        for mirrored_id in mirror_id_list:
            obj_state[N_id, k] = obj_state[mirrored_id, k]
            N_id += 1

        # save to dataframe
        for k in range(K_start, K_end):
            dfs[k].loc[:, "V%s" % str(t)] = obj_state.state[:, k].copy()
    
    # print(dfs[0])

    print("save to %s" % save_path)
    for k in range(K_start, K_end):
        _path = save_path + "/feature_{0}.xlsx".format(str(k+1))
        dfs[k].to_excel(_path, sheet_name="Sheet1")
    # # save txt for zhenlaing
    # for k in range(K_start, K_end):
    #     _path = save_path + "/feature_{0}.txt".format(str(k+1))
    #     dfs[k].to_csv(_path, sep=' ', index=False)

    # save ground truth
    ran = [[N],
           APIlist, #.tolist(),
           [K],
           [len(APIs)-1]
    ]
    # # save ground truth for zhenlaing
    # ran = {"obj": N,  # obj_num
    #        "actions": APIlist.tolist(),  # apiList
    #        "state": K,  # feature_num
    #        "api_cate": len(APIs)-1  # api_num
    # }
    
    list_data = []
    for k in range(K):
        _obj_ids_per_feature = []
        for api in APIs:
            _obj_ids_per_feature.append(api.object_ids)
        list_data.append(_obj_ids_per_feature[1:])  # ignore first api
    api_gt = []
    for api in APIs[1:]:  # ignore first api
        api_gt.append(api.to_dict())
    api_gt.append(all_body)
    
    # # convert born_spots for zhenlaing
    # born_spot_json = {}
    # born_spot_json["x"] = born_spots[:, 0].tolist()
    # born_spot_json["z"] = born_spots[:, 1].tolist()


    json.dump(born_spots.tolist(), open(save_path + "/born_spots.json", "w"), indent=4)
    # json.dump(born_spot_json, open(save_path + "/born.json", "w"), indent=4)
    json.dump(ran, open(save_path + "/ran.json", "w"), indent=4)
    json.dump(list_data, open(save_path + "/list_data.json", "w"), indent=4)
    json.dump(api_gt, open(save_path + "/api_gt.json", "w"), indent=4)

    # # convert list_data to txt for zhenlaing
    # with open(save_path + "/list_data.txt", "w") as file:
    #     for feature in list_data:
    #         for row in feature:
    #             line = ','.join(map(str, row))  # 将列表中的每个元素转换为字符串并用空格分隔
    #             file.write(line + "\n")  # 写入一行并添加换行符
    # print('finish.')



    return 1


def generate_single_agent(save_path, APIs, APIlist, N=2, K_start=0, K_end=1, T=100, min=-1, max=1, agent_range=None):

    # N: number of objects. N=joint number here. ignore the input param N.
    # K: number of features
    # A: API list. Q = lenth(A)
    # T: number of time steps
    born_area_min = -30
    born_area_max = 30
    env_flag = True
    env_noise_rate = 0.1
    other_flag = True
    if agent_range == None:
        agent_range = franka

    # calculate overall body
    K = K_end - K_start
    N_joints = franka.keys().__len__()  # =9
    all_data = np.zeros((N, N_joints, T+1, K))
    # 设置N个机械臂的出生位置
    born_spots = np.random.rand(N, 2) * (born_area_max - born_area_min) + born_area_min
    # 初始化每个机械臂的状态
    all_init_states = np.zeros((N, N_joints, K))
    for k in range(K):
        # initialize franka joints
        for idx in range(N):
            for i in range(N_joints):
                _key = list(franka.keys())[i]
                all_init_states[idx, i, k] = franka[_key]["init"][franka[_key]["control"]]
        all_data[:, :, 0, k] = all_init_states[:, :, k].copy()
    obj_state = [State(init_state=_state) for _state in all_init_states]

    # initiate other agent apis
    other_APIs = [generate_rotation_api(Q=len(APIs)-1, N=N_joints, agent_range=franka, O=N_joints, K=K)
                  for i in range(N-1)]  # len=((N-1), Q)
    all_APIs = [APIs] + other_APIs
    if other_flag:
        all_API_list = [APIlist] + [API_list_generate(len(APIs), T) for i in range(N-1)]  # len=(N, T)
    else:
        all_API_list = [APIlist] + [np.zeros(T) for i in range(N-1)]
                
    for t in range(1,T+1):
        # call apis
        for n in range(N):
            api_id = all_API_list[n][t-1]
            obj_state[n].call_api(api=all_APIs[n][api_id])
            # check validation, cut those more than max
            obj_state[n].state = check_rotation_state(obj_state[n].state, agent_range)

            # env affect
            if env_flag:
                for joint_idx in range(N_joints):
                    _key = list(franka.keys())[joint_idx]
                    _min = franka[_key]["range"][0]
                    _max = franka[_key]["range"][1]
                    _delta = (np.random.rand(1)-0.5) * (_max - _min) * env_noise_rate
                    obj_state[n].state[joint_idx] += _delta
                # check validation, cut those more than max
                obj_state[n].state = check_rotation_state(obj_state[n].state, agent_range)

            # save to all_data
            all_data[n, :, t, :] = obj_state[n].state.copy()
    
    # save simulated data
    print("save to %s" % save_path)
    # prepare columns.
    columns = []
    for t in range(T+1):
        columns.append("V%s" % str(t))
    # choose one and assigned it to the agent.
    idx = np.arange(N)
    np.random.shuffle(idx)
    my_agent_id = idx[0]
    all_data[idx, :, :, :] = all_data
    # save excel for R
    for k in range(K):
        _path = save_path + "/feature_{0}.xlsx".format(str(k+K_start+1))
        _data = all_data[:, :, :, k].reshape(N*N_joints, T+1)
        df = pd.DataFrame(_data, columns=columns)
        df.to_excel(_path, sheet_name="feature_{0}".format(str(k+K_start+1)))
    # save csv for xiaoxiao
    for k in range(K):
        _path = save_path + "/feature_{0}.csv".format(str(k+K_start+1))
        _data = all_data[:, :, :, k].reshape(N*N_joints, T+1)
        df = pd.DataFrame(_data, columns=columns)
        df.to_csv(_path, index=False) #, sheet_name="feature_{0}".format(str(k+K_start+1)))

    # save ground truth
    ran = [[N*N_joints],
           APIlist, #.tolist(),
           [K],
           [len(APIs)-1]
    ]
    list_data = []
    for k in range(K):
        _obj_ids_per_feature = []
        for api in APIs:
            _obj_id = [int(my_agent_id * N_joints + _id) for _id in api.object_ids]
            _obj_ids_per_feature.append(_obj_id)
        list_data.append(_obj_ids_per_feature[1:])  # ignore first api

    # calculate all body
    # save api gt
    api_gt = []
    all_body = set()
    for api in APIs[1:]:
        _obj_id = [int(my_agent_id * N_joints + _id) for _id in api.object_ids]
        all_body |= set(_obj_id)
        api.object_ids = [int(my_agent_id * N_joints + _id) for _id in api.object_ids]
        api_gt.append(api.to_dict())
    api_gt.append(list(all_body))
    api_gt.append([int(my_agent_id)])
    json.dump(api_gt, open(save_path + "/api_gt.json", "w"), indent=4)
    json.dump(born_spots.tolist(), open(save_path + "/born_spots.json", "w"), indent=4)
    json.dump(ran, open(save_path + "/ran.json", "w"), indent=4)
    json.dump(list_data, open(save_path + "/list_data.json", "w"), indent=4)

    return


def state_init(scene, N, K, agent_range=None):
    born_area_min = -50
    born_area_max = 50
    if "position" in scene:
        new_state = np.random.rand(N, K) * (born_area_max - born_area_min) + born_area_min
    elif "light" in scene:
        new_state = np.random.randint(low=0, high=2, size=(N, K))
    elif "rotation" in scene:
        if agent_range == None:
            agent_range = franka
        N_joints = agent_range.keys().__len__()  # =9
        new_state = np.zeros((N, N_joints, K))
        for k in range(K):
            # initialize agent joints
            for idx in range(N):
                for i in range(N_joints):
                    _key = list(franka.keys())[i]
                    new_state[idx, i, k] = franka[_key]["init"][franka[_key]["control"]]
        new_state = new_state.reshape((N*N_joints, K))
    
    return new_state


def API_init(Q, N, scene_list):
    # generage api pool
    N_joints = franka.keys().__len__()
    API_pool = {
        "2d_position": generate_api(Q=Q, N=N, O=N, K=2, min=-1, max=1, int_flag=False),
        "3d_position": generate_api(Q=Q, N=N, O=N, K=3, min=-1, max=1, int_flag=False),
        "light": generate_api(Q=Q, N=N, O=N, K=1, min=-1, max=1, int_flag=True),
        "3d_rotation": generate_rotation_api(Q=Q, N=N_joints, agent_range=franka, O=N_joints, K=1)
    }

    # choose api from the pool
    APIs_idonly = np.random.randint(low=0, high=(Q+1), size=(Q, len(scene_list)))  # choose Q apis for each scene
    # APIs_idonly = [API_list_generate(Q+1, Q+1) for i in range(len(scene_list))]  # [scene_num, Q+1]
    
    return API_pool, APIs_idonly


def update_state(api, object_state, scene_label, K_start, K_end, env_flag, noise_range, agent_range=None):
    N = object_state.N
    K = K_end-K_start
    # call apis
    object_state.call_api(api=api, K_start=K_start, K_end=K_end)
    if "light" in scene_label:
        # check validation, cut those more than 1
        object_state.state[np.where(object_state.state > 0)] = 1
        object_state.state[np.where(object_state.state < 0)] = 0

    # env affect
    if env_flag:
        if "3d_rotation" in scene_label:
            if agent_range == None:
                agent_range = franka
            N_joints = agent_range.keys().__len__()
            for joint_idx in range(N_joints):
                _key = list(agent_range.keys())[joint_idx]
                _max = agent_range[_key]["range"][1]
                _min = agent_range[_key]["range"][0]
                _delta = (np.random.rand(1)-0.5) * (_max - _min) * noise_range
                object_state.state[joint_idx] += _delta
        elif "position" in scene_label:
            env_deltas = (np.random.rand(N, K)-0.5) * noise_range
            object_state.update(delta_state=env_deltas, K_start=K_start, K_end=K_end)
        elif "light" in scene_label:
            # generate env noise
            env_affect_num = np.random.randint(low=1, high=N)
            affect_list = np.random.choice(a=N, size=env_affect_num, replace=False)
            env_deltas = np.random.choice(a=[-1,1], size=(env_affect_num, K))
            for idx in range(env_affect_num):
                object_state.state[affect_list[idx]] += env_deltas[idx]
            # check validation, cut those more than 1
            object_state.state[np.where(object_state.state > 0)] = 1
            object_state.state[np.where(object_state.state < 0)] = 0

    return object_state


def data_simulator(scene, Q, N, T, mirror_flag=False):
    Round = 10
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

    for r in range(Round):
        
        if len(scene) == 1:
            save_path = "./data/{0}_round{1}_Q{2}_N{3}_T{4}".format(scene[0], str(r), str(Q), str(N), str(T))
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            _scene = scene[0]
            K = K_dict[_scene]

            if "light" in _scene:
                APIs = generate_api(Q=Q, N=N, O=N, K=K, min=-1, max=1, int_flag=True)
                APIlist = API_list_generate(len(APIs), T)  #np.random.choice(a=len(APIs), size=T)  # API id = 0, don't call API
                generate_light(save_path, APIs=APIs, APIlist=APIlist, N=N, K_start=0, K_end=K, T=T, mirror_flag=mirror_flag)
            elif "rotation" in _scene:
                N_joints = franka.keys().__len__()  # =9
                APIs = generate_rotation_api(Q=Q, N=N_joints, agent_range=franka, O=N_joints, K=K)
                APIlist = API_list_generate(len(APIs), T)  #np.random.choice(a=len(APIs), size=T)  # API id = 0, don't call API
                generate_single_agent(save_path, APIs=APIs, APIlist=APIlist, N=N, K_start=0, K_end=K, T=T)
            else:
                APIs = generate_api(Q=Q, N=N, O=N, K=K, min=-1, max=1, int_flag=False)
                APIlist = API_list_generate(len(APIs), T)  #np.random.choice(a=len(APIs), size=T)  # API id = 0, don't call API
                generate_3d_position(save_path, APIs=APIs, APIlist=APIlist, N=N, K_start=0, K_end=K, T=T, min=-1, max=1)
        else:
            save_path = "./data/{0}_round{1}_Q{2}_N{3}_T{4}".format('_'.join(scene), str(r), str(Q), str(N), str(T))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # pass
            API_pool, APIs_idonly = API_init(Q=Q, N=N, scene_list=scene)
            pool_APIlist = API_list_generate(Q, T)  #np.random.choice(a=Q, size=T)  # API id = 0, don't call API

            # calculate total feature number
            K_total = 0
            for _scene in scene:
                K_total += K_dict[_scene]
            # init states
            all_data = np.zeros((len(scene), N, T+1, K_total))
            init_states = np.zeros((len(scene), N, K_total))
            K_start = 0
            for i, _scene in enumerate(scene):
                K_end = K_start + K_dict[_scene]
                init_states[i, :, K_start:K_end] = state_init(_scene, N, K_dict[_scene])
                all_data[i, :, 0, K_start:K_end] = init_states[i, :, K_start:K_end].copy()
                K_start = K_end
            obj_states = [State(init_state=_state) for _state in init_states]  # scene_num * (N * K)

            # generate data
            for t in range(1, T+1):
                this_API_id = pool_APIlist[t-1]
                APIs_ids_to_call = APIs_idonly[this_API_id, :]
                K_start = 0
                for scene_id, _scene in enumerate(scene):
                    K_end = K_start + K_dict[_scene]
                    _api = API_pool[_scene][APIs_ids_to_call[scene_id]]
                    noise_range = noise_dict[_scene]
                    obj_states[scene_id] = update_state(_api, obj_states[scene_id], _scene, K_start=K_start, K_end=K_end, env_flag=True, noise_range=noise_range)
                    all_data[scene_id, :, t, :] = obj_states[scene_id].state.copy()
                    K_start = K_end

            # save data
            print("save to %s" % save_path)
            # prepare columns.
            columns = []
            for t in range(T+1):
                columns.append("V%s" % str(t))
            # save excel for R
            for k in range(K_total):
                _path = save_path + "/feature_{0}.xlsx".format(str(k+1))
                _data = all_data[:, :, :, k].reshape(N*len(scene), T+1)
                df = pd.DataFrame(_data, columns=columns)
                df.to_excel(_path, sheet_name="feature_{0}".format(str(k+K_start+1)))

            # save ground truth
            ran = [[N*len(scene)],
                pool_APIlist, #.tolist(),
                [K_total],
                [Q]
            ]
            list_data = []  # len=(K_total, Q)
            # update object ids
            for k in range(K_total):
                list_data.append([])
            for APIid in range(Q):
                apis_to_call = APIs_idonly[APIid, :]
                K_start = 0           
                for scene_id, _scene in enumerate(scene):
                    K_end = K_start + K_dict[_scene]
                    for k in range(K_start, K_end):
                      list_data[k].append([int(_id + scene_id*N) 
                                           for _id 
                                           in API_pool[_scene][apis_to_call[scene_id]].object_ids])
                    K_start = K_end
                    
            # calculate all body
            # save api gt
            api_gt = []
            # all_body = set()
            for api_id in range(Q):
                apis_to_call = APIs_idonly[api_id, :]
                _api_gt = [] # len = scene_num
                for scene_id, _scene in enumerate(scene):
                    api = API_pool[_scene][apis_to_call[scene_id]]
                    obj_ids_with_compensation = [int(scene_id*N + _id) for _id in api.object_ids]
                    api_dict = api.to_dict()
                    api_dict["object_ids"] = obj_ids_with_compensation
                    _api_gt.append(api_dict)
                api_gt.append(_api_gt)

            json.dump(api_gt, open(save_path + "/api_gt.json", "w"), indent=4)
            # json.dump(born_spots.tolist(), open(save_path + "/born_spots.json", "w"), indent=4)
            json.dump(ran, open(save_path + "/ran.json", "w"), indent=4)
            json.dump(list_data, open(save_path + "/list_data.json", "w"), indent=4)


def removeNAN(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            for k in range(len(data[0][0])):
                if data[i][j][k] == "NA":
                    data[i][j][k] = 1
                elif data[i][j][k] == "NaN":
                    data[i][j][k] = 1
    return data


def evaluate_mirror(scene, ablation_flag, only_mirror_flag=False, Q=10, N=20, T=1000):
    # evaluate_style = "effect"
    # ths = [2.5758293035489004, 1.959963984540054]  # 0.01, 0.05, k=1
    # # ths = [2.807033768343811, 2.241402727604947]# 0.01, 0.05, k=2
    # # ths = [2.935199468866699, 2.3939797998185104] # 0.01, 0.05, k=3
    # # ths = [3.023341439739154, 2.497705474412374] # 0.01, 0.05, k=4
    # # ths = [3.090232306167813, 2.5758293035489004] # 0.01, 0.05, k=5
    # # ths = [3.143980287069073, 2.638257273476751] # 0.01, 0.05, k=6   
    # effect_flag = True

    evaluate_style = "p"
    if "rotation" in scene[0]:
        correction = 27*10*1
    elif "2d_position" in scene[0]:
        correction = 20*10*2
    elif "3d_position" in scene[0]:
        correction = 20*10*3
    elif "light" in scene[0]:
        correction = 20*10*1
    ths = [0.05]  # /correction   #0.05/(20*10*3)] #, 0.01/(27*10*1)]  # [0.05, 0.01]
    effect_flag = False

    for th in ths:
        prediction_root = "./prediction/origin_exp"
        gt_root = "./data"

        all_acc = []
        all_recall = []
        all_precision = []
        all_spec = []
        all_F1 = []

        scene_name = ""
        for _scene in scene:
            scene_name = scene_name + _scene + '_'

            # load prediction result
            if ablation_flag:
                result_pths = sorted(glob.glob(prediction_root + "/{0}_round*_Q{1}_N{2}_T{3}_{4}list.json".format(_scene, 
                                                                                                                Q,
                                                                                                                N,
                                                                                                                T,
                                                                                                                evaluate_style)))
            else:
                result_pths = sorted(glob.glob(prediction_root + "/{0}_round*_{1}list.json".format(_scene, evaluate_style)))
            
            
            # check body number
            min_N = 100
            for result_pth in result_pths:
                file_name = result_pth.split("/")[-1]
                # find ground truth path
                gt_folder = "{0}/{1}".format(gt_root, file_name.replace("_{0}list.json".format(evaluate_style), ""))
                gt_list = json.load(open(gt_folder + "/api_gt.json", "r"))
                if "rotation" in _scene:
                    body_num = len(gt_list[-2])
                else:
                    body_num = len(gt_list[-1])
                if body_num < min_N:
                    min_N = body_num

            # choose mirrored object number
            mirror_obj_num = np.random.randint(low=1, high=min_N+1)
            
            for result_pth in result_pths:
                file_name = result_pth.split("/")[-1]
                # find ground truth path
                gt_folder = "{0}/{1}".format(gt_root, file_name.replace("_{0}list.json".format(evaluate_style), ""))
                gt_list = json.load(open(gt_folder + "/api_gt.json", "r"))
                if "rotation" in _scene:
                    gt_obj_ids = gt_list[-2]
                else:
                    gt_obj_ids = gt_list[-1]
                ran = json.load(open(gt_folder + "/ran.json", "r"))
                N_physic = ran[0][0]
                N_total = N_physic + mirror_obj_num
                # load prediction result
                pred = json.load(open(result_pth, "r"))  # len=(N, Q, K)
                # choose mirrored objects & add mirrored predictions
                mirrored_ids = np.random.choice(a=gt_obj_ids, size=mirror_obj_num, replace=False)
                for k in range(len(pred)):
                    for mirrored_obj_id in mirrored_ids:
                        pred[k].append(pred[k][mirrored_obj_id]) # pred shape -> (K, N+mirror_num, Q)
                if effect_flag:
                    # convert to ndarray
                    pred_array = np.array(pred)
                    # mean of all N*Q*K
                    mean_eff = np.mean(pred_array)
                    # various
                    var_eff = np.var(pred_array, ddof=1)
                    eff_min = mean_eff - th*var_eff
                    eff_max = mean_eff + th*var_eff
                
                for api_id in range(sum(type(gt) is dict for gt in gt_list)):
                    pred_obj = set()
                    for k in range(len(pred)):
                        _pred_array = np.array(pred[k])
                        _p = _pred_array[:, api_id]
                        if effect_flag:
                            pred_obj = set(np.where(np.logical_or(_p < eff_min, _p > eff_max))[0]) | pred_obj
                            pass
                        else:
                            pred_obj = set(np.where(_p < th)[0]) | pred_obj
                    # add mirrored ids
                    gt_obj = set(gt_list[api_id]["object_ids"])
                    mirror_gt_obj = set()
                    mirrored_id_for_this_api = set(mirrored_ids) & gt_obj
                    for _i in mirrored_id_for_this_api:
                        mirror_index_to_add = np.where(mirrored_ids==_i)[0] + N_physic
                        gt_obj = gt_obj | set(mirror_index_to_add)
                        mirror_gt_obj = mirror_gt_obj | set(mirror_index_to_add)
                    # calculate metircs
                    if only_mirror_flag:
                        physic_gt_obj = set(gt_list[api_id]["object_ids"])
                        mirror_pred_obj = pred_obj - physic_gt_obj
                        TP_list = mirror_pred_obj & mirror_gt_obj
                        pred_F = set(range(N_total)) - mirror_pred_obj
                        F_gt = set(range(N_total)) - mirror_gt_obj
                        TF_list = mirror_pred_obj & F_gt
                        FP_list = pred_F & F_gt
                        FN_list = mirror_gt_obj - mirror_pred_obj
                        predict_len = len(mirror_pred_obj)
                        gt_len = len(mirror_gt_obj)
                    else:
                        TP_list = pred_obj & gt_obj
                        pred_F = set(range(N_total)) - pred_obj
                        F_gt = set(range(N_total)) - gt_obj
                        TF_list = pred_obj & F_gt
                        FP_list = pred_F & F_gt
                        FN_list = gt_obj - pred_obj
                        predict_len = len(pred_obj)
                        gt_len = len(gt_obj)
                    # acc
                    _acc = (len(TP_list) + len(FP_list)) / N_total
                    all_acc.append(_acc)
                    # recall
                    if gt_len == 0:
                        _recall = 1
                    else:
                        _recall = len(TP_list)/gt_len
                    all_recall.append(_recall)
                    # precision
                    if predict_len == 0:
                        _precision = 0
                    else:
                        _precision = len(TP_list)/predict_len
                    all_precision.append(_precision)
                    # specificity
                    _spec = len(FP_list) / len(F_gt)
                    all_spec.append(_spec)
                    # F1
                    # _f1 = 2*len(TP_list) / (2*len(TP_list) + len(FP_list) + len(FN_list))
                    if (_recall + _precision) != 0:
                        _f1 = 2*_recall*_precision / (_recall + _precision)
                    else:
                        _f1 = 0
                    all_F1.append(_f1)

        print("recall: {0}, \n precision: {1}".format(np.mean(all_recall), np.mean(all_precision)))
        metrics = [
            np.mean(all_acc),
            np.mean(all_recall),
            np.mean(all_precision),
            np.mean(all_spec),
            np.mean(all_F1)
        ]
        
        with open("./results/{0}{1}{2}_only_mirror_metrics.csv".format(scene_name, evaluate_style, th), "w") as f:
            writer = csv.writer(f)
            for value in metrics:
                writer.writerow([value])
        print("save to ./results/{0}{1}{2}_only_mirror_metrics.csv".format(scene_name, evaluate_style, th))




def evaluate(scene, ablation_flag, cross_flag=False, Q=10, N=20, T=1000):
    # evaluate_style = "effect"
    # # ths = [2.5758293035489004, 1.959963984540054]  # 0.01, 0.05, k=1
    # ths = [2.807033768343811, 2.241402727604947]# 0.01, 0.05, k=2
    # # ths = [2.935199468866699, 2.3939797998185104] # 0.01, 0.05, k=3
    # # ths = [3.023341439739154, 2.497705474412374] # 0.01, 0.05, k=4
    # # ths = [3.090232306167813, 2.5758293035489004] # 0.01, 0.05, k=5
    # # ths = [3.143980287069073, 2.638257273476751] # 0.01, 0.05, k=6   
    # effect_flag = True

    evaluate_style = "p"
    ths = [0.05/(27*10*1)]  #0.05/(20*10*3)] #, 0.01/(27*10*1)]  # [0.05, 0.01]
    effect_flag = False

    for th in ths:
        prediction_root = "./prediction/S1"
        gt_root = "./data"

        all_acc = []
        all_recall = []
        all_precision = []
        all_spec = []
        all_F1 = []
        scene_name = ""
        if not cross_flag:
            for _scene in scene:
                scene_name = scene_name + _scene + '_'
                # load prediction result
                if ablation_flag:
                    result_pths = sorted(glob.glob(prediction_root + "/{0}_round*_Q{1}_N{2}_T{3}_{4}list.json".format(_scene, 
                                                                                                                    Q,
                                                                                                                    N,
                                                                                                                    T,
                                                                                                                    evaluate_style)))
                else:
                    result_pths = sorted(glob.glob(prediction_root + "/{0}_round*_{1}list.json".format(_scene, evaluate_style)))
                for result_pth in result_pths:
                    # clear
                    acc_per_round = []
                    recall_per_round = []
                    precision_per_round = []
                    spec_per_round = []
                    F1_per_round = []

                    file_name = result_pth.split("/")[-1]
                    # find ground truth path
                    gt_folder = "{0}/{1}".format(gt_root, file_name.replace("_{0}list.json".format(evaluate_style), ""))
                    gt_list = json.load(open(gt_folder + "/api_gt.json", "r"))
                    ran = json.load(open(gt_folder + "/ran.json", "r"))
                    N = ran[0][0]
                    # load prediction result
                    pred = json.load(open(result_pth, "r"))  # len=(N, Q, K)
                    if effect_flag:
                        # convert to ndarray
                        pred_array = np.array(pred)
                        # mean of all N*Q*K
                        mean_eff = np.mean(pred_array)
                        # various
                        var_eff = np.var(pred_array, ddof=1)
                        eff_min = mean_eff - th*var_eff
                        eff_max = mean_eff + th*var_eff
                    
                    for api_id in range(sum(type(gt) is dict for gt in gt_list)):
                        pred_obj = set()
                        for k in range(len(pred)):
                            _pred_array = np.array(pred[k])
                            _p = _pred_array[:, api_id]
                            if effect_flag:
                                pred_obj = set(np.where(np.logical_or(_p < eff_min, _p > eff_max))[0]) | pred_obj
                                pass
                            else:
                                pred_obj = set(np.where(_p < th)[0]) | pred_obj
                        # calculate metircs
                        TP_list = pred_obj & set(gt_list[api_id]["object_ids"])
                        pred_F = set(range(N)) - pred_obj
                        F_gt = set(range(N)) - set(gt_list[api_id]["object_ids"])
                        TF_list = pred_obj & F_gt
                        FP_list = pred_F & F_gt
                        FN_list = set(gt_list[api_id]["object_ids"]) - pred_obj
                        # acc
                        _acc = (len(TP_list) + len(FP_list)) / N
                        acc_per_round.append(_acc)
                        all_acc.append(_acc)
                        # recall
                        _recall = len(TP_list)/len(gt_list[api_id]["object_ids"])
                        recall_per_round.append(_recall)
                        all_recall.append(_recall)
                        # precision
                        if len(pred_obj) == 0:
                            _precision = 0
                        else:
                            _precision = len(TP_list)/len(pred_obj)
                        precision_per_round.append(_precision)
                        all_precision.append(_precision)
                        # specificity
                        _spec = len(FP_list) / len(F_gt)
                        spec_per_round.append(_spec)
                        all_spec.append(_spec)
                        # F1
                        # _f1 = 2*len(TP_list) / (2*len(TP_list) + len(FP_list) + len(FN_list))
                        if (_recall + _precision) != 0:
                            _f1 = 2*_recall*_precision / (_recall + _precision)
                        else:
                            _f1 = 0
                        F1_per_round.append(_f1)
                        all_F1.append(_f1)

                    
                    # print("For {0}:\n    m-recall: {1}, m-precision: {2}".format(file_name, 
                    #                                                             np.mean(recall_per_round), 
                    #                                                             np.mean(precision_per_round)))
        else:
            scene_name = "_".join(scene) + "_"
            result_pths = sorted(glob.glob(prediction_root + "/{0}round*_Q{1}_N{2}_T{3}_{4}list.json".format(scene_name, 
                                                                                                            Q,
                                                                                                            N,
                                                                                                            T,
                                                                                                            evaluate_style)))
            for result_pth in result_pths:
                file_name = result_pth.split("/")[-1]
                # find ground truth path
                gt_folder = "{0}/{1}".format(gt_root, file_name.replace("_{0}list.json".format(evaluate_style), ""))
                gt_list = json.load(open(gt_folder + "/api_gt.json", "r"))
                ran = json.load(open(gt_folder + "/ran.json", "r"))
                N_total = ran[0][0]
                # load prediction result
                pred = json.load(open(result_pth, "r"))  # len=(K, N, Q)
                if effect_flag:
                    # convert to ndarray
                    pred_array = np.array(removeNAN(pred))
                    # mean of all N*Q*K
                    mean_eff = np.mean(pred_array)
                    # various
                    var_eff = np.var(pred_array, ddof=1)
                    eff_min = mean_eff - th*var_eff
                    eff_max = mean_eff + th*var_eff
                
                pred_array = np.array(removeNAN(pred))
                pred_array = np.min(pred_array, axis=0)
                for api_id in range(len(gt_list)):
                    _p = pred_array[:, api_id]
                    if effect_flag:
                        pred_obj = set(np.where(np.logical_or(_p < eff_min, _p > eff_max))[0])
                        pass
                    else:
                        pred_obj = set(np.where(_p < th)[0])
                    # prepare gt obj
                    gt_obj = set()
                    for APIs in gt_list[api_id]:
                        gt_obj = set(APIs["object_ids"]) | gt_obj
                    # calculate metircs
                    TP_list = pred_obj & gt_obj
                    pred_F = set(range(N_total)) - pred_obj
                    F_gt = set(range(N_total)) - gt_obj
                    TF_list = pred_obj & F_gt
                    FP_list = pred_F & F_gt
                    FN_list = gt_obj - pred_obj
                    # acc
                    _acc = (len(TP_list) + len(FP_list)) / N_total
                    all_acc.append(_acc)
                    # recall
                    _recall = len(TP_list)/len(gt_obj)
                    all_recall.append(_recall)
                    # precision
                    if len(pred_obj) == 0:
                        _precision = 0
                    else:
                        _precision = len(TP_list)/len(pred_obj)
                    all_precision.append(_precision)
                    # specificity
                    _spec = len(FP_list) / len(F_gt)
                    all_spec.append(_spec)
                    # F1
                    # _f1 = 2*len(TP_list) / (2*len(TP_list) + len(FP_list) + len(FN_list))
                    if (_recall + _precision) != 0:
                        _f1 = 2*_recall*_precision / (_recall + _precision)
                    else:
                        _f1 = 0
                    all_F1.append(_f1)
            
        print("recall: {0}, \n precision: {1}".format(np.mean(all_recall), np.mean(all_precision)))
        metrics = [
            np.mean(all_acc),
            np.mean(all_recall),
            np.mean(all_precision),
            np.mean(all_spec),
            np.mean(all_F1)
        ]
        
        with open("./results/{0}{1}{2}_metrics.csv".format(scene_name, evaluate_style, th), "w") as f:
            writer = csv.writer(f)
            for value in metrics:
                writer.writerow([value])
        print("save to ./results/{0}{1}{2}_metrics.csv".format(scene_name, evaluate_style, th))

    return metrics


def ablation_study(scene, Q_origin, N_origin, T_origin):
    # set hyper-parameters
    ablation_Q = range(2, 21, 2)
    ablation_N = range(2,13,1)  # 3 or 5 for single agent
    ablation_T = range(100, 1401, 100)
    # ablation_T = range(400, 1401, 100)
    ablation_T = [30,40,50,60,70,80,90,100,120,140,160,180,200,250,300,400,500,600,700,800,900,1000,1200,1400]

    # for scene in scenes:
    # # test Q
    # N = N_origin
    # T = T_origin
    # for Q in ablation_Q:
    #     data_simulator(scene, Q, N, T)
    
    # # test N
    # Q = Q_origin
    # T = T_origin
    # for N in ablation_N:
    #     data_simulator(scene, Q, N, T)
    
    # test T
    Q = Q_origin
    N = N_origin
    for T in ablation_T:
        data_simulator(scene, Q, N, T)
    
    return


def ablation_plot(metrics_array, ablation_range, xlabel):
    # 画图
    # 绘制每列数据的曲线图
    column_names = [
        "Accuracy",
        "Recall",
        "Precision",
        "Specificity",
        "F1 score"
    ]
    # for i in range(metrics_array.shape[1]):
    #     x = ablation_range  # 横坐标值
    #     y = metrics_array[:, i]  # 纵坐标值

    #     # 使用 Matplotlib 绘制曲线图
    #     plt.plot(x, y, label=column_names[i])

    # prepare seaborn data
    ablation_values, column_indices = np.meshgrid(ablation_range, np.arange(metrics_array.shape[1]))
    bbb = metrics_array.copy()
    bbb = bbb.swapaxes(0, 1)
    sns_data = {
        'Ablation_Range': ablation_values.flatten(),
        'Metrics': bbb.flatten(),
        'Column': column_indices.flatten()
    }
    # sns.set_style("whitegrid")
    plot = sns.relplot(x='Ablation_Range', y='Metrics', 
                kind='line', data=sns_data, 
                # col="Column", 
                hue="Column", style="Column",
                ci="sd",
                # legend=False,
                height=6, aspect=1.5,
                linewidth=2.5, markers=True, dashes=False)
    # plt.gcf().set_facecolor("white")
    plot.ax.set_facecolor("#f0f0f0")
    plot.ax.grid(color='white', linestyle='-', linewidth=1)

    plt.xticks(ablation_range)

    # 添加标签和图例
    plt.xlabel(xlabel)
    plt.ylabel('Evaluation Metrics')
    # plt.title('')
    # # plt.legend(loc='lower left')

    # 显示图形
    plt.show()


def ablation_evaluation(scene, ablation_flag, Q_origin, N_origin, T_origin):
    # set hyper-parameters
    ablation_Q = range(2, 21, 2)
    ablation_N = range(2, 13, 1)  # 3 or 5 for single agent
    # ablation_N = range(10, 101, 10)
    ablation_T = range(400, 1401, 100)
    # ablation_T = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300,400,500,600,700,800,900,1000,1200,1400]

    # evaluate Q
    N = N_origin
    T = T_origin
    Q_metrics = []
    for Q in ablation_Q:
        Q_metrics.append(evaluate(scene, ablation_flag, False, Q, N, T))  # shape=(5,)
    ablation_plot(np.array(Q_metrics), ablation_Q, 'Q')
    
    
    # test N
    Q = Q_origin
    T = T_origin
    N_metrics = []
    for N in ablation_N:
        N_metrics.append(evaluate(scene, ablation_flag, False, Q, N, T))
    ablation_plot(np.array(N_metrics), ablation_N, 'N')
    
    # test T
    Q = Q_origin
    N = N_origin
    T_metrics = []
    for T in ablation_T:
        T_metrics.append(evaluate(scene, ablation_flag, False, Q, N, T))
    ablation_plot(np.array(T_metrics), ablation_T, 'T')
    
    return

    
# 读取单智能体设定
# franka 机械臂 joints, 0=x, 1=y, 2=z
franka = json.load(open("./utils/franka.json", "r")) 
# 设置随机种子
np.random.seed(7)

if __name__ == "__main__":
    scenes = [
        # ["3d_rotation"],  # 单智能体
        ["2d_position"],
        ["3d_position"],
        ["light"],
        # ["2d_position", "light"],
        # ["3d_position", "light"],
        # ["2d_position", "3d_position"],
        # ["2d_position", "3d_position", "light"]
    ]

    Q_origin = 10
    if "rotation" in scenes[0][0]:
        N_origin = 3
    else:
        N_origin = 20
    T_origin = 1000

    for scene in scenes:
        # data_simulator(scene, Q_origin, N_origin, T_origin)
        # ablation_study(scene, Q_origin, N_origin, T_origin)
        # evaluate(scene, ablation_flag=False, cross_flag=True)
        # ablation_evaluation(scene, ablation_flag=True, Q_origin=Q_origin, N_origin=N_origin, T_origin=T_origin)
        evaluate_mirror(scene, ablation_flag=False, only_mirror_flag=True, Q=Q_origin, N=N_origin, T=T_origin)