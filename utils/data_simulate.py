# generate simulated positions

import numpy as np
import pandas as pd
import os
import json

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
    
    def update(self, delta_state):
        self.state += delta_state
    
    def call_api(self, api):
        for id in range(len(api.object_ids)):
            self.state[api.object_ids[id], :] += api.deltas[id, :]


class Light_State():
    def __init__(self, init_state):
        self.state = init_state
        self.N = init_state.shape[0]
        self.K = init_state.shape[1]
    
    def update(self, delta_state):
        self.state += delta_state
    
    def call_api(self, api):
        for id in range(len(api.object_ids)):
            self.state[api.object_ids[id], :] += api.deltas[id, :]


class API():
    def __init__(self, object_ids, feature_num):
        self.object_ids = object_ids
        self.feature_num = feature_num
        self.deltas = np.zeros((len(object_ids), feature_num))

    def update(self, deltas):
        self.deltas = deltas


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
            deltas = np.random.rand(object_num, K)  # [0,1)
            # add bias to meet min&max
            deltas = deltas * (max - min) + min
        api.update(deltas=deltas)
        APIs.append(api)
    return APIs



def generate_3d_position(save_path, APIs, APIlist, N=20, K_start=0, K_end=3, T=1000, min=-1, max=1):
    # N: number of objects
    # K: number of features
    # A: API list. Q = lenth(A)
    # T: number of time steps
    born_area_min = -50
    born_area_max = 50

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
        env_deltas = np.random.rand(N, K) * (max - min) + min
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

    
    # save ground truth
    ran = [[N],
           APIlist.tolist(),
           [K],
           [len(APIs)-1]
    ]
    list_data = []
    for k in range(K):
        _obj_ids_per_feature = []
        for api in APIs:
            _obj_ids_per_feature.append(api.object_ids)
        list_data.append(_obj_ids_per_feature[1:])  # ignore first api
    # for api in APIs:
    #     _obj_ids_per_feature = []
    #     for k in range(K):
    #         _obj_ids_per_feature.append(api.object_ids)
    #     list_data.append(_obj_ids_per_feature)
    json.dump(ran, open(save_path + "/ran.json", "w"), indent=4)
    json.dump(list_data, open(save_path + "/list_data.json", "w"), indent=4)


def generate_light(save_path, APIs, APIlist, N=20, K_start=0, K_end=1, T=100, min=-1, max=1):

    # N: number of objects
    # K: number of features
    # A: API list. Q = lenth(A)
    # T: number of time steps
    born_area_min = -50
    born_area_max = 50

    # prepare columns.
    K = K_end - K_start
    columns = []
    for t in range(T+1):
        columns.append("V%s" % str(t))
    indexes = []
    for n in range(N):
        indexes.append("O%s" % str(n+1))

    dfs = []
    born_spots = np.random.rand(N, 2) * (born_area_max - born_area_min) + born_area_min
    init_states = np.zeros((N, K))
    for k in range(K_start, K_end):
        dfs.append(pd.DataFrame(columns=columns, index=indexes))
        # initialize
        init_states[:, k] = np.random.randint(low=0, high=2, size=(N))
        dfs[k].loc[:, "V0"] = init_states[:, k].copy()
    obj_state = State(init_state=init_states)

    for t in range(1,T+1):
        # # generate N objects' features
        # env_deltas = np.random.rand(N, K) * (max - min) + min
        # obj_state.update(delta_state=env_deltas)

        # generate N objects' APIs
        api_id = APIlist[t-1]

        # call APIs once every time step
        obj_state.call_api(api=APIs[api_id])

        # check validation, cut those more than 1
        obj_state.state[np.where(obj_state.state > 0)] = 1
        obj_state.state[np.where(obj_state.state < 0)] = 0

        # save to dataframe
        for k in range(K_start, K_end):
            dfs[k].loc[:, "V%s" % str(t)] = obj_state.state[:, k].copy()
    
    # print(dfs[0])

    print("save to %s" % save_path)
    # for k in range(K_start, K_end):
    #     _path = save_path + "/feature_{0}.xlsx".format(str(k+1))
    #     dfs[k].to_excel(_path, sheet_name="Sheet1")
    # # save json for zhenlaing
    # for k in range(K_start, K_end):
    #     _path = save_path + "/feature_{0}.json".format(str(k+1))
    #     dfs[k].to_json(_path, force_ascii=False)
    # save json for zhenlaing
    for k in range(K_start, K_end):
        _path = save_path + "/feature_{0}.txt".format(str(k+1))
        dfs[k].to_csv(_path, sep=' ', index=False)

    # save ground truth for zhenlaing
    ran = {"obj_num": N,
           "apiList": APIlist.tolist(),
           "feature_num": K,
           "api_num": len(APIs)-1
    }
    list_data = []
    for k in range(K):
        _obj_ids_per_feature = []
        for api in APIs:
            _obj_ids_per_feature.append(api.object_ids)
        list_data.append(_obj_ids_per_feature[1:])  # ignore first api
    # for api in APIs:
    #     _obj_ids_per_feature = []
    #     for k in range(K):
    #         _obj_ids_per_feature.append(api.object_ids)
    #     list_data.append(_obj_ids_per_feature)
    json.dump(born_spots.tolist(), open(save_path + "/born_spots.json", "w"), indent=4)
    json.dump(ran, open(save_path + "/ran.json", "w"), indent=4)
    json.dump(list_data, open(save_path + "/list_data.json", "w"), indent=4)


    return 1


def generate_single_agent(save_path, APIs, APIlist, N=20, K_start=0, K_end=1, T=100, min=-1, max=1):

    # N: number of objects
    # K: number of features
    # A: API list. Q = lenth(A)
    # T: number of time steps
    born_area_min = -50
    born_area_max = 50

    # prepare columns.
    K = K_end - K_start
    columns = []
    for t in range(T+1):
        columns.append("V%s" % str(t))
    indexes = []
    for n in range(N):
        indexes.append("O%s" % str(n+1))

    dfs = []
    born_spots = np.random.rand(N, 2) * (born_area_max - born_area_min) + born_area_min
    init_states = np.zeros((N, K))
    for k in range(K_start, K_end):
        dfs.append(pd.DataFrame(columns=columns, index=indexes))
        # initialize
        init_states[:, k] = np.random.randint(low=0, high=2, size=(N))
        dfs[k].loc[:, "V0"] = init_states[:, k].copy()
    obj_state = State(init_state=init_states)

    for t in range(1,T+1):
        # # generate N objects' features
        # env_deltas = np.random.rand(N, K) * (max - min) + min
        # obj_state.update(delta_state=env_deltas)

        # generate N objects' APIs
        api_id = APIlist[t-1]

        # call APIs once every time step
        obj_state.call_api(api=APIs[api_id])

        # check validation, cut those more than 1
        obj_state.state[np.where(obj_state.state > 0)] = 1
        obj_state.state[np.where(obj_state.state < 0)] = 0

        # save to dataframe
        for k in range(K_start, K_end):
            dfs[k].loc[:, "V%s" % str(t)] = obj_state.state[:, k].copy()
    
    # print(dfs[0])

    print("save to %s" % save_path)
    # for k in range(K_start, K_end):
    #     _path = save_path + "/feature_{0}.xlsx".format(str(k+1))
    #     dfs[k].to_excel(_path, sheet_name="Sheet1")
    # save json for liang
    for k in range(K_start, K_end):
        _path = save_path + "/feature_{0}.json".format(str(k+1))
        dfs[k].to_json(_path, force_ascii=False, indent=4)

    # save ground truth
    ran = [[N],
           APIlist.tolist(),
           [K],
           [len(APIs)-1]
    ]
    list_data = []
    for k in range(K):
        _obj_ids_per_feature = []
        for api in APIs:
            _obj_ids_per_feature.append(api.object_ids)
        list_data.append(_obj_ids_per_feature[1:])  # ignore first api
    # for api in APIs:
    #     _obj_ids_per_feature = []
    #     for k in range(K):
    #         _obj_ids_per_feature.append(api.object_ids)
    #     list_data.append(_obj_ids_per_feature)
    json.dump(born_spots.tolist(), open(save_path + "/born_spots.json", "w"), indent=4)
    json.dump(ran, open(save_path + "/ran.json", "w"), indent=4)
    json.dump(list_data, open(save_path + "/list_data.json", "w"), indent=4)


    return 1


def data_simulator(scene):
    Round = 10

    for r in range(Round):
        # initialization
        Q = 10
        N = 20
        T = 1000
        feature_id = 0
        save_path = "./data/{0}_round{1}".format(scene[0], str(r))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for _scene in scene:
            if "2d" in _scene:
                K = 2
            elif "3d" in _scene:
                K = 3
            if "light" in _scene:
                K = 1

            if "light" in _scene:
                APIs = generate_api(Q=Q, N=N, O=N, K=K, min=-1, max=1, int_flag=True)
                APIlist = np.random.choice(a=len(APIs), size=T)  # API id = 0, don't call API
                generate_light(save_path, APIs=APIs, APIlist=APIlist, N=20, K_start=0, K_end=K, T=T)
            elif "rotation" in _scene:
                APIs = generate_api(Q=Q, N=N, O=N, K=K, min=-1, max=1, int_flag=True)
                APIlist = np.random.choice(a=len(APIs), size=T)  # API id = 0, don't call API
                generate_single_agent(save_path, APIs=APIs, APIlist=APIlist, N=20, K_start=0, K_end=K, T=T)
            else:
                APIs = generate_api(Q=Q, N=N, O=N, K=K, min=-1, max=1, int_flag=False)
                APIlist = np.random.choice(a=len(APIs), size=T)  # API id = 0, don't call API
                generate_3d_position(save_path, APIs=APIs, APIlist=APIlist, N=20, K_start=0, K_end=K, T=T)


if __name__ == "__main__":
    scenes = [
        # ["3d_rotation"],  # 单智能体
        # ["2d_position"],
        ["3d_position"],
        # ["light"]
    ]

    for scene in scenes:
        data_simulator(scene)