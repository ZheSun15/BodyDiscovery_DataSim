# generate simulated positions

import numpy as np
import pandas as pd
import os
import json
import glob

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
    
    # save csv for xiaoxiao
    for k in range(K_start, K_end):
        _path = save_path + "/feature_{0}.csv".format(str(k+1))
        dfs[k].to_csv(_path, sheet_name="feature_{0}".format(str(k+1)))


    
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
    api_gt = []
    for api in APIs[1:]:  # ignore first api
        api_gt.append(api.to_dict())
    api_gt.append(all_body)
    json.dump(ran, open(save_path + "/ran.json", "w"), indent=4)
    json.dump(list_data, open(save_path + "/list_data.json", "w"), indent=4)
    json.dump(api_gt, open(save_path + "/api_gt.json", "w"), indent=4)


def generate_light(save_path, APIs, APIlist, N=20, K_start=0, K_end=1, T=100, min=-1, max=1):

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
    born_spots = np.random.rand(N, 2) * (born_area_max - born_area_min) + born_area_min
    init_states = np.zeros((N, K))
    for k in range(K_start, K_end):
        dfs.append(pd.DataFrame(columns=columns, index=indexes))
        # initialize
        init_states[:, k] = np.random.randint(low=0, high=2, size=(N))
        dfs[k].loc[:, "V0"] = init_states[:, k].copy()
    obj_state = State(init_state=init_states)

    for t in range(1,T+1):
        # generate N objects' features
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
           APIlist.tolist(),
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


def generate_single_agent(save_path, APIs, APIlist, N=20, K_start=0, K_end=1, T=100, min=-1, max=1):

    # N: number of objects
    # K: number of features
    # A: API list. Q = lenth(A)
    # T: number of time steps
    born_area_min = -50
    born_area_max = 50

    # franka 机械臂 joints, 0=x, 1=y, 2=z
    franka = {
        "panda_link1": {
            "init": [0,0,0],
            "control": 2,  # z axis
            "range": [0, 85],
        },
        "panda_link2": {
            "init": [-90,0,0],
            "control": 2,  # z axis
            "range": [-30, 30],
        },
        "panda_link3": {
            "init": [90,0,0],
            "control": 2,  # z axis
            "range": [0, 85],
        },
        "panda_link4": {
            "init": [90,0,0],
            "control": 2,  # z axis
            "range": [0, 75],
        },
        "panda_link5": {
            "init": [-90,0,0],
            "control": 2,  # z axis
            "range": [0, 85],
        },
        "panda_link6": {
            "init": [90,0,0],
            "control": 2,  # z axis
            "range": [-85, 0],
        },
        "panda_link7": {
            "init": [90,0,0],
            "control": 2,  # z axis
            "range": [-80, -30],
        },
        "panda_leftfinger": {
            "init": [0,0,0],
            "control": 0,  # x axis
            "range": [-20, 0],
        },
        "panda_rightfinger": {
            "init": [0,0,0],
            "control": 0,  # x axis
            "range": [0, 20],
        }
    }

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
                generate_light(save_path, APIs=APIs, APIlist=APIlist, N=N, K_start=0, K_end=K, T=T)
            elif "rotation" in _scene:
                APIs = generate_api(Q=Q, N=N, O=N, K=K, min=-1, max=1, int_flag=True)
                APIlist = np.random.choice(a=len(APIs), size=T)  # API id = 0, don't call API
                generate_single_agent(save_path, APIs=APIs, APIlist=APIlist, N=N, K_start=0, K_end=K, T=T)
            else:
                APIs = generate_api(Q=Q, N=N, O=N, K=K, min=-1, max=1, int_flag=False)
                APIlist = np.random.choice(a=len(APIs), size=T)  # API id = 0, don't call API
                generate_3d_position(save_path, APIs=APIs, APIlist=APIlist, N=N, K_start=0, K_end=K, T=T, min=-1, max=1)


def evaluate(scene):

    p_th = 0.05

    prediction_root = "./prediction"
    gt_root = "./data"

    all_acc = []
    all_recall = []
    all_precision = []
    all_AP = []
    all_F1 = []
    for _scene in scene:
        result_pths = sorted(glob.glob(prediction_root + "/{0}_*_plist.json".format(_scene)))
        for result_pth in result_pths:
            # clear
            acc_per_round = []
            recall_per_round = []
            precision_per_round = []
            AP_per_round = []
            F1_per_round = []

            file_name = result_pth.split("/")[-1]
            # find ground truth path
            gt_folder = "{0}/{1}".format(gt_root, file_name.replace("_plist.json", ""))
            gt_list = json.load(open(gt_folder + "/api_gt.json", "r"))
            ran = json.load(open(gt_folder + "/ran.json", "r"))
            N = ran[0][0]
            # load prediction result
            pred = json.load(open(result_pth, "r"))
            
            for api_id in range(len(gt_list) - 1):
                pred_obj = set()
                for k in range(len(pred)):
                    _pred_array = np.array(pred[k])
                    _p = _pred_array[:, api_id]
                    pred_obj = set(np.where(_p < p_th)[0]) | pred_obj
                # calculate metircs
                TP_list = pred_obj & set(gt_list[api_id]["object_ids"])
                pred_F = range(N) - pred_obj
                F_gt = range(N) - set(gt_list[api_id]["object_ids"])
                FP_list = pred_F & F_gt
                # acc
                _acc = (len(TP_list) + len(FP_list)) / N
                acc_per_round.append(_acc)
                all_acc.append(_acc)
                # recall
                _recall = len(TP_list)/len(gt_list[api_id]["object_ids"])
                all_recall.append(_recall)
                recall_per_round.append(_recall)
                # precision

                if len(pred_obj) == 0:
                    all_precision.append(0)
                    precision_per_round.append(0)
                else:
                    all_precision.append(len(TP_list)/len(pred_obj))
                    precision_per_round.append(len(TP_list)/len(pred_obj))
            
            # print("For {0}:\n    m-recall: {1}, m-precision: {2}".format(file_name, 
            #                                                             np.mean(recall_per_round), 
            #                                                             np.mean(precision_per_round)))
    
        print("recall: {0}, \n precision: {1}".format(np.mean(all_recall), np.mean(all_precision)))

                    


if __name__ == "__main__":
    scenes = [
        ["3d_rotation"],  # 单智能体
        # ["2d_position"],
        # ["3d_position"],
        # ["light"]
    ]

    # 设置随机种子
    np.random.seed(7)
    
    for scene in scenes:
        data_simulator(scene)
        # evaluate(scene)