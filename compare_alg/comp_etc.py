import copy
import json
import os

import numpy as np
import pandas as pd

np.random.seed(1)  # 设置随机种子


def change_dict(old_dict, change):
    new_dict = copy.deepcopy(old_dict)
    for key, value in change.items():
        if isinstance(value, dict) and key in new_dict and isinstance(new_dict[key], dict):
            new_dict[key] = change_dict(new_dict[key], value)
        else:
            new_dict[key] = value
    return new_dict


def xls_read(filename: str) -> np.array:
    return np.array(pd.read_excel(filename))


def json_read(filename: str):
    with open(filename, 'r', encoding='utf8') as f:
        json_data = json.load(f)
    return json_data


def mkdir(dir_path):
    # make sure the directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def txt_save(filename, data):
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError

    if filename[-4:] != '.txt':
        filename += '.txt'
    json_data = json.dumps(data, default=convert_numpy, indent=4)
    with open(filename, "w") as file:
        file.write(json_data)


# def list_save(exp_name: str, index_name: str, data: list, log_dir='log/'):  # 保存log
#     log_path = log_dir + exp_name if log_dir[-1] == '/' else log_dir + '/' + exp_name
#     mkdir(log_path)
#     filename = log_path + '/' + index_name
#     if filename[-4:] != '.txt':
#         filename += '.txt'
#     file = open(filename, 'w+')
#     for i in range(len(data)):
#         s = str(data[i]).replace('[', '').replace(']', '')
#         s = s.replace("'", '').replace(',', '') + '\n'
#         file.write(s)
#     file.close()


def single_generate_flow_most_rou(rou_path: str, total_car_num: list, penetration: float, bias_ratio: float):  # 生成车流rou.xml
    """
    根据入口和出口决定驶入车道,右、直、左分别为0,1,2
                | 1 |
            ____|   |____
            0     △     2
            ````|   |````
                | 3 |
    """
    # rou_path = 'sumo_sim_env/rou/' + self.rou + n_file + '.rou.xml'
    Dict = {'0,3': 0, '0,2': 1, '0,1': 2,
            '1,0': 0, '1,3': 1, '1,2': 2,
            '2,1': 0, '2,0': 1, '2,3': 2,
            '3,2': 0, '3,1': 1, '3,0': 2}

    points = np.linspace(0, 3000, len(total_car_num) + 1)
    time_period = [[points[i], points[i + 1]] for i in range(len(points) - 1)]

    with open(rou_path, 'w', encoding='utf-8') as f:
        # 按格式写入车辆类型的最大加速度，最大减速度，车长度，最大速度，未装备自动路由设备
        f.write('<routes>\n' +
                ' <vType id="CAV" accel="3" decel="8" length="4" maxSpeed="15" reroute="false"' +
                ' color="1,0,0" carFollowModel="CACC" probability="' + str(penetration) + '" />\n' +
                ' <vType id="HDV" accel="3" decel="8" length="4" maxSpeed="15" reroute="false"' +
                ' probability="' + str(1 - penetration) + '" />\n' +
                ' <vTypeDistribution id="typedist1" vTypes="CAV HDV" />\n\n')

        most_edge_in = np.random.choice(4)  # 按概率分布随机选择驶入口
        most_remain_edge = [0, 1, 2, 3]
        most_remain_edge.remove(most_edge_in)
        most_remain_edge.remove((most_edge_in + 3) % 4)
        most_edge_out = np.random.choice(most_remain_edge)  # 按概率分布随机选择驶入口
        print(most_edge_in, most_edge_out)

        count = 0  # 记录已写入的车辆数
        for time in range(len(time_period)):
            depart_time = np.random.randint(time_period[time][0], time_period[time][1], total_car_num[time])  # 驶入时间
            depart_time.sort()
            for time_car in range(total_car_num[time]):
                count += 1
                # edge_in, edge_out = np.random.choice(range(0, 4), 2, False)  # 随机选择驶入口和驶出口,无放回*

                if np.random.random() < bias_ratio and depart_time[time_car] >= 1500:
                    edge_in = most_edge_in
                    edge_out = most_edge_out
                else:
                    edge_in = np.random.choice(4)  # 按概率分布随机选择驶入口
                    remain_edge = np.delete(np.array([0, 1, 2, 3]), edge_in)
                    edge_out = np.random.choice(remain_edge)  # 随机选择驶出口

                lane_in = Dict[str(edge_in) + ',' + str(edge_out)]  # 由于不考虑变道模型，故需要限制驶入车道
                f.write('  <vehicle id="car_' + str(count) + '" depart="' + str(depart_time[time_car]) +
                        '" departLane="' + str(lane_in) + '" arrivalLane="' + str(np.random.randint(3)) +
                        '" departSpeed="max" type="typedist1">\n' +
                        '    <route edges="edge_' + str(edge_in) + ' -edge_' + str(edge_out) + '"/>\n' +
                        '  </vehicle>\n\n')
        f.write('</routes>\n')


if __name__ == "__main__":
    np.random.seed(3407)  # 设置随机种子
    for i in range(15):
        n_file = i + 1
        rou_file_num = [str(n_file) if n_file >= 10 else '0' + str(n_file)][0]
        # single_generate_flow_most_rou(rou_path='../sumo_sim_env/single/new/rou.rou' + rou_file_num + '.xml',  # seed 1
        single_generate_flow_most_rou(rou_path='../sumo_sim_env/single/new_test/rou.rou' + rou_file_num + '.xml',
                                      total_car_num=[40, 60, 100, 200, 500, 200, 100, 150, 210, 120, 90, 30],
                                      penetration=0.2,
                                      bias_ratio=0.16)
        # new       # new_test
        # 1 3       # 3 0
        # 2 0       # 2 3
        # 1 2       # 3 0
        # 1 2       # 1 2
        # 3 1       # 2 0
        # 1 3       # 2 0
        # 3 0       # 2 3
        # 2 0       # 3 0
        # 1 2       # 0 1
        # 0 2       # 2 3
        # 2 0       # 1 2
        # 2 0       # 1 3
        # 0 1       # 1 2
        # 0 2       # 0 2
        # 3 1       # 0 1
