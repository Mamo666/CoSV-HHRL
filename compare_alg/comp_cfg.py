
from comp_etc import change_dict


def get_agent_configs(modify_dict, is_cotv):
    light, cav = modify_dict['light'], modify_dict['cav']
    light_cfg = cotv_light_configs if is_cotv else hh_light_configs
    return change_dict(light_cfg, light), change_dict(CAV_configs, cav)


env_configs = {
    # 信号灯时长设置
    'yellow': 3,  # 全红灯时长可以设为0，但黄灯不能为0
    'red': 2,
    'min_green': 5,
    'max_green': 35,

    # 针对单路口的配置
    'single': {
        'base_cycle_length': 100,  # 基准周期时长，用以进行标准化处理
        'base_lane_length': 300,  # 基准道路长度，用以进行标准化处理
        'max_speed': 13.89,  # 道路允许最大车速, m/s
        'max_acc': 3,  # 最大加速度, m/s^2
        'car_length': 4,    # 车辆长度均为4m
        'time_step': 1,  # 仿真步长为1s

        # 文件路径设置
        'sumocfg_path': '../sumo_sim_env/collision_env_comp.sumocfg',    # 从代码到env.sumocfg的路径
        'rou_path': 'single/6_0.3/',
        'net_path': 'single/no_lane_change.net.xml',  # 路网文件只会有一个,故写全
    },

    # 针对2*2路网的配置
    'four': {
        'base_cycle_length': 100,  # 基准周期时长，用以进行标准化处理
        'base_lane_length': 300,  # 基准道路长度，用以进行标准化处理
        'max_speed': 13.89,  # 道路允许最大车速, m/s
        'max_acc': 3,  # 最大加速度, m/s^2
        'time_step': 1,  # 仿真步长为1s
        'car_length': 4,  # 车辆长度均为4m

        # 文件路径设置
        'sumocfg_path': '../sumo_sim_env/collision_env_comp.sumocfg',  # 从代码到env.sumocfg的路径
        'rou_path': 'four/6_0.3/',
        'net_path': 'four/no_lane_change_2_2.net.xml',
        'ctrl_lane_path': '../sumo_sim_env/four/control_links.json',  # node-incoming_lane
    },

    # 针对108路网的配置
    'Chengdu': {
        'base_cycle_length': 100,  # 基准周期时长，用以进行标准化处理
        'base_lane_length': 100,    # 基准道路长度，用以进行标准化处理
        'max_speed': 13.89,         # 道路允许最大车速, m/s
        'max_acc': 3,               # 最大加速度, m/s^2
        'time_step': 1,             # 仿真步长为1s
        'car_length': 5,    # 车辆长度均为4m

        # 文件路径设置
        'sumocfg_path': '../sumo_sim_env/collision_env.sumocfg',    # 从代码到env.sumocfg的路径
        'holon_dir': '../sumo_sim_env/Chengdu/',
        'rou_path': 'Chengdu/rou_high/',
        'net_path': 'Chengdu/net.net.xml',
        'add_lane_path': '../sumo_sim_env/Chengdu/intersection_lane.xlsx',  # lane-add_lane
        'ctrl_lane_path': '../sumo_sim_env/Chengdu/control_links.json',     # node-incoming_lane
        'del_intersection': ['n_168', 'n_208', 'n_210', 'n_214', 'n_226', 'n_250', 'n_276', 'n_285', 'n_299', 'n_303',
                             'n_304', 'n_307', 'n_310', 'n_311', 'n_334', 'n_336', 'n_344', 'n_345', 'n_326'],  # 3-legs
    }
}


hh_light_configs = {
    'train_model': True,
    'load_model_name': None,
    'use_macro': True,  # 一般别改
    'use_micro': True,
    'cav_head': False,   # True 则只控制CAV，反之则任意车都能控制, 注意两个要一起修改

    'macro': {
        'obs_dim': 4,   # 路口智能体的状态维度 [下一相位one-hot, 各相位车辆数, 各相位排队数]
        'state_dim': 32,        # RNN层的输出维度
        'act_dim': 5,           # 路口智能体的动作空间 [下个相位持续时间]
        'T': 1,     # 这里不能改！
        'hidden_dim': [400, 300],    # actor和critic网络隐藏层维度一样
        'memory_capacity': 5000,
        'learn_start_ratio': 0.2,
        'var': .6,
    },

    'micro': {
        'obs_dim': 3,  # 路口智能体的状态维度 [各相位车辆数, 各相位排队数]
        'state_dim': 32,  # RNN层的输出维度
        'act_dim': 1,  # 路口智能体的动作空间 [下个相位]
        'T': 1,     # 这里不能改！
        'hidden_dim': [400, 300],  # actor和critic网络隐藏层维度一样。
        'memory_capacity': 20000,
        'learn_start_ratio': 0.2,
        'var': .6,
    },

    # 信号灯时长设置
    'yellow': env_configs['yellow'],  # 全红灯时长可以设为0，但黄灯不能为0
    'red': env_configs['red'],
    'min_green': env_configs['min_green'],  # 单个相位绿灯时间最小值,5s
    'max_green': env_configs['max_green'],  # 单个相位绿灯时间最大值,35s

    'var': .6,
    'tau': 0.005,  # 软更新参数
    'gamma': 0.95,  # .95  20步
    'batch_size': 64,  # 批大小
    'critic_hidden_dim': [400, 300],
    'actor_learning_rate': 0.0001,
    'critic_learning_rate': 0.001,
    'actor_scheduler_step': 200,
    'critic_scheduler_step': 400,

}

CAV_configs = {
    'use_CAV': True,    # 无需更改
    'cav_head': False,   # True 则只控制CAV，反之则任意车都能控制
    'train_model': True,
    'load_model_name': None,

    'cav': {
        'obs_dim': 5 + 2,   # 车辆智能体的状态维度 [与前车距离、与前车速度差、与路口距离、当前车速、当前加速度、信号灯指示符、倒计时]
        'state_dim': 32,   # LSTM输出维度   # !16!
        'act_dim': 1,    # 车辆智能体的动作空间 [决定车辆加速度的大小]
        'T': 1,    # 不宜设置过大，因为要攒够这么多步的obs才能开始决策和学习
        'hidden_dim': [128, 128],  # actor和critic网络隐藏层维度一样。
    },
    'hidden_dim': [128, 128],   # !

    'var': .6,
    'tau': 0.005,           # 软更新参数
    'gamma': 0.95,           # 比路灯短视! 10步
    'batch_size': 64,       # 批大小
    'memory_capacity': 20000,    # fixed goal 7
    'learn_start_ratio': 0.2,    # fixed goal
    'actor_learning_rate': 0.0001,
    'critic_learning_rate': 0.001,
    'actor_scheduler_step': 2000,   # !
    'critic_scheduler_step': 1500,  # !
}


cotv_light_configs = {
    'train_model': True,
    'load_model_name': None,
    'use_light': True,  # 一般别改
    'cav_head': False,   # True 则只控制CAV，反之则任意车都能控制, 注意两个要一起修改

    'cotv': {
        'obs_dim': 44,   # 路口智能体的状态维度 [下一相位one-hot, 各相位车辆数, 各相位排队数]
        'state_dim': 32,        # RNN层的输出维度(因为T=1，实际上用不到)
        'act_dim': 1,           # 路口智能体的动作空间 [是否切相位]
        'T': 1,     # 这里不能改！
        'hidden_dim': [400, 300],    # actor和critic网络隐藏层维度一样
        'memory_capacity': 5000,
        'learn_start_ratio': 0.2,
        'var': .6,
    },

    # 信号灯时长设置
    'yellow': env_configs['yellow'],  # 全红灯时长可以设为0，但黄灯不能为0
    'red': 0,
    'green': env_configs['min_green'],  # 单个相位绿灯时间最小值,5s

    'var': .6,
    'tau': 0.005,  # 软更新参数
    'gamma': 0.95,  # .95  20步
    'batch_size': 64,  # 批大小
    'critic_hidden_dim': [400, 300],
    'actor_learning_rate': 0.0001,
    'critic_learning_rate': 0.001,
    'actor_scheduler_step': 200,
    'critic_scheduler_step': 400,

}