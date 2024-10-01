"""
这版的Light无论如何只有一个网络(把TP去掉)
[T/P/tp/TP] [Gv/tgv/pgv/tpgv] [V/TV/PV/tpV]    # tpg:HATD3,v:worker,TPGV:TD3
"""

import numpy as np
from collections import deque
from comp_alg import TD3Single

np.random.seed(3407)  # 设置随机种子


class hhLightAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.use_macro = config['use_macro']
        self.use_micro = config['use_micro']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None
        self.cav_head = config['cav_head']

        config['macro']['memory_capacity'] = config['macro']['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配
        config['micro']['memory_capacity'] = config['micro']['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配

        self.macro_net = TD3Single(config, 'macro')
        self.micro_net = TD3Single(config, 'micro')

        self.save = lambda path, ep: (self.macro_net.save(path + 'macro_agent_' + self.holon_name + '_ep_' + str(ep)),
                                      self.micro_net.save(path + 'micro_agent_' + self.holon_name + '_ep_' + str(ep)))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else '99'
            self.macro_net.load('../model/{}/macro_agent_{}_ep_{}'.format(config['load_model_name'], self.holon_name, load_ep))
            self.micro_net.load('../model/{}/micro_agent_{}_ep_{}'.format(config['load_model_name'], self.holon_name, load_ep))

        self.var_t = config['macro']['var']
        self.var_p = config['micro']['var']

        self.min_green = config['min_green']
        self.max_green = config['max_green']
        self.yellow = config['yellow']
        self.red = config['red']
        self.micro_modify_bound = 3     # 论文把micro修改范围限制在[-3,3]

        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.r_t_list = []
        self.r_p_list = []

        self.time_index = {light: 0 for light in self.light_id}
        self.time_strategy = {light: [] for light in self.light_id}      # 用以存储每个信号灯的配时方案
        self.micro_action_flag = {light: [] for light in self.light_id}  # 用以指示当前相位是否进行了相位延长
        self.micro_action_car_name = {light: deque(maxlen=2) for light in self.light_id}    # 用来记录micro决策时离路口最近的车的id，用于比较决策后该车是否通过

    @property
    def pointer(self):  # 实际用途：在train时看要不要开GUI和训练过程中\r实时打印pointer看一下
        return self.macro_net.pointer

    @property
    def learn_begin(self):
        return self.macro_net.learn_begin

    def step(self, env):
        tl, pl = [], []
        for light in self.light_id:
            t, p = self._step(env, light)
            tl.append(t)
            pl.append(p)
        return tl, pl
        # return tl[0], pl[0]  # 只向外展示第一个路口的动作

    def _step(self, env, light):
        green_ratio, micro_actual_time = None, None
        """宏观智能体做决策"""
        if self.time_index[light] == 0:    # 当前信号灯需要重新配时，生成宏观智能体的配时策略
            if not self.use_macro or (not self.train_model and not self.load_model):
                ave_green = (self.max_green - self.min_green) / 2 + self.min_green
                strategy = [ave_green, self.yellow, self.red, ave_green, self.yellow, self.red,
                            ave_green, self.yellow, self.red, ave_green, self.yellow, self.red]
            else:
                o_t = env.get_macro_agent_state(light)
                self.o_t_list[light].append(o_t)

                if self.train_model:
                    if self.macro_net.pointer < self.macro_net.learn_begin and not self.load_model:
                        a_t = np.random.random(self.macro_net.a_dim) * 2 - 1
                    else:
                        a_t = self.macro_net.choose_action(o_t)
                    a_t = np.clip(np.random.normal(0, self.var_t, size=a_t.shape) + a_t, -1, 1)
                else:
                    a_t = self.macro_net.choose_action(o_t)
                self.a_t_list[light].append(a_t)
                action_v = (a_t + 1) / 2
                cycle = round(action_v[0] * (self.max_green * 4 - self.min_green * 4) + self.min_green * 4)
                # 将输出动作后4维归一化处理,并乘以周期长度,得到各相位的绿灯持续时长
                # green_ratio = [i / (sum(action_v[1:]) + 1e-8) * cycle for i in action_v[1:]]
                green_ratio = [round(action_v[i + 1] / (sum(action_v[1:]) + 1e-8) * cycle) for i in range(len(action_v) - 1)]
                # 把误差加到第一相位，一般误差一两秒
                green_ratio[0] += sum(green_ratio) - cycle
                strategy = [green_ratio[0], self.yellow, self.red, green_ratio[1], self.yellow, self.red,
                            green_ratio[2], self.yellow, self.red, green_ratio[3], self.yellow, self.red]
            self.time_strategy[light] = strategy

            r_t = env.get_light_reward(light)   # 直接复用我自己的压力值奖励。师兄那个只考虑了控制车道的，我这是edge
            self.r_t_list.append(r_t)
            if len(self.o_t_list[light]) >= 2:
                self.macro_net.store_transition(self.o_t_list[light][-2], self.a_t_list[light][-2], r_t, self.o_t_list[light][-1])
            if self.train_model and self.macro_net.pointer >= self.macro_net.learn_begin:
                self.var_t = max(0.01, self.var_t * 0.99)  # 0.9-40 0.99-400 0.999-4000
                self.macro_net.learn()

            self.time_index[light] = sum(strategy)
            self.micro_action_flag[light] = [False] * 4  # 在一个周期开始时，每个阶段决策指示符初始化

        if self.use_micro:
            micro_agent_action_time = [sum(self.time_strategy[light][:1]) - 5, sum(self.time_strategy[light][:4]) - 5,
                                       sum(self.time_strategy[light][:7]) - 5, sum(self.time_strategy[light][:10]) - 5]   # 找到微观智能体做决策的时间
            """微观智能体做决策"""
            if sum(self.time_strategy[light]) - self.time_index[light] in micro_agent_action_time:    # 微观智能体开始做决策
                phase_index = micro_agent_action_time.index(sum(self.time_strategy[light]) - self.time_index[light])  # 相位决策指示符
                if not self.micro_action_flag[light][phase_index]:  # 当该指示符没有被执行时，则进行微观智能体的决策操作(如果micro增加时间导致再次访问到这个时间，则不再次动作)
                    car_name_1, o_p = env.get_micro_agent_state(light, self.cav_head)   # 距离路口最近车辆状态作为微观智能体状态[停止线距离、车辆速度、车辆加速度]
                    self.o_p_list[light].append(o_p)
                    self.micro_action_car_name[light].append(car_name_1)

                    if o_p[0] == -1:    # 如果当前相位已经没有车，直接去掉剩余绿灯
                        a_p = -1
                    elif self.train_model:
                        if self.micro_net.pointer < self.micro_net.learn_begin and not self.load_model:
                            a_p = np.random.random(self.micro_net.a_dim) * 2 - 1
                        else:
                            a_p = self.micro_net.choose_action(o_p)
                        a_p = np.clip(np.random.normal(0, self.var_p, size=a_p.shape) + a_p, -1, 1)
                    else:
                        a_p = self.micro_net.choose_action(o_p)
                    self.a_p_list[light].append(a_p)

                    micro_actual_time = round(float(a_p) * self.micro_modify_bound)
                    self.time_strategy[light][phase_index] += micro_actual_time
                    self.time_index[light] += micro_actual_time                        # 剩余时间改变
                    self.micro_action_flag[light][phase_index] = True

            micro_agent_phase_end_time = [sum(self.time_strategy[light][:1]), sum(self.time_strategy[light][:4]),
                                          sum(self.time_strategy[light][:7]), sum(self.time_strategy[light][:10])]  # 相位切换时刻
            if sum(self.time_strategy[light]) - self.time_index[light] in micro_agent_phase_end_time:
                phase_index = micro_agent_phase_end_time.index(sum(self.time_strategy[light]) - self.time_index[light])  # 相位决策指示符
                if self.micro_action_flag[light][phase_index]:  # 当微观智能体决策过该相位，看效果获取奖励并存经验训练
                    last_o_p = (self.micro_action_car_name[light][-1], self.o_p_list[light][-1])
                    new_car, _ = env.get_micro_agent_state(light, self.cav_head)  # 距离路口最近车辆状态作为微观智能体状态[停止线距离、车辆速度、车辆加速度]
                    actual_time = round(float(self.a_p_list[light][-1]) * self.micro_modify_bound)
                    r_p = env.get_micro_agent_reward(last_o_p, new_car, actual_time)
                    self.r_p_list.append(r_p)
                    if len(self.o_p_list[light]) >= 2:
                        self.micro_net.store_transition(self.o_p_list[light][-2], self.a_p_list[light][-2], r_p, self.o_p_list[light][-1])
                    if self.train_model and self.micro_net.pointer >= self.micro_net.learn_begin:
                        self.var_p = max(0.01, self.var_p * 0.99)  # 0.9-40 0.99-400 0.999-4000
                        self.micro_net.learn()

        curr_index = 0
        while (sum(self.time_strategy[light]) - self.time_index[light] >= sum(self.time_strategy[light][:curr_index+1])
               and curr_index+1 < len(self.time_strategy[light])):
            curr_index += 1

        env.set_light_action(light, curr_index, self.time_strategy[light][curr_index])       # 设定当前相位配时时长
        self.time_index[light] -= 1

        return green_ratio, micro_actual_time

    def reset(self):
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.r_t_list = []
        self.r_p_list = []

        self.time_index = {light: 0 for light in self.light_id}
        self.time_strategy = {light: [] for light in self.light_id}      # 用以存储每个信号灯的配时方案
        self.micro_action_flag = {light: [] for light in self.light_id}  # 用以指示当前相位是否进行了相位延长
        self.micro_action_car_name = {light: deque(maxlen=2) for light in self.light_id}    # 用来记录micro决策时离路口最近的车的id，用于比较决策后该车是否通过


class CoTVLightAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.use_light = config['use_light']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None
        self.cav_head = config['cav_head']

        config['cotv']['memory_capacity'] = config['cotv']['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配

        self.network = TD3Single(config, 'cotv')

        self.save = lambda path, ep: self.network.save(path + 'light_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else '99'
            self.network.load('../model/{}/light_agent_{}_ep_{}'.format(config['load_model_name'], self.holon_name, load_ep))

        self.var = config['var']

        self.green = config['green']
        self.yellow = config['yellow']
        self.red = 0    # 全红

        self.o_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_list = {light: deque(maxlen=2) for light in self.light_id}
        self.r_list = []

        self.time_index = {light: 0 for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env):
        al = []
        for light in self.light_id:
            a = self._step(env, light)
            al.append(a)
        return al

    def _step(self, env, light):
        change_phase = None
        if self.time_index[light] == 0:
            if self.color[light] == 'y' and self.red != 0:  # 黄灯结束切红灯
                env.set_light_action(light, self.phase_list[light][-2] * 3 + 2, self.red)
                self.time_index[light], self.color[light] = self.red, 'r'
            elif self.color[light] == 'r' or (self.color[light] == 'y' and self.red == 0):  # 红灯结束或（黄灯结束且无全红相位）切绿灯
                env.set_light_action(light, self.phase_list[light][-1] * 3, self.green)
                self.time_index[light], self.color[light] = self.green, 'g'
            elif self.color[light] == 'g':     # 为保持时间尺度一致
                phase_next = (self.phase_list[light][-1] + 1) % 4  # # # To do: 可选相位 # # #

                o_l = env.get_cotv_light_obs(light, cav_only=self.cav_head)
                self.o_list[light].append(np.array(o_l))
                if self.network.pointer < self.network.learn_begin and not self.load_model:  # 随机填充
                    a_l = np.random.random(self.network.a_dim) * 2 - 1
                else:
                    a_l = self.network.choose_action(o_l)

                a_l = np.clip(np.random.normal(0, self.var, size=a_l.shape) + a_l, -1, 1) if self.train_model else a_l
                a_l = (a_l + 1) / 2
                change_phase = round(a_l[0])
                self.a_list[light].append(change_phase)    # todo 存实际动作还是真实输出？

                if change_phase == 1 and phase_next != self.phase_list[light][-1]:
                    self.phase_list[light].append(phase_next)

                    env.set_light_action(light, self.phase_list[light][-2] * 3 + 1, self.yellow)
                    self.time_index[light], self.color[light] = self.yellow, 'y'
                else:
                    env.set_light_action(light, self.phase_list[light][-1] * 3, self.green)
                    self.time_index[light] = self.green

                r_l = env.get_light_reward(light)
                self.r_list.append(r_l)
                if len(self.o_list[light]) >= 2:
                    self.network.store_transition(self.o_list[light][-2], self.a_list[light][-2], r_l, self.o_list[light][-1])
                if self.train_model and self.network.pointer >= self.network.learn_begin:
                    self.var = max(0.01, self.var * 0.99)  # 0.9-40 0.99-400 0.999-4000
                    self.network.learn()

        self.time_index[light] -= 1

        return change_phase

    def reset(self):
        self.o_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_list = {light: deque(maxlen=2) for light in self.light_id}
        self.r_list = []

        self.time_index = {light: 0 for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}


"""
    车辆智能体
"""


class hhCavAgent:
    """理论上整个路网只用一个cav智能体，但时间所限不改这个了，目前是一个分区一个cav智能体。"""
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.ctrl_all_lane = True   # COTV-MADRL原文应该控制所有入口车道的头车而不是当前相位车道
        self.ctrl_lane_num = 8 if self.ctrl_all_lane else 2  # 每个时刻控制的入口车道数。每一时刻都控制所有方向的车道

        self.use_CAV = config['use_CAV']
        self.cav_head = config['cav_head']  # 是否考虑混合流，即在选车时只选CAV还是所有
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None

        self.network = TD3Single(config, 'cav')
        self.save = lambda path, ep: self.network.save(path + 'cav_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else '99'
            self.network.load('../model/' + config['load_model_name'] + '/cav_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.T = config['cav']['T']

        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}

        self.trans_buffer = {}
        self.reward_list = []

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env):
        if self.use_CAV:
            global_income_cav = []
            for light in self.light_id:
                curr_cav = env.get_head_cav_id(light, cav_head=self.cav_head, curr_phase=not self.ctrl_all_lane)
                global_income_cav.extend(curr_cav)
                self.ctrl_cav[light].append(curr_cav)
            self.global_income_cav.append(global_income_cav)
        real, next_a = [], []
        for light_idx, light in enumerate(self.light_id):
            r, n = self._step(env, light)
            real.append(r)
            next_a.append(n)
        return real, next_a

    def _step(self, env, light):
        next_acc, real_a = None, None

        if self.use_CAV:
            # 对比两时刻头CAV，上时刻还有现在没了(可能切相位或驶出)的要reset一下跟驰
            for cav_id in self.ctrl_cav[light][-2]:
                if cav_id is not None and cav_id not in self.global_income_cav[-1]:
                    env.reset_head_cav(cav_id)
                    self.reward_list.append(self.trans_buffer[cav_id]['reward'])

                    del self.trans_buffer[cav_id]

            for cav_id in self.ctrl_cav[light][-1]:
                if cav_id:  # cav is not None
                    o_v = env.get_head_cav_obs(cav_id)  # list

                    if cav_id not in self.trans_buffer:  # == 0
                        self.trans_buffer[cav_id] = {'obs': [o_v],  # 存储车辆每一步的obs
                                                     'action': [],  # 每一步的action
                                                     'real_acc': [],  # 每一步的action
                                                     'reward': []}
                    else:  # >=1
                        self.trans_buffer[cav_id]['obs'].append(o_v)

                    cav_obs = self.trans_buffer[cav_id]['obs']
                    if len(cav_obs) >= self.T + 1:  # 没存满就先不控制
                        if self.train_model:  # 加噪声
                            if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                                a_v = np.random.random(self.network.a_dim) * 2 - 1
                            else:
                                a_v = self.network.choose_action(cav_obs[-self.T:])
                            a_v = np.clip(np.random.normal(0, self.var, size=a_v.shape) + a_v, -1, 1)
                        else:
                            a_v = self.network.choose_action(cav_obs[-self.T:])
                        self.trans_buffer[cav_id]['action'].append(a_v)
                        next_acc = a_v[0]    # [-1,1]
                        real_a = cav_obs[-1][2]    # [-?,1]
                        self.trans_buffer[cav_id]['real_acc'].append(real_a)   # 获取的是上一时步的实际acc

                        reward = env.get_car_agent_reward(cav_obs[-2], cav_obs[-1], real_a)
                        # reward = env.get_cav_reward(cav_obs[-1], self.trans_buffer[cav_id]['real_acc'][-2],
                        #                             self.trans_buffer[cav_id]['action'][-2]) if len(cav_obs) >= 1 + self.T else 0
                        self.trans_buffer[cav_id]['reward'].append(reward)

                        if self.train_model and len(cav_obs) >= self.T + 1:
                            self.network.store_transition(np.array(cav_obs[-self.T - 1: -1]).flatten(),
                                                          # self.trans_buffer[cav_id]['action'][-2],
                                                          self.trans_buffer[cav_id]['real_acc'][-1],    # 当前时刻的real_acc存的是上一时刻动作的真实效果
                                                          self.trans_buffer[cav_id]['reward'][-1],
                                                          np.array(cav_obs[-self.T:]).flatten())

                        env.set_head_cav_action(cav_id, cav_obs[-1][1], next_acc)
                        # print('cav obs:', cav_obs[-1], 'a:', next_acc, 'r:', self.trans_buffer[cav_id]['reward'][-1])
            if self.train_model and self.pointer >= self.learn_begin:
                self.var = max(0.01, self.var * 0.999)  # 0.9-40 0.99-400 0.999-4000
                self.network.learn()
        return (real_a, next_acc) if not real_a or not next_acc else (real_a * env.max_acc, next_acc * env.max_acc)

    def reset(self):
        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}

        self.trans_buffer = {}
        self.reward_list = []


class CoTVCavAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.ctrl_all_lane = True   # CoTV原文也是控制所有车道而不只是当前相位
        self.ctrl_lane_num = 8 if self.ctrl_all_lane else 2  # 每个时刻控制的入口车道数。每一时刻都控制所有方向的车道

        self.use_CAV = config['use_CAV']
        self.cav_head = config['cav_head']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None

        self.network = TD3Single(config, 'cav')
        self.save = lambda path, ep: self.network.save(path + 'cav_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else '99'
            self.network.load('../model/' + config['load_model_name'] + '/cav_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.T = config['cav']['T']

        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}

        self.trans_buffer = {}
        self.reward_list = []

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env):
        if self.use_CAV:
            global_income_cav = []
            for light in self.light_id:
                curr_cav = env.get_head_cav_id(light, cav_head=self.cav_head, curr_phase=not self.ctrl_all_lane)
                global_income_cav.extend(curr_cav)
                self.ctrl_cav[light].append(curr_cav)
            self.global_income_cav.append(global_income_cav)
        real, next_a = [], []
        for light_idx, light in enumerate(self.light_id):
            r, n = self._step(env, light)
            real.append(r)
            next_a.append(n)
        return real, next_a

    def _step(self, env, light):
        next_acc, real_a = None, None

        if self.use_CAV:
            # 对比两时刻头CAV，上时刻还有现在没了(可能切相位或驶出)的要reset一下跟驰
            for cav_id in self.ctrl_cav[light][-2]:
                if cav_id is not None and cav_id not in self.global_income_cav[-1]:
                    env.reset_head_cav(cav_id)
                    self.reward_list.append(self.trans_buffer[cav_id]['reward'])

                    del self.trans_buffer[cav_id]

            for cav_id in self.ctrl_cav[light][-1]:
                if cav_id:  # cav is not None
                    o_v = env.get_head_cav_obs(cav_id)  # list

                    if cav_id not in self.trans_buffer:  # == 0
                        self.trans_buffer[cav_id] = {'obs': [o_v],  # 存储车辆每一步的obs
                                                     'action': [],  # 每一步的action
                                                     'real_acc': [],  # 每一步的action
                                                     'reward': []}
                    else:  # >=1
                        self.trans_buffer[cav_id]['obs'].append(o_v)

                    cav_obs = self.trans_buffer[cav_id]['obs']
                    if len(cav_obs) >= self.T + 1:  # 没存满就先不控制
                        if self.train_model:  # 加噪声
                            if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                                a_v = np.random.random(self.network.a_dim) * 2 - 1
                            else:
                                a_v = self.network.choose_action(cav_obs[-self.T:])
                            a_v = np.clip(np.random.normal(0, self.var, size=a_v.shape) + a_v, -1, 1)
                        else:
                            a_v = self.network.choose_action(cav_obs[-self.T:])
                        self.trans_buffer[cav_id]['action'].append(a_v)
                        next_acc = a_v[0]    # [-1,1]
                        real_a = cav_obs[-1][2]    # [-?,1]
                        self.trans_buffer[cav_id]['real_acc'].append(real_a)   # 获取的是上一时步的实际acc

                        reward = env.get_cotv_car_reward(cav_id)
                        # reward = env.get_car_agent_reward(cav_obs[-2], cav_obs[-1], real_a)
                        # reward = env.get_cav_reward(cav_obs[-1], self.trans_buffer[cav_id]['real_acc'][-2],
                        #                             self.trans_buffer[cav_id]['action'][-2]) if len(cav_obs) >= 1 + self.T else 0
                        self.trans_buffer[cav_id]['reward'].append(reward)

                        if self.train_model and len(cav_obs) >= self.T + 1:
                            self.network.store_transition(np.array(cav_obs[-self.T - 1: -1]).flatten(),
                                                          # self.trans_buffer[cav_id]['action'][-2],
                                                          self.trans_buffer[cav_id]['real_acc'][-1],    # 当前时刻的real_acc存的是上一时刻动作的真实效果
                                                          self.trans_buffer[cav_id]['reward'][-1],
                                                          np.array(cav_obs[-self.T:]).flatten())

                        env.set_head_cav_action(cav_id, cav_obs[-1][1], next_acc)
                        # print('cav obs:', cav_obs[-1], 'a:', next_acc, 'r:', self.trans_buffer[cav_id]['reward'][-1])
            if self.train_model and self.pointer >= self.learn_begin:
                self.var = max(0.01, self.var * 0.999)  # 0.9-40 0.99-400 0.999-4000
                self.network.learn()

        return (real_a, next_acc) if not real_a or not next_acc else (real_a * env.max_acc, next_acc * env.max_acc)

    def reset(self):
        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}

        self.trans_buffer = {}
        self.reward_list = []
