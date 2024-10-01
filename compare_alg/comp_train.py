
# -*- coding:utf-8 -*-
import time
import socket
import platform
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import comp_etc as utils
from comp_env import Environment
from comp_agt import hhCavAgent, hhLightAgent, CoTVLightAgent, CoTVCavAgent
from comp_cfg import env_configs, get_agent_configs

np.random.seed(3407)  # 设置随机种子


def launch_experiment(exp_cfg, save_model=True, single_flag=True, cotv_flag=True, max_episodes=100, gui_on_after_learn=True):
    sumo_gui = False    # 默认最开始不打开gui，在开始学习后要不要打开根据gui_on_after_learn来决定
    exp_cfg['turn_on_gui_after_learn_start'] = gui_on_after_learn if platform.system() == 'Windows' else False
    light_configs, cav_configs = get_agent_configs(exp_cfg['modify_dict'], cotv_flag)

    light_class = CoTVLightAgent if cotv_flag else hhLightAgent
    cav_class = CoTVCavAgent if cotv_flag else hhCavAgent

    experiment_name = exp_cfg['experiment_name']
    utils.mkdir('../log/' + experiment_name)
    tf_log_path = '../log/' if platform.system() == 'Windows' else '../../tf-logs/'
    writer = SummaryWriter(tf_log_path + experiment_name)
    model_dir = '../model/' + experiment_name + '/'
    env = Environment(env_configs, single_flag)
    light_id_list = env.get_light_id()
    holon_light_list = [light_id_list]  # four&single都是用一个agent控所有，因此可以这么写。108需要修改这里。独立路口[[n_0], [n_1]...]

    light_agent = [light_class(light_idl, light_configs) for light_idl in holon_light_list]
    cav_agent = [cav_class(light_idl, cav_configs) for light_idl in holon_light_list]

    utils.txt_save('../log/' + str(experiment_name) + '/configs',
                   {'env': env_configs, 'light': light_configs, 'cav': cav_configs})
    utils.txt_save('../log/' + str(experiment_name) + '/exp_cfg', exp_cfg)

    light_start_learn, cav_start_learn = None, None
    for episode in range(max_episodes):
        rou_file_num = np.random.randint(1, 31)  # 随机选取一个训练环境
        print("Ep:", episode, "File:", env.rou_path, rou_file_num, '\t', time.strftime("%Y-%m-%d %H:%M:%S"))
        if light_agent[0].pointer > light_agent[0].learn_begin and not light_start_learn:
            light_start_learn = episode
        if cav_agent[0].pointer > cav_agent[0].learn_begin and not cav_start_learn:
            cav_start_learn = episode
        if light_start_learn and cav_start_learn:
            sumo_gui = exp_cfg['turn_on_gui_after_learn_start']
        env.start_env(sumo_gui, n_file=rou_file_num)

        waiting_time, halting_num, emission, fuel_consumption, mean_speed, time_loss = [], [], [], [], [], []

        for t in range(3000):
            for hid in range(len(holon_light_list)):
                # tensorboard只显示每个区第一个路口的动作
                if cotv_flag:
                    l_a = light_agent[hid].step(env)
                    if l_a[0] is not None:
                        writer.add_scalar('change phase/' + str(episode), l_a[0], t)
                else:
                    mac_a, mic_a = light_agent[hid].step(env)
                    if mac_a[0] is not None:
                        writer.add_scalar('green time/' + str(episode), sum(mac_a[0]), t)
                        writer.add_scalar('phase time/phase1/' + str(episode), mac_a[0][0], t)
                        writer.add_scalar('phase time/phase2/' + str(episode), mac_a[0][1], t)
                        writer.add_scalar('phase time/phase3/' + str(episode), mac_a[0][2], t)
                        writer.add_scalar('phase time/phase4/' + str(episode), mac_a[0][3], t)
                    if mic_a[0] is not None:
                        writer.add_scalar('micro_actual_time/' + str(episode), mic_a[0], t)

                real_a, v_a = cav_agent[hid].step(env)
                if v_a[0] is not None:
                    writer.add_scalar('head CAV action/' + str(episode), v_a[0], t)
                    writer.add_scalar('head CAV acc_real/' + str(episode), real_a[0], t)

            env.step_env()

            if t % 10 == 0:  # episode % 10 == 9 and
                w, h, e, f, v, timeLoss = env.get_performance()
                waiting_time.append(w)
                halting_num.append(h)
                emission.append(e)
                fuel_consumption.append(f)
                mean_speed.append(v)
                time_loss.append(timeLoss)

            print('\r', t, '\t', light_agent[0].pointer, cav_agent[0].pointer, flush=True, end='')

        print('\n', episode)
        if cotv_flag:
            ep_light_r = sum(sum(light_agent[_].r_list) for _ in range(len(holon_light_list)))
            writer.add_scalar('light reward', ep_light_r, episode)
            print('\tlight0:\tpointer=', light_agent[0].network.pointer, '\treward=', ep_light_r)
        else:
            ep_macro_r = sum(sum(light_agent[_].r_t_list) for _ in range(len(holon_light_list)))
            ep_micro_r = sum(sum(light_agent[_].r_p_list) for _ in range(len(holon_light_list)))
            writer.add_scalar('macro reward', ep_macro_r, episode)
            writer.add_scalar('micro reward', ep_micro_r, episode)
            print('\tmacro0:\tpointer=', light_agent[0].macro_net.pointer, '\treward=', ep_macro_r,
                  '\n\tmicro0:\tpointer=', light_agent[0].micro_net.pointer, '\treward=', ep_micro_r)

        ep_cav_r = sum(sum(sum(sublist) for sublist in cav_agent[_].reward_list) for _ in range(len(holon_light_list)))
        ep_wait = sum(waiting_time)
        ep_halt = sum(halting_num)
        ep_emission = sum(emission)
        ep_fuel = sum(fuel_consumption)
        ep_speed = sum(mean_speed) / len(mean_speed)
        ep_timeloss = sum(time_loss)
        writer.add_scalar('cav reward', ep_cav_r, episode)
        writer.add_scalar('waiting time', ep_wait, episode)
        writer.add_scalar('halting count', ep_halt, episode)
        writer.add_scalar('carbon emission', ep_emission, episode)
        writer.add_scalar('fuel consumption', ep_fuel, episode)
        writer.add_scalar('average speed', ep_speed, episode)
        writer.add_scalar('time loss', ep_timeloss, episode)
        writer.add_scalar('collision', env.collision_count, episode)

        print('\tcav:\tpointer=', cav_agent[0].pointer, '\treward=', ep_cav_r,
              '\n\twait=', ep_wait, '\thalt=', ep_halt,
              '\tspeed=', ep_speed, '\tcollision=', env.collision_count,
              '\temission=', ep_emission, '\tfuel_consumption=', ep_fuel, '\ttime_loss=', ep_timeloss,
              '\n* ', experiment_name, ' *, light start at EP:', light_start_learn, ', cav start at EP:', cav_start_learn)

        # 重置智能体内暂存的列表, 顺便实现每10轮存储一次模型参数
        for hid in range(len(holon_light_list)):
            light, cav = light_agent[hid], cav_agent[hid]
            if save_model:
                utils.mkdir(model_dir)
                if episode % 10 == 9:
                    light.save(model_dir, episode)
                    cav.save(model_dir, episode)
            light.reset()
            cav.reset()
        env.end_env()

    utils.txt_save('../log/' + str(experiment_name) + '/start_learn', {'light': light_start_learn, 'cav': cav_start_learn})


def run_multiple_train(experience_cfg, series_name, max_ep):
    with open('train_hist.txt', 'a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + '\ton ' + str(socket.gethostname()) + '\t\t' + series_name + '\n')
        f.write('max_ep: ' + str(max_ep) + '\tusing flow feat: all 30 files\n')
        f.write(str(experience_cfg) + '\n\n')

    single = 'single' in series_name
    for key in experience_cfg:
        series_name = series_name + '/' if series_name[-1] != '/' else series_name
        experience_cfg[key]['experiment_name'] = series_name + key
        print(experience_cfg[key]['experiment_name'], 'start running')
        is_cotv_flag = 'cotv' in experience_cfg[key]['experiment_name']
        launch_experiment(experience_cfg[key], save_model=True, single_flag=single, cotv_flag=is_cotv_flag, max_episodes=max_ep)


if __name__ == '__main__':
    series_name = 'single_0901/compare'
    # series_name = 'four_0901/compare'
    max_episodes = 100  # 训练轮数

    experience_cfg = {
        'hh_ctrlCAV_cavMem40w': {'modify_dict': {'light': {'cav_head': True}, 'cav': {'cav_head': True, 'memory_capacity': 400000}}},
        # 'hh_ctrlHDV_cavMem40w': {'modify_dict': {'light': {'cav_head': False}, 'cav': {'cav_head': False, 'memory_capacity': 400000}}},
        'hh_ctrlHDV_cavMem200w': {'modify_dict': {'light': {'cav_head': False}, 'cav': {'cav_head': False, 'memory_capacity': 2000000}}},
        'cotv_ctrlHDV_g20_cavMem20w': {'modify_dict': {'light': {'green': 20}, 'cav': {'memory_capacity': 200000}}},
        'cotv_ctrlHDV_g5_cavMem20w': {'modify_dict': {'light': {'green': 5}, 'cav': {'memory_capacity': 200000}}},
        'cotv_ctrlCAV_g20_cavMem20w': {'modify_dict': {'light': {'green': 20, 'cav_head': True}, 'cav': {'cav_head': True, 'memory_capacity': 200000}}},
        'cotv_ctrlCAV_g5_cavMem20w': {'modify_dict': {'light': {'green': 5, 'cav_head': True}, 'cav': {'cav_head': True, 'memory_capacity': 200000}}},
    }

    # experience_cfg = {  # 实验名和series_name任一有'cotv'则启用COTV实验，否则运行hh实验。'modify_dict'的两个子字典不可省略
    #     'cotv_ctrlHDV_g40_cavMem50w': {'modify_dict': {'light': {'green': 40}, 'cav': {'memory_capacity': 500000}}},
    #     'cotv_ctrlHDV_g20_cavMem50w': {'modify_dict': {'light': {'green': 20}, 'cav': {'memory_capacity': 500000}}},
    #     'cotv_ctrlHDV_g5_cavMem50w': {'modify_dict': {'light': {'green': 5}, 'cav': {'memory_capacity': 500000}}},
    # }

    run_multiple_train(experience_cfg, series_name, max_episodes)

