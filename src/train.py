# -*- coding:utf-8 -*-
import time
import socket
import platform

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils
from agent import IndependentLightAgent, ManagerLightAgent, IndependentCavAgent, WorkerCavAgent, LoyalCavAgent, FullIndependentLightAgent
from configs import env_configs, get_agent_configs
from environment import Environment

np.random.seed(3407)  # 设置随机种子


def setting_train(base_key, change):
    experience_cfg_base = {
        'baseline': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False, },
                            'cav': {'use_CAV': False, }}},
        'T': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': False, }}},
        'P': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': False, }}},
        'V': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'TV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'PV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'tp': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True, },
                            'cav': {'use_CAV': False, }}},
        'tpV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'Gv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'tgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'pgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'tpgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True,
                                      'use_goal': True, },
                            'cav': {'use_CAV': True, }}},
    }
    return utils.change_dict(experience_cfg_base[base_key], {'modify_dict': change})


def launch_train(
        exp_cfg,
        save_model=True,
        single_flag=True,
        max_episodes=100,
        gui_on_after_learn=True
):
    sumo_gui = False    # 默认最开始不打开gui，在开始学习后要不要打开根据gui_on_after_learn来决定
    exp_cfg['turn_on_gui_after_learn_start'] = gui_on_after_learn if platform.system() == 'Windows' else False
    light_configs, cav_configs = get_agent_configs(exp_cfg['modify_dict'])

    experiment_name = exp_cfg['experiment_name']
    utils.mkdir('../log/' + experiment_name)
    tf_log_path = '../log/' if platform.system() == 'Windows' else '../../tf-logs/'
    writer = SummaryWriter(tf_log_path + experiment_name)
    model_dir = '../model/' + experiment_name + '/'
    env = Environment(env_configs, single_flag)
    light_id_list = env.get_light_id()
    holon_light_list = [light_id_list]  # four&single都是用一个agent控所有，因此可以这么写。108需要修改这里。独立路口[[n_0], [n_1]...]

    if exp_cfg['use_HRL']:
        if 'loyal' in exp_cfg['experiment_name']:   # 方便起见，以检索实验名中有无loyal字段来判断cav是否使用loyal
            cav_agent = [LoyalCavAgent(light_idl, cav_configs) for light_idl in holon_light_list]
        elif 'TP' in exp_cfg['experiment_name'] and not exp_cfg['modify_dict']['light']['use_goal']:   # 如果TP而且没用G
            cav_agent = [IndependentCavAgent(light_idl, cav_configs) for light_idl in holon_light_list]
        else:
            cav_agent = [WorkerCavAgent(light_idl, cav_configs) for light_idl in holon_light_list]

        if 'TP' in exp_cfg['experiment_name']:   # 方便起见，以检索实验名中有无TP字段来判断light是否使用HAML
            light_agent = [FullIndependentLightAgent(light_idl, light_configs) for light_idl in holon_light_list]
        else:
            light_agent = [ManagerLightAgent(light_idl, light_configs, cav_agent[hid].get_oa, cav_agent[hid].network.policy)
                           for hid, light_idl in enumerate(holon_light_list)]
    else:
        light_agent = [IndependentLightAgent(light_idl, light_configs) for light_idl in holon_light_list]
        cav_agent = [IndependentCavAgent(light_idl, cav_configs) for light_idl in holon_light_list]

    utils.txt_save(tf_log_path + str(experiment_name) + '/configs',
                   {'env': env_configs, 'light': light_configs, 'cav': cav_configs})
    utils.txt_save(tf_log_path + str(experiment_name) + '/exp_cfg', exp_cfg)

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
                if light_agent[hid].__class__.__name__ in ['ManagerLightAgent', 'FullIndependentLightAgent']:
                    l_t, l_p, goal = light_agent[hid].step(env)
                else:   # 'IndependentLightAgent'
                    l_t, l_p = light_agent[hid].step(env)
                    goal = [None] * len(holon_light_list[hid])   # dim=路口数
                real_a, v_a = cav_agent[hid].step(env, goal, l_p)

                # tensorboard只显示每个区第一个路口的动作
                if l_t[0] is not None:
                    writer.add_scalar('green time/' + str(episode), l_t[0], t)
                if l_p[0] is not None:
                    writer.add_scalar('next phase/' + str(episode), l_p[0], t)
                # if goal[0] is not None:
                #     writer.add_scalar('advice speed lane0/' + str(episode), goal[0][0] * env.max_speed, t)
                #     # print(goal * env.max_speed)
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

            pg = goal[0].tolist() if goal[0] is not None else [""]
            print('\r', t, '\t', light_agent[0].pointer, cav_agent[0].pointer, '\tg:', pg[0], flush=True, end='')

        ep_light_r = sum(sum(light_agent[_].reward_list) for _ in range(len(holon_light_list)))
        ep_cav_r = sum(sum(sum(sublist) for sublist in cav_agent[_].reward_list) for _ in range(len(holon_light_list)))
        ep_wait = sum(waiting_time)
        ep_halt = sum(halting_num)
        ep_emission = sum(emission)
        ep_fuel = sum(fuel_consumption)
        ep_speed = sum(mean_speed) / len(mean_speed)
        ep_timeloss = sum(time_loss)

        writer.add_scalar('light reward', ep_light_r, episode)
        writer.add_scalar('cav reward', ep_cav_r, episode)
        writer.add_scalar('waiting time', ep_wait, episode)
        writer.add_scalar('halting count', ep_halt, episode)
        writer.add_scalar('carbon emission', ep_emission, episode)
        writer.add_scalar('fuel consumption', ep_fuel, episode)
        writer.add_scalar('average speed', ep_speed, episode)
        writer.add_scalar('time loss', ep_timeloss, episode)
        writer.add_scalar('collision', env.collision_count, episode)

        print('\n', episode,
              '\n\tlight:\tpointer=', light_agent[0].pointer, '\tvar=', light_agent[0].var, '\treward=', ep_light_r,
              '\n\tcav:\tpointer=', cav_agent[0].pointer, '\tvar=', cav_agent[0].var, '\treward=', ep_cav_r,
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
        launch_train(experience_cfg[key], save_model=True, single_flag=single, max_episodes=max_ep)


if __name__ == '__main__':
    # series_name = 'single_0901'  # 注意，下面用文件夹名是否包含single判断是用单路口还是多路口
    series_name = 'four_0901'  # 注意，下面用文件夹名是否包含single判断是用单路口还是多路口
    max_episodes = 100  # 训练轮数

    experience_cfg = {
        # 'T': setting_train('T', {}),    #
        # 'P': setting_train('P', {}),    #
        # 'tp': setting_train('tp', {}),  #
        # 'V': setting_train('V', {}),    #
        # 'TV': setting_train('TV', {}),
        # 'PV': setting_train('PV', {}),
        # 'tpV': setting_train('tpV', {}),
        # 'Gv': setting_train('Gv', {}),  #
        # 'tgv': setting_train('tgv', {}),
        # 'pgv': setting_train('pgv', {}),
        # 'tpgv': setting_train('tpgv', {}),
        # # 'tpgv_useAdj': setting_train('tpgv', {'light': {'use_adj': True}}),   # four only
        # 'tpgv_alpha05': setting_train('tpgv', {'cav': {'alpha': 0.5}}),
        # 'tpgv_alpha02': setting_train('tpgv', {'cav': {'alpha': 0.2}}),
        # 'TPGv_alpha05': setting_train('tpgv', {'cav': {'alpha': 0.5}}),
        'TP': setting_train('tpgv', {'light': {'use_goal': False}}),

        # 'tpgv_naive4_l2w_c10w_cT2_GRU_timeFactor': setting_train('tpgv', {
        #     'light': {'vehicle': {'act_dim': 4}, 'memory_capacity': 20000},
        #     'cav': {'high_goal_dim': 4, 'cav': {'T': 2}, 'memory_capacity': 100000}}),
        # 'tpgv_naive1_l2w_c10w_cT2_GRU_timeFactor': setting_train('tpgv', {
        #     'light': {'vehicle': {'act_dim': 1}, 'memory_capacity': 20000},
        #     'cav': {'high_goal_dim': 1, 'cav': {'T': 2}, 'memory_capacity': 100000}}),
        # 'tpgv_naive4_l2w_c10w_cT2_GRU_timeFactor_alpha05': setting_train('tpgv', {
        #     'light': {'vehicle': {'act_dim': 4}, 'memory_capacity': 20000},
        #     'cav': {'high_goal_dim': 4, 'cav': {'T': 2}, 'memory_capacity': 100000, 'alpha': 0.5}}),
        # 'tpgv_naive4_l2w_c10w_cT2_GRU_timeFactor_alpha02_load': setting_train('tpgv', {
        #     'light': {'vehicle': {'act_dim': 4}, 'memory_capacity': 20000,
        #               'load_model_name': '0828_single_603/tpgv_naive4_l2w_c10w_cT2_GRU_timeFactor', 'load_model_ep': 49},
        #     'cav': {'high_goal_dim': 4, 'cav': {'T': 2}, 'memory_capacity': 100000, 'alpha': 0.2,
        #             'load_model_name': '0828_single_603/tpgv_naive4_l2w_c10w_cT2_GRU_timeFactor', 'load_model_ep': 49}}),
    }

    run_multiple_train(experience_cfg, series_name, max_episodes)