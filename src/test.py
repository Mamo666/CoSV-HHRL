# -*- coding:utf-8 -*-
import time
import platform

from torch.utils.tensorboard import SummaryWriter

import utils
from agent import IndependentLightAgent, ManagerLightAgent, IndependentCavAgent, WorkerCavAgent, FullIndependentLightAgent
from configs import env_configs, get_agent_configs
from environment import Environment


def setting_test(base_key, change, to_be_tested, load_ep):
    """to_be_tested: load_model_name"""
    experience_cfg_base = {
        'baseline': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': None, },
                            'cav': {'use_CAV': False,
                                    'train_model': False,
                                    'load_model_name': None, }}},
        'T': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': to_be_tested, 
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': False,
                                    'train_model': False,
                                    'load_model_name': None, }}},
        'P': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested, 
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': False,
                                    'train_model': False,
                                    'load_model_name': None, }}},
        'V': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': None, },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested, 
                                    'load_model_ep': load_ep}}},
        'TV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': to_be_tested, 
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested,
                                    'load_model_ep': load_ep}}},
        'PV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested,
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested,
                                    'load_model_ep': load_ep}}},
        'tp': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested, 
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': False,
                                    'train_model': False,
                                    'load_model_name': None, }}},
        'tpV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested, 
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested, 
                                    'load_model_ep': load_ep}}},
        'Gv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': to_be_tested, 
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested, 
                                    'load_model_ep': load_ep}}},
        'tgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': to_be_tested, 
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested, 
                                    'load_model_ep': load_ep}}},
        'pgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested, 
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested, 
                                    'load_model_ep': load_ep}}},
        'tpgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True,
                                      'use_goal': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested,
                                      'load_model_ep': load_ep},
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested,
                                    'load_model_ep': load_ep}}},
    }
    return utils.change_dict(experience_cfg_base[base_key], {'modify_dict': change})


def launch_test(exp_cfg, test_rou_path, single_flag=True, gui_on=True):
    if platform.system() != 'Windows':
        gui_on = False
    light_configs, cav_configs = get_agent_configs(exp_cfg['modify_dict'])

    experiment_name = exp_cfg['experiment_name']
    utils.mkdir('../log/' + experiment_name)
    tf_log_path = '../log/' if platform.system() == 'Windows' else '../../tf-log/'
    writer = SummaryWriter(tf_log_path + experiment_name)

    env_cfg = env_configs['single'] if single_flag else env_configs['four']
    env_cfg['sumocfg_path'] = '../sumo_sim_env/collision_env_test.sumocfg'  # 防止两边同时运行修改时撞车
    env_cfg['rou_path'] = test_rou_path

    env = Environment(env_configs, single_flag)
    light_id_list = env.get_light_id()
    holon_light_list = [light_id_list]  # four&single都是用一个agent控所有，因此可以这么写。108需要修改这里。独立路口[[n_0], [n_1]...]

    if exp_cfg['use_HRL']:
        if 'TP' in exp_cfg['experiment_name'] and not exp_cfg['modify_dict']['light']['use_goal']:   # 如果TP而且没用G
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

    evaluate_index = ['wait', 'halt', 'emission', 'fuel', 'speed', 'timeloss', 'collision']
    ep_performance = {_: [] for _ in evaluate_index}
    for episode in range(30):
        rou_file_num = episode + 1
        print("Ep:", episode, "File:", env.rou_path, rou_file_num, '\t', time.strftime("%Y-%m-%d %H:%M:%S"))
        env.start_env(gui_on, n_file=rou_file_num)

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

            print('\r', t, flush=True, end='')

        ep_wait = sum(waiting_time)
        ep_halt = sum(halting_num)
        ep_fuel = sum(fuel_consumption)
        ep_emission = sum(emission)
        ep_timeloss = sum(time_loss)
        ep_speed = sum(mean_speed) / len(mean_speed)
        ep_collision = env.collision_count

        print('\n', episode,
              '\n\twait=', ep_wait, '\thalt=', ep_halt,
              '\tspeed=', ep_speed, '\tcollision=', ep_collision,
              '\temission=', ep_emission, '\tfuel_consumption=', ep_fuel, '\ttime_loss=', ep_timeloss)

        writer_scalar_name = ['waiting time', 'halting count', 'carbon emission', 'fuel consumption',
                              'average speed', 'time loss', 'collision']
        ep_list = [ep_wait, ep_halt, ep_emission, ep_fuel, ep_speed, ep_timeloss, ep_collision]
        for i in range(len(evaluate_index)):
            ep_performance[evaluate_index[i]].append(ep_list[i])
            writer.add_scalar(writer_scalar_name[i], ep_list[i], episode)
        utils.txt_save('../log/' + experiment_name + '/performance_index', ep_performance)  # 方便起见，结果就存到同一个地方

        # 重置智能体内暂存的列表
        for hid in range(len(holon_light_list)):
            light_agent[hid].reset()
            cav_agent[hid].reset()
        env.end_env()


def run_multiple_test(experience_cfg, test_rou):
    single = 'single' in test_rou
    rou_name = 'four' if not single else 'single'
    for key in experience_cfg:
        experience_cfg[key]['experiment_name'] = 'test/' + rou_name + '/' + key
        print(experience_cfg[key]['experiment_name'], 'start testing')
        launch_test(experience_cfg[key], test_rou, single_flag=single, gui_on=True)


if __name__ == '__main__':
    # test_rou = 'single/test_603/'
    # model_dir = 'single_0901/'
    test_rou = 'four/test_603/'
    model_dir = 'four_0901/'

    experience_cfg = {
        # 'fixed': setting_test('baseline', {}, to_be_tested=None, load_ep=None),
        # 'T': setting_test('T', {}, to_be_tested=model_dir+'T', load_ep=99),  #
        # 'P': setting_test('P', {}, to_be_tested=model_dir+'P', load_ep=99),  #
        # 'tp': setting_test('tp', {}, to_be_tested=model_dir+'tp', load_ep=99),  #
        # 'V': setting_test('V', {}, to_be_tested=model_dir+'V', load_ep=99),  #
        # # 'TV': setting_test('TV', {}, to_be_tested=model_dir+'TV', load_ep=99),
        # # 'PV': setting_test('PV', {}, to_be_tested=model_dir+'PV', load_ep=99),
        # # 'tpV': setting_test('tpV', {}, to_be_tested=model_dir+'tpV', load_ep=99),
        # 'Gv': setting_test('Gv', {}, to_be_tested=model_dir+'Gv', load_ep=99),  #
        # # 'tgv': setting_test('tgv', {}, to_be_tested=model_dir+'tgv', load_ep=99),
        # # 'pgv': setting_test('pgv', {}, to_be_tested=model_dir+'pgv', load_ep=99),
        # 'tpgv': setting_test('tpgv', {}, to_be_tested=model_dir+'tpgv', load_ep=99),
        # 'tpgv_alpha05': setting_test('tpgv', {'cav': {'alpha': 0.5}}, to_be_tested=model_dir+'tpgv_alpha05', load_ep=99),
        # 'tpgv_alpha02': setting_test('tpgv', {'cav': {'alpha': 0.2}}, to_be_tested=model_dir+'tpgv_alpha02', load_ep=99),

        # 'TPGv_alpha05': setting_test('tpgv', {'cav': {'alpha': 0.5}}, to_be_tested=model_dir+'TPGv_alpha05', load_ep=99),
        'TP': setting_test('tpgv', {'light': {'use_goal': False}}, to_be_tested=model_dir + 'TP_alpha05', load_ep=99),# 还没跑four/TP
    }

    run_multiple_test(experience_cfg, test_rou)