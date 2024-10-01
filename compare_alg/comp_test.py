# -*- coding:utf-8 -*-
import time
import platform

from torch.utils.tensorboard import SummaryWriter

import comp_etc as utils
from comp_env import Environment
from comp_agt import hhCavAgent, hhLightAgent, CoTVLightAgent, CoTVCavAgent
from comp_cfg import env_configs, get_agent_configs


def launch_test(exp_cfg, test_rou_path, single_flag=True, cotv_flag=True, gui_on=True):
    if platform.system() != 'Windows':
        gui_on = False
    light_configs, cav_configs = get_agent_configs(exp_cfg['modify_dict'], cotv_flag)

    light_class = CoTVLightAgent if cotv_flag else hhLightAgent
    cav_class = CoTVCavAgent if cotv_flag else hhCavAgent

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

    light_agent = [light_class(light_idl, light_configs) for light_idl in holon_light_list]
    cav_agent = [cav_class(light_idl, cav_configs) for light_idl in holon_light_list]

    utils.txt_save('../log/' + str(experiment_name) + '/configs',
                   {'env': env_configs, 'light': light_configs, 'cav': cav_configs})
    utils.txt_save('../log/' + str(experiment_name) + '/exp_cfg', exp_cfg)

    evaluate_index = ['wait', 'halt', 'emission', 'fuel', 'speed', 'timeloss', 'collision']
    ep_performance = {_: [] for _ in evaluate_index}
    for episode in range(30):
        rou_file_num = episode + 1
        print("Ep:", episode, "File:", env.rou_path, rou_file_num, '\t', time.strftime("%Y-%m-%d %H:%M:%S"))
        env.start_env(gui_on, n_file=rou_file_num)

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
        utils.txt_save('../log/' + experiment_name + '/performance_index', ep_performance)

        # 重置智能体内暂存的列表
        for hid in range(len(holon_light_list)):
            light_agent[hid].reset()
            cav_agent[hid].reset()
        env.end_env()


def run_multiple_test(experience_cfg, test_rou):
    single = 'single' in test_rou
    for key in experience_cfg:
        experience_cfg[key]['experiment_name'] = 'test/' + key
        print(experience_cfg[key]['experiment_name'], 'start testing')
        launch_test(experience_cfg[key], test_rou, single_flag=single, gui_on=True)


if __name__ == '__main__':
    test_rou = 'single/test_603/'

    # experience_cfg = {
    #     'hh_ctrl_hdv_macro_500ep': {
    #         'modify_dict': {'light': {'cav_head': False, 'use_micro': False}, 'cav': {'use_CAV': False}}},
    #     'hh_ctrl_hdv_micro_500ep': {
    #         'modify_dict': {'light': {'cav_head': False, 'load_macro_name': to_be_tested + '/hh_ctrl_hdv_macro_500ep'},
    #                         'cav': {'use_CAV': False}}},
    # }
    experience_cfg = {
        'cotv_ctrl_hdv_g20': {'modify_dict': {
            'light': {'cav_head': False, 'green': 20, 'load_model_name': to_be_tested + '/cotv_ctrl_hdv_g20'},
            'cav': {'cav_head': False, 'load_model_name': to_be_tested + '/cotv_ctrl_hdv_g20'}}},
        'cotv_ctrl_hdv_g40': {'modify_dict': {
            'light': {'cav_head': False, 'green': 40, 'load_model_name': to_be_tested + '/cotv_ctrl_hdv_g40'},
            'cav': {'cav_head': False, 'load_model_name': to_be_tested + '/cotv_ctrl_hdv_g40'}}},
    }

    experience_cfg = {
        # 'tpgv_naive4_l2w_c72w_cT2_GRU_None': setting_test('tpgv', {
        #     'light': {'vehicle': {'act_dim': 4}, 'memory_capacity': 20000},
        #     'cav': {'high_goal_dim': 4, 'cav': {'T': 2}, 'memory_capacity': 720000}
        # }, to_be_tested='0828_single_603/tpgv_naive4_l2w_c72w_cT2_GRU_None'),
        'tp_naive4_l2w_c72w_cT2_GRU_None': setting_test('tp', {
            'light': {'vehicle': {'act_dim': 4}, 'memory_capacity': 20000},
            'cav': {'high_goal_dim': 4, 'cav': {'T': 2}, 'memory_capacity': 720000}
        }, to_be_tested='0828_single_603/tp_naive4_l2w_c72w_cT2_GRU_None', load_ep=49),
        'T_naive4_l2w_c72w_cT2_GRU_None': setting_test('T', {
            'light': {'vehicle': {'act_dim': 4}, 'memory_capacity': 20000},
            'cav': {'high_goal_dim': 4, 'cav': {'T': 2}, 'memory_capacity': 720000}
        }, to_be_tested='0828_single_603/T_naive4_l2w_c72w_cT2_GRU_None', load_ep=49),

    }
    run_multiple_test(experience_cfg, test_rou)