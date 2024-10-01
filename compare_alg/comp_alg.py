
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

torch.cuda.manual_seed(3407)
torch.manual_seed(3407)
np.random.seed(3407)  # 设置随机种子
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, obs_dim, out_dim, t, rnn_num_layer=1, use_bilstm=True):
        super(LSTM, self).__init__()
        self.T = t

        rnn_dim = out_dim   # 乱设的，可能有问题？
        self.obs_dim = obs_dim
        # self.lstm_layer = nn.LSTM(input_size=obs_dim, hidden_size=rnn_dim, num_layers=rnn_num_layer, batch_first=True,
        #                           bidirectional=use_bilstm)
        self.lstm_layer = nn.GRU(input_size=obs_dim, hidden_size=rnn_dim, num_layers=rnn_num_layer, batch_first=True,
                                  bidirectional=use_bilstm)
        self.fc = nn.Linear(rnn_dim * 2 if use_bilstm else rnn_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self.T, self.obs_dim)
        r_out, _ = self.lstm_layer(x)
        x = self.fc(r_out[:, -1, :])
        return x


class Actor(nn.Module):  # 定义 actor 网络结构
    def __init__(self, obs_dim, state_dim, action_dim, t, hidden_dim, max_action=1):
        super(Actor, self).__init__()
        self.T = t

        if t > 1:
            self.lstm = LSTM(obs_dim, state_dim, t)
        else:
            state_dim = obs_dim

        self.l1 = nn.Linear(state_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], action_dim)
        self.max_action = max_action

    def forward(self, s):
        if self.T > 1:
            s = self.lstm(s)
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


"""
    TD3 算法
"""


class CriticSingle(nn.Module):  # 定义 critic 网络结构
    def __init__(self, o_dim, s_dim, a_dim, t, hidden_dim):
        super(CriticSingle, self).__init__()
        self.t = t

        if t > 1:
            self.lstm = LSTM(o_dim, s_dim, t)
        else:
            s_dim = o_dim

        # Q1 architecture   计算 Q1
        self.l1 = nn.Linear(s_dim + a_dim, hidden_dim[0])
        self.l2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l3 = nn.Linear(hidden_dim[1], 1)

        # Q2 architecture   计算 Q2
        self.l4 = nn.Linear(s_dim + a_dim, hidden_dim[0])
        self.l5 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.l6 = nn.Linear(hidden_dim[1], 1)

    def forward(self, state, action):  # 注意此处，直接把两个网络写在一起，这样就可以只用一个梯度下降优化器
        if self.t > 1:
            state = self.lstm(state)

        sa = torch.cat([state, action], 1)  # 将s和a横着拼接在一起

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)    # 直接输出线性计算后的值作为Q值

        x2 = F.relu(self.l4(sa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, state, action):  # 新增一个Q值输出的方法，只使用其中一个网络的结果作为输出，避免重复计算
        if self.t > 1:
            state = self.lstm(state)

        sa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        q = self.l3(x1)
        return q


class TD3Single:
    def __init__(self, cfg, task):
        self.o_dim = cfg[task]['obs_dim']    # 单步观测维度
        self.a_dim = cfg[task]['act_dim']    # 动作维度
        self.s_dim = cfg[task]['state_dim']  # LSTM输出维度
        self.t = cfg[task]['T']

        self.obs_dim = self.o_dim * self.t
        self.a1_dim = self.a2_dim = self.a3_dim = self.a_dim      # 与HATD3一致，方便外部调用
        self.choose_time_action = self.choose_phase_action = self.choose_goal = self.choose_action      # 方便外部调用

        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  # 软更新系数
        self.batch_size = cfg['batch_size']

        if task in ('macro', 'micro', 'cotv'):
            self.memory_capacity = cfg[task]['memory_capacity']  # 记忆库大小
            self.learn_begin = self.memory_capacity * cfg[task]['learn_start_ratio']  # 存满一定比例的记忆库之后开始学习并用网络输出动作
        else:
            self.memory_capacity = cfg['memory_capacity']  # 记忆库大小
            self.learn_begin = self.memory_capacity * cfg['learn_start_ratio']   # 存满一定比例的记忆库之后开始学习并用网络输出动作
        self.memory = np.zeros((self.memory_capacity, self.obs_dim * 2 + self.a_dim + 1))
        self.pointer = 0

        # 创建对应的四个网络
        self.actor = Actor(self.o_dim, self.s_dim, self.a_dim, self.t, cfg[task]['hidden_dim']).to(device)
        self.actor_target = Actor(self.o_dim, self.s_dim, self.a_dim, self.t, cfg[task]['hidden_dim']).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # 存储网络名字和对应参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_learning_rate'])
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                               step_size=cfg['actor_scheduler_step'], gamma=0.5)

        self.critic = CriticSingle(self.o_dim, self.s_dim, self.a_dim, self.t, cfg[task]['hidden_dim']).to(device)
        self.critic_target = CriticSingle(self.o_dim, self.s_dim, self.a_dim, self.t, cfg[task]['hidden_dim']).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_learning_rate'])
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=cfg['critic_scheduler_step'], gamma=0.5)

        self.policy_freq = 2
        self.total_it = 0

    def store_transition(self, o, a, r, o_):
        transition = np.hstack((o, a, r, o_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, o):
        obs = torch.FloatTensor(o).view(1, -1, self.o_dim).to(device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def learn(self):
        self.total_it += 1
        # mini batch sample
        indices = np.random.choice(min(self.pointer, self.memory_capacity), size=self.batch_size)   # 注意，这里是默认有放回
        batch_trans = self.memory[indices, :]

        obs = torch.FloatTensor(batch_trans[:, :self.obs_dim]).to(device)
        action = torch.FloatTensor(batch_trans[:, self.obs_dim: self.obs_dim + self.a_dim]).to(device)
        reward = torch.FloatTensor(batch_trans[:, -self.obs_dim - 1: -self.obs_dim]).to(device)
        next_obs = torch.FloatTensor(batch_trans[:, -self.obs_dim:]).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)       # noise=0.2, clip=0.5
            next_action = (self.actor_target(next_obs) + noise).clamp(-1, 1)    # 默认动作空间[-1,1]

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.scheduler_actor.step()

    def save(self, filename):
        # 保存模型
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        # 加载模型
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
