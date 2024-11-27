import torch
import os.path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import statistics
from torch.distributions import Categorical

class actor_LSTM(nn.Module):
    def __init__(self, args):
        super(actor_LSTM, self).__init__()
        self.n_action = args.n_action
        self.machineEmbedding = nn.Linear(args.machine_embedding_dim, args.embedding_dim)
        self.jobsEmbedding = nn.Linear(args.jobs_embedding_dim, args.embedding_dim)
        self.sequenceLSTM = nn.LSTM(2 * args.embedding_dim, args.embedding_dim, 2, batch_first=True)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.layers = nn.ModuleList([nn.ModuleDict({
            'a1': nn.Linear(args.embedding_dim, args.embedding_dim),
            'a2': nn.Linear(args.embedding_dim, args.embedding_dim),
            'a3': nn.Linear(args.embedding_dim, self.n_action)
        }) for i in range(1)])

    def instanceEmbedding(self, state):
        jobsInfor = torch.Tensor(state[0])
        # jobsInfor = (jobsInfor - jobsInfor.mean()) / jobsInfor.std(dim=1)
        machineInfor = torch.Tensor(state[1])
        # machineInfor = (machineInfor - machineInfor.mean()) / machineInfor.std(dim=1)
        jobsInfor = self.jobsEmbedding(jobsInfor)
        machineInfor = self.machineEmbedding(machineInfor)
        input = torch.cat((jobsInfor, machineInfor), dim=1)
        instanceEmbedding = self.sequenceLSTM(input.unsqueeze(0))[0]
        return instanceEmbedding

    def forward(self, state):
        stateEmebded = self.instanceEmbedding(state)
        for l in self.layers:
            output = l['a3'](self.activate_func(l['a2'](self.activate_func(l['a1'](stateEmebded)))))
        a_prob = F.softmax(output, dim=2)
        return a_prob

class critic_LSTM(nn.Module):
    def __init__(self, args):
        super(critic_LSTM, self).__init__()
        self.n_action = args.n_action
        self.machineEmbedding = nn.Linear(args.machine_embedding_dim, args.embedding_dim)
        self.jobsEmbedding = nn.Linear(args.jobs_embedding_dim, args.embedding_dim)
        self.sequenceLSTM = nn.LSTM(2 * args.embedding_dim, args.embedding_dim, batch_first=True)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        # self.output_layer = nn.Linear(args.embedding_dim, 1)
        self.layers = nn.ModuleList([nn.ModuleDict({
            'a1': nn.Linear(args.embedding_dim, args.embedding_dim),
            'a2': nn.Linear(args.embedding_dim, args.embedding_dim),
            'a3': nn.Linear(args.embedding_dim, 1)
        }) for i in range(1)])

    def instanceEmbedding(self, state):
        jobsInfor = torch.Tensor(state[0])
        machineInfor = torch.Tensor(state[1])
        jobsInfor = self.jobsEmbedding(jobsInfor)
        machineInfor = self.machineEmbedding(machineInfor)
        input = torch.cat((jobsInfor, machineInfor), dim=1)
        instanceEmbedding = self.sequenceLSTM(input.unsqueeze(0))[0]
        return instanceEmbedding

    def forward(self, state):
        stateEmebded = self.instanceEmbedding(state)
        # output = self.output_layer(stateEmebded)
        for l in self.layers:
            output = l['a3'](self.activate_func(l['a2'](self.activate_func(l['a1'](stateEmebded)))))
        a_prob = F.softmax(output, dim=1)
        return a_prob

# design 2:将上层agent选择的action作为下层agent的输入
class actor(nn.Module):
    def __init__(self, args):
        super(actor, self).__init__()
        self.n_action = args.n_action
        self.machineEmbedding = nn.Linear(args.machine_embedding_dim, args.embedding_dim)
        self.jobsEmbedding = nn.Linear(args.jobs_embedding_dim, args.embedding_dim)
        self.actionEmbedding = nn.Linear(args.action_embedding_dim, args.embedding_dim)
        self.sequenceLSTM = nn.LSTM(3 * args.embedding_dim, args.embedding_dim, batch_first=True)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.layers = nn.ModuleList([nn.ModuleDict({
            'a1': nn.Linear(args.embedding_dim, args.embedding_dim),
            'a2': nn.Linear(args.embedding_dim, args.embedding_dim),
            'a3': nn.Linear(args.embedding_dim, self.n_action)
        }) for i in range(1)])

    def instanceEmbedding(self, state):
        jobsInfor = torch.Tensor(state[0])
        machineInfor = torch.Tensor(state[1])
        selected_action = torch.Tensor(state[2])
        jobsInfor = self.jobsEmbedding(jobsInfor)
        machineInfor = self.machineEmbedding(machineInfor)
        selected_action = self.actionEmbedding(selected_action)
        input = torch.cat((jobsInfor, machineInfor, selected_action), dim=1)
        instanceEmbedding = self.sequenceLSTM(input.unsqueeze(0))[0]
        return instanceEmbedding

    def forward(self, state):
        stateEmebded = self.instanceEmbedding(state)
        for l in self.layers:
            output = l['a3'](self.activate_func(l['a2'](self.activate_func(l['a1'](stateEmebded)))))
        a_prob = F.softmax(output, dim=2)
        return a_prob

class critic(nn.Module):
    def __init__(self, args):
        super(critic, self).__init__()
        self.n_action = args.n_action
        self.machineEmbedding = nn.Linear(args.machine_embedding_dim, args.embedding_dim)
        self.jobsEmbedding = nn.Linear(args.jobs_embedding_dim, args.embedding_dim)
        self.actionEmbedding = nn.Linear(args.action_embedding_dim, args.embedding_dim)
        self.sequenceLSTM = nn.LSTM(3 * args.embedding_dim, args.embedding_dim, batch_first=True)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.layers = nn.ModuleList([nn.ModuleDict({
            'a1': nn.Linear(args.embedding_dim, args.embedding_dim),
            'a2': nn.Linear(args.embedding_dim, args.embedding_dim),
            'a3': nn.Linear(args.embedding_dim, 1)
        }) for i in range(1)])

    def instanceEmbedding(self, state):
        jobsInfor = torch.Tensor(state[0])
        machineInfor = torch.Tensor(state[1])
        selected_action = torch.Tensor(state[2])
        jobsInfor = self.jobsEmbedding(jobsInfor)
        machineInfor = self.machineEmbedding(machineInfor)
        selected_action = self.actionEmbedding(selected_action)
        input = torch.cat((jobsInfor, machineInfor, selected_action), dim=1)
        instanceEmbedding = self.sequenceLSTM(input.unsqueeze(0))[0]
        return instanceEmbedding

    def forward(self, state):
        stateEmebded = self.instanceEmbedding(state)
        for l in self.layers:
            output = l['a3'](self.activate_func(l['a2'](self.activate_func(l['a1'](stateEmebded)))))
        a_prob = F.softmax(output, dim=1)
        return a_prob

class Agent:
    def __init__(self, id, args):
        self.task_id = id
        self.num_tasks = args.num_tasks
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.last_action = np.random.randint(args.num_tasks)
        self.save_cycle = args.save_freq
        self.state_dim = args.RP_dim
        self.machine_state_dim = args.machine_state_dim
        self.model_file_dir = 'E:/Phd_work/myWork/Code/2' + '/trained_models/Controller_agent/'
        self.actor = actor_LSTM(args)
        self.critic = critic_LSTM(args)
        # self.actor = actor(args)
        # self.critic = critic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)


    def select_action(self, s):

        s[0] = [s[0]]
        s[1] = [s[1]]

        with torch.no_grad():
            dist = Categorical(self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0][0], a_logprob.numpy()[0][0]

    def select_action_test(self, s):
        s[0] = [s[0]]
        s[1] = [s[1]]
        if np.random.uniform() < 0.5:
            with torch.no_grad():
                dist = Categorical(self.actor(s))
                a = dist.sample()
                action_pro = dist.log_prob(a)
                a = np.argmax(action_pro)
                a = a.numpy()
        else:
            a = np.random.randint(8)
        return a, None

    def update(self, replay_buffer, total_steps):
        # s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        s = replay_buffer.s
        a = replay_buffer.a
        a_logprob = replay_buffer.a_logprob
        r = replay_buffer.r
        s_ = replay_buffer.s_
        dw = replay_buffer.dw
        done = replay_buffer.done

        adv = []
        gae = 0
        jobs = np.zeros((self.batch_size+1, self.state_dim))
        machine = np.zeros((self.batch_size+1, self.state_dim))

        jobs_ = np.zeros((self.batch_size+1, self.state_dim))
        machine_ = np.zeros((self.batch_size+1, self.state_dim))

        with torch.no_grad():  # adv and v_target have no gradient
            for i in range(self.batch_size):
                jobs[i] = s[i][0][:]
                machine[i] = s[i][1][:]
                jobs_[i] = s_[i][0][:]
                machine_[i] = s_[i][1][:]
            training_input = [jobs, machine]
            training_input_next = [jobs_, machine_]
            vs = self.critic(training_input)
            vs_ = self.critic(training_input_next)
            dw = torch.from_numpy(dw)
            r = torch.from_numpy(r)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        for _ in range(self.K_epochs):
            probs = self.actor(training_input)
            if torch.isnan(probs).any():
                print("probs张量包含NaN值")
            # assert torch.allclose(torch.sum(probs, dim=2), torch.ones((1, 33)))
            dist_now = Categorical(probs=probs)
            dist_entropy = dist_now.entropy().view(-1, 1)
            # a = torch.tensor(a)
            a_logprob_now = dist_now.log_prob(torch.tensor(a).squeeze()).view(-1, 1)
            # a_logprob_now = dist_now.log_prob(a.sequeeze()).view(-1, 1)
            ratios = torch.exp(a_logprob_now - torch.Tensor(a_logprob))
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
            # actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
            actor_loss = torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
            self.optimizer_actor.zero_grad()
            actor_loss.mean().backward()
            if self.use_grad_clip:  # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()
            v_s = self.critic(training_input)
            critic_loss = F.mse_loss(v_target, v_s)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:  # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def select_action_withAction(self, s):
        s[0] = [s[0]]
        s[1] = [s[1]]
        s[2] = [s[2]]

        with torch.no_grad():
            dist = Categorical(self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0][0], a_logprob.numpy()[0][0]

    def update_withAction(self, replay_buffer, total_steps):
        # s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        s = replay_buffer.s
        a = replay_buffer.a
        a_logprob = replay_buffer.a_logprob
        r = replay_buffer.r
        s_ = replay_buffer.s_
        dw = replay_buffer.dw
        done = replay_buffer.done
        high_level_a = replay_buffer.high_level_a

        adv = []
        gae = 0
        jobs = np.zeros((self.batch_size+1, self.state_dim))
        machine = np.zeros((self.batch_size+1, self.state_dim))
        action = np.zeros((self.batch_size+1, self.num_tasks))

        jobs_ = np.zeros((self.batch_size+1, self.state_dim))
        machine_ = np.zeros((self.batch_size+1, self.state_dim))
        action_ = np.zeros((self.batch_size+1, self.num_tasks))

        with torch.no_grad():  # adv and v_target have no gradient
            for i in range(self.batch_size):
                jobs[i] = s[i][0][:]
                machine[i] = s[i][1][:]
                action[i] = high_level_a[i]
                jobs_[i] = s_[i][0][:]
                machine_[i] = s_[i][1][:]
                action_[i] = high_level_a[i+1]
            training_input = [jobs, machine, action]
            training_input_next = [jobs_, machine_, action_]
            vs = self.critic(training_input)
            vs_ = self.critic(training_input_next)
            dw = torch.from_numpy(dw)
            r = torch.from_numpy(r)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        for _ in range(self.K_epochs):
            probs = self.actor(training_input)
            if torch.isnan(probs).any():
                print("probs张量包含NaN值")
            # assert torch.allclose(torch.sum(probs, dim=2), torch.ones((1, 33)))
            dist_now = Categorical(probs=probs)
            dist_entropy = dist_now.entropy().view(-1, 1)
            # a = torch.tensor(a)
            a_logprob_now = dist_now.log_prob(torch.tensor(a).squeeze()).view(-1, 1)
            # a_logprob_now = dist_now.log_prob(a.sequeeze()).view(-1, 1)
            ratios = torch.exp(a_logprob_now - torch.Tensor(a_logprob))
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
            actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
            self.optimizer_actor.zero_grad()
            actor_loss.mean().backward()
            if self.use_grad_clip:  # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()
            v_s = self.critic(training_input)
            critic_loss = F.mse_loss(v_target, v_s)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:  # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save_model(self, model_path, train_step, task_id):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        sub_file_name = str(train_step // self.save_cycle)
        # if not os.path.exists(self.model_file_dir):
        #     os.makedirs(self.model_file_dir)
        path = model_path + '/' + sub_file_name + '_agent_' + str(task_id) + '.pkl'
        # path = self.model_file_dir + sub_file_name + '_agent_' + str(task_id)   # +'_net_parameters.pkl'
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
