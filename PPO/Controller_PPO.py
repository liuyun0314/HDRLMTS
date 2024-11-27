import torch
import os.path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

########
class controller_Actor(nn.Module):
    def __init__(self, args):
        super(controller_Actor, self).__init__()
        self.num_hidden_layer = args.num_hidden_layer
        self.use_attention_fusion = args.use_attention_fusion
        self.input_layer1 = nn.Linear(args.task_state_dim, args.hidden_width)
        self.input_layer2 = nn.Linear(args.task_state_dim, args.hidden_width)
        self.input_layer3 = nn.Linear(args.task_state_dim, args.hidden_width)
        self.input_layer4 = nn.Linear(args.machine_state_dim, args.hidden_width)
        self.input_layer5 = nn.Linear(args.action_state_dim, args.action_embedding_width)
        self.hidden_layer1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.hidden_layer2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.hidden_layer3 = nn.Linear(args.hidden_width, args.hidden_width)
        self.hidden_layer4 = nn.Linear(args.hidden_width, args.hidden_width)
        self.hidden_layer5 = nn.Linear(args.action_embedding_width, args.hidden_width)
        self.output_layer = nn.Linear(4 * args.hidden_width+1, args.num_tasks)
        self.self_attention = nn.MultiheadAttention(4 * args.hidden_width+1, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.logsoft = nn.LogSoftmax(1)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.input_layer1)
            orthogonal_init(self.input_layer2)
            orthogonal_init(self.input_layer3)
            orthogonal_init(self.input_layer4)
            orthogonal_init(self.input_layer5)
            orthogonal_init(self.hidden_layer1)
            orthogonal_init(self.hidden_layer2)
            orthogonal_init(self.hidden_layer3)
            orthogonal_init(self.hidden_layer4)
            orthogonal_init(self.hidden_layer5)
            orthogonal_init(self.output_layer, gain=0.01)

    def self_attention_fusion(self, s1, s2, s3, s4, s5):
        # 将输入堆叠成 (sequence_length, batch_size, hidden_width) 的形状
        stacked_inputs = torch.cat([s1, s2, s3, s4, s5], dim=1)
        # stacked_inputs = stacked_inputs.permute(1, 0)  # 调整维度顺序
        # 自注意力机制
        stacked_inputs = stacked_inputs.unsqueeze(0)
        attention_output, _ = self.self_attention(stacked_inputs, stacked_inputs, stacked_inputs)
        # stacked_inputs = stacked_inputs.permute(1, 0)
        # 将自注意力输出与原始输入相加
        fused_output = attention_output.sum(dim=0)

        return fused_output

    def forward(self, states):

        task_input1 = states[0]
        task_input1 = torch.Tensor(task_input1)
        task_input2 = states[1]
        task_input2 = torch.Tensor(task_input2)
        task_input3 = states[2]
        task_input3 = torch.Tensor(task_input3)
        machine_input = states[3]
        machine_input = torch.Tensor(machine_input)
        action_input = states[4]
        action_input = torch.Tensor(action_input)

        # task_input1 = torch.unsqueeze(torch.tensor(task_input1, dtype=torch.float), 0)
        # task_input2 = torch.unsqueeze(torch.tensor(task_input2, dtype=torch.float), 0)
        # task_input3 = torch.unsqueeze(torch.tensor(task_input3, dtype=torch.float), 0)
        # machine_input = torch.unsqueeze(torch.tensor(machine_input, dtype=torch.float), 0)
        # action_input = torch.unsqueeze(torch.tensor(action_input, dtype=torch.float), 0)

        s1 = self.activate_func(self.input_layer1(task_input1))
        s2 = self.activate_func(self.input_layer2(task_input2))
        s3 = self.activate_func(self.input_layer3(task_input3))
        s4 = self.activate_func(self.input_layer4(machine_input))
        s5 = self.activate_func(self.input_layer5(action_input))
        # for i in range(self.num_hidden_layer):
        #     s1 = self.activate_func(self.hidden_layer1(s1))
        #     s2 = self.activate_func(self.hidden_layer2(s2))
        #     s3 = self.activate_func(self.hidden_layer3(s3))
        #     s4 = self.activate_func(self.hidden_layer3(s4))
        #     s5 = self.activate_func(self.hidden_layer3(s5))
        if self.use_attention_fusion:

            fused_output = self.self_attention_fusion(s1, s2, s3, s4, s5)
            output = self.activate_func(self.output_layer(fused_output))
        else:
            input = torch.cat((s1, s2, s3, s4, s5), dim=1)
            output = self.activate_func(self.output_layer(input))
        a_prob = F.softmax(output, dim=1)
        self.logsoft(a_prob)
        return a_prob

class controller_Critic(nn.Module):
    def __init__(self, args):
        super(controller_Critic, self).__init__()
        self.num_hidden_layer = args.num_hidden_layer
        self.use_attention_fusion = False #  args.use_attention_fusion
        self.input_layer1 = nn.Linear(args.task_state_dim, args.hidden_width)
        self.input_layer2 = nn.Linear(args.task_state_dim, args.hidden_width)
        self.input_layer3 = nn.Linear(args.task_state_dim, args.hidden_width)
        self.input_layer4 = nn.Linear(args.machine_state_dim, args.hidden_width)
        self.input_layer5 = nn.Linear(args.action_state_dim, args.hidden_width)
        self.hidden_layer1 = nn.Linear(args.hidden_width, args.hidden_width)
        self.hidden_layer2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.hidden_layer3 = nn.Linear(args.hidden_width, args.hidden_width)
        self.hidden_layer4 = nn.Linear(args.hidden_width, args.hidden_width)
        self.hidden_layer5 = nn.Linear(args.hidden_width, args.hidden_width)
        self.output_layer = nn.Linear(5*args.hidden_width, 1)
        self.self_attention = nn.MultiheadAttention(5 * args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.input_layer1)
            orthogonal_init(self.input_layer2)
            orthogonal_init(self.input_layer3)
            orthogonal_init(self.input_layer4)
            orthogonal_init(self.input_layer5)
            orthogonal_init(self.hidden_layer1)
            orthogonal_init(self.hidden_layer2)
            orthogonal_init(self.hidden_layer3)
            orthogonal_init(self.hidden_layer4)
            orthogonal_init(self.hidden_layer5)
            orthogonal_init(self.output_layer)

    def self_attention_fusion(self, s1, s2, s3, s4, s5):
        stacked_inputs = torch.cat([s1, s2, s3, s4, s5], dim=1)
        stacked_inputs = stacked_inputs.unsqueeze(0)
        attention_output, _ = self.self_attention(stacked_inputs, stacked_inputs, stacked_inputs)
        fused_output = attention_output.sum(dim=0)

        return fused_output

    def forward(self, states):
        task_input1 = states[0]
        task_input1 = torch.Tensor(task_input1)
        task_input2 = states[1]
        task_input2 = torch.Tensor(task_input2)
        task_input3 = states[2]
        task_input3 = torch.Tensor(task_input3)
        machine_input = states[3]
        machine_input = torch.Tensor(machine_input)
        action_input = states[4]
        action_input = torch.Tensor(action_input)

        s1 = self.activate_func(self.input_layer1(task_input1))
        s2 = self.activate_func(self.input_layer2(task_input2))
        s3 = self.activate_func(self.input_layer3(task_input3))
        s4 = self.activate_func(self.input_layer4(machine_input))
        s5 = self.activate_func(self.input_layer5(action_input))
        if self.use_attention_fusion:
            fused_output = self.self_attention_fusion(s1, s2, s3, s4, s5)
            output = self.activate_func(self.output_layer(fused_output))
        else:
            input = torch.cat((s1, s2, s3, s4, s5), dim=1)
            output = self.activate_func(self.output_layer(input))
        return output

#########

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.global_state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.num_tasks)
        # Trick 10: use tahn
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]


        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, state):
        state = (state - state.mean()) / state.std()
        s = self.activate_func(self.fc1(state))
        s = self.activate_func(self.fc2(s))
        a_prob = self.fc3(s)
        a_prob = F.softmax(a_prob, dim=1)
        return a_prob

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.global_state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        # Trick 10: use tahn
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, state):
        s = self.activate_func(self.fc1(state))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

class PPO:
    def __init__(self, args):
        self.batch_size = args.global_batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.control_lr_a  # Learning rate of actor
        self.lr_c = args.control_lr_c  # Learning rate of critic
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
        self.task_state_dim = args.task_state_dim
        self.machine_state_dim = args.machine_state_dim
        self.model_file_dir = 'E:/Phd_work/myWork/Code/2' + '/trained_models/Controller_agent/'

        self.actor = controller_Actor(args)
        self.critic = controller_Critic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):
        s[0] = [s[0]]
        s[1] = [s[1]]
        s[2] = [s[2]]
        s[3] = [s[3]]
        s[4] = [s[4]]
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def select_action(self, s):

        # s = torch.squeeze(s)
        # s = [s[0], s[1], s[2], s[3], s[4]]
        s[0] = [s[0]]
        s[1] = [s[1]]
        s[2] = [s[2]]
        s[3] = [s[3]]
        s[4] = [s[4]]
        with torch.no_grad():
            # prob = self.actor(s)
            # a = np.argmax(prob)
            # a_logprob = np.max(prob)
            # a_logprob = self.actor(s)
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0], a_logprob.numpy()[0]

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
        task1 = np.zeros((self.batch_size, self.task_state_dim))
        task2 = np.zeros((self.batch_size, self.task_state_dim))
        task3 = np.zeros((self.batch_size, self.task_state_dim))
        machine = np.zeros((self.batch_size, self.machine_state_dim))
        actions = np.zeros((self.batch_size, 3))

        task1_ = np.zeros((self.batch_size, self.task_state_dim))
        task2_ = np.zeros((self.batch_size, self.task_state_dim))
        task3_ = np.zeros((self.batch_size, self.task_state_dim))
        machine_ = np.zeros((self.batch_size, self.machine_state_dim))
        actions_ = np.zeros((self.batch_size, 3))

        with torch.no_grad():  # adv and v_target have no gradient
            for i in range(self.batch_size):
                task1[i] = s[i][0]
                task1_[i] = s_[i][0]
                task2[i] = s[i][1]
                task2_[i] = s_[i][1]
                task3[i] = s[i][2]
                task3_[i] = s_[i][2]
                machine[i] = s[i][3]
                machine_[i] = s_[i][3]
                actions[i] = s[i][4]
                actions_[i] = s_[i][4]
            training_input = [task1, task2, task3, machine, actions]
            training_input_next = [task1_, task2_, task3_, machine_, actions_]

            vs = self.critic(training_input)
            vs_ = self.critic(training_input_next)
            dw = torch.from_numpy(dw)
            r = torch.from_numpy(r)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            # done = torch.from_numpy(done)
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        # Optimize policy for K epochs:
        # a = torch.tensor(a)
        for _ in range(self.K_epochs):
            dist_now = Categorical(probs=self.actor(training_input))
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

    def save_model(self, model_path, train_step):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        sub_file_name = str(train_step // self.save_cycle)
        path = model_path + '/' + sub_file_name + '_Controller_agent.pkl'
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
