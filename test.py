import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn.functional as F
from time import sleep
import os

# 定义游戏环境和设备
env = gym.make("FlappyBird-v0", render_mode="human")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型路径
model_path = "model/ppo_bed_model.pth"

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

# 定义 PPO 代理类
class PPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, model_path):
        self.model = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.model.load_state_dict(torch.load(model_path)['actor'])  # 加载模型的 'actor' 部分
        self.model.eval()  # 设置模型为评估模式

    def take_action(self, state):
        state_tensor = torch.tensor([state], dtype=torch.float).to(device)
        with torch.no_grad():
            action = self.model(state_tensor).argmax().item()
        return action

# 创建 PPO 代理
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 64
agent = PPOAgent(state_dim, hidden_dim, action_dim, model_path)

# 运行游戏并观察动画
state, _ = env.reset()
done = False
total_reward = 0

try:
    while not done:
        # 使用代理选择动作
        action = agent.take_action(state)

        # 执行动作并观察游戏动画
        next_state, reward, terminate, truncated, _ = env.step(action)
        done = terminate or truncated
        total_reward += reward
        env.render()

        state = next_state

finally:
    # 输出游戏得分
    print(f"游戏结束，得分：{total_reward}")
    # 手动关闭游戏画面
    input("按任意键关闭游戏画面...")
    env.close()  # 关闭游戏环境
