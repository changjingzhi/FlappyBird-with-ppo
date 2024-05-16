## 这是使用ppo算法+FalppyBird环境的强化学习实例。
更详细的介绍请看我的博客[博客链接](https://chenlidbk.xyz/2024/04/30/tiankeng6/)

## 基本资料
# 定义超参数和环境
actor_lr = 1e-6
critic_lr = 1e-5
num_episodes = 100000
hidden_dim = 64
gamma = 0.99
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env_name = 'FlappyBird-v0'

基本上一轮在GTX3080显卡上需要跑
![训练过程](img\img1.png)

训练结果截图
![结果过程截图1](img\img2.png)
![结果过程截图2](img\img3.png)