from __future__ import annotations
import os
import sys
import math
import time
import argparse
import datetime
from typing import Tuple, List, Deque, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import csv


# ------------------------------------------------------------
# 超参数（含 ICM）
# ------------------------------------------------------------
@dataclass
class Cfg:
    # 环境
    GRID_W: int = 10
    GRID_H: int = 10
    CELL_SIZE: int = 40
    WINDOW_W: int = GRID_W * CELL_SIZE
    WINDOW_H: int = GRID_H * CELL_SIZE
    FPS: int = 120
    # 训练
    MAX_EPISODES: int = 30_000  # 增加训练轮数
    SAVE_FREQ: int = 500
    LOG_FREQ: int = 10
    GAMMA: float = 0.995  # 更高的折扣因子，重视长期奖励
    LAMBDA: float = 0.95
    EPS_CLIP: float = 0.2
    VALUE_COEF: float = 0.5
    ENTROPY_COEF: float = 0.01
    LR: float = 3e-4  # 提高学习率，使用学习率调度
    BATCH_SIZE: int = 2048
    MINI_BATCH: int = 256
    K_EPOCHS: int = 8
    MAX_GRAD_NORM: float = 0.5
    # 学习率调度
    LR_DECAY: float = 0.995  # 学习率衰减因子
    MIN_LR: float = 1e-5     # 最小学习率
    # 网络
    STATE_DIM: int = 24 
    HIDDEN_DIM: int = 512
    
      # 优化网络结构，使用更合理的维度
    ACTION_DIM: int = 3
    # ICM
    ICM_FEATURE_DIM: int = 128
    ICM_LR: float = 3e-4    # 调整 ICM 学习率
    ICM_BETA: float = 0.2
    ICM_ETA: float = 0.1    # 增强内在奖励影响
    ICM_ETA_DECAY: float = 0.999  # 内在奖励衰减
    # 渲染
    RENDER_EVERY: int = 1000
    # 奖励
    REWARD_FOOD: float = 10.0    # 食物奖励
    REWARD_DEATH: float = -10.0   # 死亡惩罚
    REWARD_STEP: float = 0.0      # 步数惩罚
    REWARD_SURVIVAL: float = 0.1  # 生存奖励
    REWARD_CLOSE_FOOD: float = 0.5  # 接近食物奖励
    REWARD_AWAY_FOOD: float = -0.2 # 远离食物惩罚
    # 课程学习
    CURRICULUM_LEARNING: bool =   True  # 启用课程学习
    GRID_SIZE_START: int = 6         # 初始网格大小
    GRID_SIZE_END: int = 10          # 最终网格大小
    CURRICULUM_EPISODES: int = 5000  # 课程学习阶段轮数

# ------------------------------------------------------------
# 颜色
# ------------------------------------------------------------
class Color:
    BG = (10, 10, 10)
    SNAKE_HEAD = (0, 200, 0)
    SNAKE_BODY = (0, 120, 0)
    FOOD = (220, 20, 20)
    TEXT = (255, 255, 255)

# ------------------------------------------------------------
# 环境
# ------------------------------------------------------------
class SnakeEnv:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> np.ndarray:
        self.direction: int = 0  # 0: Right, 1: Up, 2: Left, 3: Down
        mid_x, mid_y = self.cfg.GRID_W // 2, self.cfg.GRID_H // 2
        self.snake: Deque[Tuple[int, int]] = deque([(mid_x, mid_y)])
        self._spawn_food()
        self.done: bool = False
        self.steps: int = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.steps += 1
        # 0: Turn Right, 1: Go Straight, 2: Turn Left
        if action == 0:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        x, y = self.snake[0]
        dx, dy = [(1, 0), (0, -1), (-1, 0), (0, 1)][self.direction]
        nx, ny = x + dx, y + dy
        new_head = (nx, ny)

        # Check collision with wall or self
        if (nx < 0 or nx >= self.cfg.GRID_W or
            ny < 0 or ny >= self.cfg.GRID_H or
            new_head in self.snake):
            self.done = True
            return self._get_state(), self.cfg.REWARD_DEATH, True

        self.snake.appendleft(new_head)

        # 基础奖励
        reward = self.cfg.REWARD_STEP + self.cfg.REWARD_SURVIVAL
        
        # 智能距离奖励
        old_dist = abs(x - self.food[0]) + abs(y - self.food[1])
        new_dist = abs(nx - self.food[0]) + abs(ny - self.food[1])
        dist_change = old_dist - new_dist
        
        # 根据距离变化给予奖励
        if dist_change > 0:  # 接近食物
            reward += self.cfg.REWARD_CLOSE_FOOD * dist_change
        elif dist_change < 0:  # 远离食物
            reward += self.cfg.REWARD_AWAY_FOOD * abs(dist_change)
        
        # 检查是否吃到食物
        if new_head == self.food:
            reward += self.cfg.REWARD_FOOD
            # 额外奖励：蛇越长，奖励越高
            length_bonus = len(self.snake) * 0.1
            reward += length_bonus
            self._spawn_food()
        else:
            self.snake.pop()
            
        # 避免循环奖励：检测是否在重复路径
        if hasattr(self, 'position_history'):
            self.position_history.append(new_head)
            # 如果最近10步中有重复位置，给予轻微惩罚
            if len(self.position_history) > 10:
                recent_positions = list(self.position_history)[-10:]
                if len(set(recent_positions)) < 8:  # 如果重复率过高
                    reward -= 0.05
        else:
            self.position_history = deque([new_head], maxlen=20)
            
        # 安全性奖励：检测是否接近危险
        danger_count = self._count_nearby_dangers(nx, ny)
        if danger_count > 0:
            reward -= danger_count * 0.1  # 接近危险给予惩罚
        


        return self._get_state(), reward, self.done
        
    def _count_nearby_dangers(self, x: int, y: int) -> int:
        """计算位置周围3x3区域内的危险数量"""
        danger_count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                # 检查是否撞墙或撞到自己
                if (nx < 0 or nx >= self.cfg.GRID_W or
                    ny < 0 or ny >= self.cfg.GRID_H or
                    (nx, ny) in self.snake):
                    danger_count += 1
        return danger_count

    def _spawn_food(self) -> None:
        while True:
            pos = (np.random.randint(0, self.cfg.GRID_W),
                   np.random.randint(0, self.cfg.GRID_H))
            if pos not in self.snake:
                self.food = pos
                break

    def _get_state(self) -> np.ndarray:
        head = self.snake[0]
        x, y = head
        fx, fy = self.food

        # Danger perception: left, front, right relative to current direction
        danger_left, danger_front, danger_right = self._danger_perception()

        # One-hot encoding of current direction
        dir_l = int(self.direction == 2)
        dir_r = int(self.direction == 0)
        dir_u = int(self.direction == 1)
        dir_d = int(self.direction == 3)

        # One-hot encoding of food direction relative to head
        food_left = int(fx < x)
        food_right = int(fx > x)
        food_up = int(fy < y)
        food_down = int(fy > y)

        # Normalized distance to walls in each direction
        dist_left = x / self.cfg.GRID_W
        dist_right = (self.cfg.GRID_W - 1 - x) / self.cfg.GRID_W
        dist_up = y / self.cfg.GRID_H
        dist_down = (self.cfg.GRID_H - 1 - y) / self.cfg.GRID_H

        # 新增：8个方向上的蛇身存在情况
        body_8dir = self._body_in_8_directions()

        # State vector
        state = np.concatenate((
            np.array([
                danger_left, danger_front, danger_right, # 0,1,2
                dir_l, dir_r, dir_u, dir_d,              # 3,4,5,6
                food_left, food_right, food_up, food_down, # 7,8,9,10
                dist_left, dist_right, dist_up, dist_down, # 11,12,13,14
                len(self.snake) / (self.cfg.GRID_W * self.cfg.GRID_H) # 15
            ], dtype=np.float32),
            body_8dir # 16-23
        ))
        return state

    def _danger_perception(self) -> Tuple[int, int, int]:
        x, y = self.snake[0]
        # Directions: Right, Up, Left, Down
        dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        left_dir = (self.direction + 1) % 4
        right_dir = (self.direction - 1) % 4
        front_dir = self.direction

        def collision(nx, ny):
            return (nx < 0 or nx >= self.cfg.GRID_W or
                    ny < 0 or ny >= self.cfg.GRID_H or
                    (nx, ny) in self.snake)

        lx, ly = x + dirs[left_dir][0], y + dirs[left_dir][1]
        fx, fy = x + dirs[front_dir][0], y + dirs[front_dir][1]
        rx, ry = x + dirs[right_dir][0], y + dirs[right_dir][1]

        return (int(collision(lx, ly)),
                int(collision(fx, fy)),
                int(collision(rx, ry)))

    def _body_in_8_directions(self) -> np.ndarray:
        """检查蛇头周围8个方向是否有身体"""
        head = self.snake[0]
        hx, hy = head
        body_set = set(self.snake) # 使用 set 加速查找

        # 8 directions: N, NE, E, SE, S, SW, W, NW
        directions_8 = [(0, -1), (1, -1), (1, 0), (1, 1),
                        (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        
        body_flags = []
        for dx, dy in directions_8:
            # 检查该方向上最近的非空单元格
            tx, ty = hx + dx, hy + dy
            while 0 <= tx < self.cfg.GRID_W and 0 <= ty < self.cfg.GRID_H:
                if (tx, ty) in body_set:
                    body_flags.append(1.0)
                    break
                elif (tx, ty) == self.food: # 如果先遇到食物，则该方向无身体
                     body_flags.append(0.0)
                     break
                tx += dx
                ty += dy
            else: # 循环正常结束，说明该方向上没有身体或食物
                body_flags.append(0.0)
        
        return np.array(body_flags, dtype=np.float32)


    def render(self, surf: pygame.Surface) -> None:
        surf.fill(Color.BG)
        # Draw food
        fx, fy = self.food
        pygame.draw.rect(surf, Color.FOOD,
                         (fx * self.cfg.CELL_SIZE + 2, fy * self.cfg.CELL_SIZE + 2,
                          self.cfg.CELL_SIZE - 4, self.cfg.CELL_SIZE - 4))
        # Draw snake
        for idx, (sx, sy) in enumerate(self.snake):
            color = Color.SNAKE_HEAD if idx == 0 else Color.SNAKE_BODY
            pygame.draw.rect(surf, color,
                             (sx * self.cfg.CELL_SIZE, sy * self.cfg.CELL_SIZE,
                              self.cfg.CELL_SIZE, self.cfg.CELL_SIZE))

# ------------------------------------------------------------
# 主干网络 (Actor-Critic) - 优化版本
# ------------------------------------------------------------
class ResidualBlock(nn.Module):
    """残差块，用于提高网络深度和稳定性"""
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用层归一化提高稳定性
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)  # 残差连接

class ActorCritic(nn.Module):
    def __init__(self, s_dim: int, h_dim: int, a_dim: int) -> None:
        super().__init__()
        # 使用更优化的网络架构
        self.input_layer = nn.Sequential(
            nn.Linear(s_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU()
        )
        
        # 使用多个残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(h_dim, dropout_rate=0.1) for _ in range(3)
        ])
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.LayerNorm(h_dim // 2),
            nn.ReLU()
        )
        
        # 独立的Actor和Critic头
        self.actor = nn.Sequential(
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.ReLU(),
            nn.Linear(h_dim // 4, a_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.ReLU(),
            nn.Linear(h_dim // 4, 1)
        )
        
        self.apply(self._init_weights)
        
        # 初始化输出层权重
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.actor[-1].bias, 0)
        nn.init.constant_(self.critic[-1].bias, 0)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Linear):
            # 使用正交初始化，更适合强化学习
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        
        # 通过残差块
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        features = self.feature_layer(x)
        
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        return logits, value

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> Tuple[int, float, float]:
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()
        return action.item(), log_prob, value.item()

    def evaluate(self, x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(a)
        entropy = dist.entropy()
        return log_probs, value, entropy

# ============================================================
# ICM 网络 - 优化版本
# ============================================================
class ICMEncoder(nn.Module):
    """ICM编码器，使用残差连接提高特征提取能力"""
    def __init__(self, s_dim: int, feature_dim: int):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256, dropout_rate=0.05) for _ in range(2)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return self.output_layer(x)

class ICM(nn.Module):
    """
    优化的ICM网络，使用更好的架构和正则化
    s_dim  -> feature_dim (encoder)
    (feature_dim, action) -> next_feature (forward)
    (feature_dim, next_feature) -> action_logits (inverse)
    """
    def __init__(self, s_dim: int, a_dim: int, feature_dim: int):
        super().__init__()
        # 使用改进的编码器
        self.encoder = ICMEncoder(s_dim, feature_dim)
        
        # 优化的前向模型
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + a_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # 优化的逆向模型
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, a_dim)
        )
        
        # 使用正交初始化
        self.apply(self._init_weights)
        
    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
            nn.init.constant_(m.bias, 0)

    def feature(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def pred_next_feature(self, phi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi, action], dim=-1)
        return self.forward_model(x)

    def pred_action_logits(self, phi: torch.Tensor, next_phi: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi, next_phi], dim=-1)
        return self.inverse_model(x)

# ------------------------------------------------------------
# 缓存 (Rollout Buffer)
# ------------------------------------------------------------
class RolloutBuffer:
    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        # ICM 需要存储下一个状态
        self.next_states: List[torch.Tensor] = []

    def push(self, state, action, logprob, reward, value, done, next_state) -> None:
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_states.append(torch.tensor(next_state, dtype=torch.float32))

    def get(self) -> Tuple[torch.Tensor, ...]:
        return (torch.stack(self.states),
                torch.tensor(self.actions, dtype=torch.long),
                torch.tensor(self.logprobs, dtype=torch.float32),
                torch.tensor(self.rewards, dtype=torch.float32),
                torch.tensor(self.values, dtype=torch.float32),
                torch.tensor(self.dones, dtype=torch.float32),
                torch.stack(self.next_states))

    def clear(self) -> None:
        self.__init__()

# ------------------------------------------------------------
# PPO + ICM 智能体 - 优化版本
# ------------------------------------------------------------
class PPOAgent:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episode_count = 0
        self.icm_eta = cfg.ICM_ETA  # 动态内在奖励系数
        
        # 初始化 Actor-Critic 网络
        self.net = ActorCritic(cfg.STATE_DIM, cfg.HIDDEN_DIM, cfg.ACTION_DIM).to(self.device)
        
        # 初始化 ICM 网络
        self.icm = ICM(cfg.STATE_DIM, cfg.ACTION_DIM, cfg.ICM_FEATURE_DIM).to(self.device)
        
        # 优化器 - 使用AdamW提高泛化能力
        self.optimizer = optim.AdamW(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4, eps=1e-5)
        self.icm_optimizer = optim.AdamW(self.icm.parameters(), lr=cfg.ICM_LR, weight_decay=1e-4, eps=1e-5)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.LR_DECAY)
        self.icm_scheduler = optim.lr_scheduler.ExponentialLR(self.icm_optimizer, gamma=cfg.LR_DECAY)
        
        # 经验缓存
        self.buffer = RolloutBuffer()
        
        # 损失函数
        self.mse = nn.MSELoss()
        
        # 课程学习相关
        self.current_grid_size = cfg.GRID_SIZE_START if cfg.CURRICULUM_LEARNING else cfg.GRID_SIZE_END
        
    def update_curriculum(self, episode: int) -> None:
        """更新课程学习进度"""
        if not self.cfg.CURRICULUM_LEARNING:
            return
            
        # 线性增加网格大小
        progress = min(episode / self.cfg.CURRICULUM_EPISODES, 1.0)
        new_size = int(self.cfg.GRID_SIZE_START + 
                      (self.cfg.GRID_SIZE_END - self.cfg.GRID_SIZE_START) * progress)
        
        if new_size != self.current_grid_size:
            self.current_grid_size = new_size
            print(f"课程学习更新: 网格大小增加到 {self.current_grid_size}x{self.current_grid_size}")
    
    def update_learning_rate(self) -> None:
        """更新学习率"""
        # 确保学习率不低于最小值
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr > self.cfg.MIN_LR:
            self.scheduler.step()
            self.icm_scheduler.step()
            
        # 更新内在奖励系数
        self.icm_eta *= self.cfg.ICM_ETA_DECAY
        self.icm_eta = max(self.icm_eta, 0.01)  # 确保不低于最小值

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, value = self.net.forward(state_t)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            # 选择概率最高的动作
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).item()
        return action.item(), log_prob, value.item()

    def store(self, *args) -> None:
        self.buffer.push(*args)

    def compute_returns_advantages(self, last_value: float, done: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)

        # 如果 episode 未结束，则使用 critic 网络估计的 value 作为最后一步的 value
        if not done:
            last_value_tensor = torch.tensor(last_value, dtype=torch.float32, device=self.device)
        else:
            last_value_tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        # 将最后一步的 value 添加到 values 列表末尾，用于计算 GAE
        values = torch.cat([values, last_value_tensor.unsqueeze(0)])

        returns = []
        gae = 0.0
        
        # 从后往前计算 GAE 和 Returns
        for step in reversed(range(len(rewards))):
            # 计算 TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            # 如果是 episode 的最后一步，则 V(s_{t+1}) = 0
            delta = rewards[step] + self.cfg.GAMMA * values[step + 1] * (1.0 - dones[step]) - values[step]
            # 计算 GAE: A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
            gae = delta + self.cfg.GAMMA * self.cfg.LAMBDA * (1.0 - dones[step]) * gae
            # Returns_t = A_t + V(s_t)
            returns.insert(0, gae + values[step])

        returns_t = torch.stack(returns)
        # Advantages = Returns - Values
        advantages = returns_t - values[:-1] 
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns_t, advantages

    def update(self, last_value: float, done: bool) -> Dict[str, float]:
        # 1. 从 buffer 中获取数据
        states, actions, old_logprobs, rewards, values, dones, next_states = self.buffer.get()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        next_states = next_states.to(self.device)
        
        # 2. ===== ICM 计算内在奖励 =====
        with torch.no_grad():
            # 编码当前状态和下一个状态
            phi = self.icm.feature(states)
            next_phi = self.icm.feature(next_states)
            # 使用 ICM 前向模型预测下一个特征
            pred_next_phi = self.icm.pred_next_feature(
                phi, 
                nn.functional.one_hot(actions, self.cfg.ACTION_DIM).float()
            )
            # 计算内在奖励: η * ||φ̂(s_{t+1}) - φ(s_{t+1})||^2
            intrinsic_reward = self.icm_eta * torch.sum((pred_next_phi - next_phi) ** 2, dim=-1)
            # 将内在奖励加到外部奖励上
            rewards += intrinsic_reward.cpu()

        # 3. 计算 Returns 和 Advantages (使用更新后的奖励)
        returns, advantages = self.compute_returns_advantages(last_value, done)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # 4. ===== PPO 更新 =====
        # Mini-batch PPO
        dataset_size = actions.size(0)
        idx = torch.arange(dataset_size)
        
        # 记录损失用于统计
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0
        
        for epoch in range(self.cfg.K_EPOCHS):
            # 每个 epoch 随机打乱数据索引
            idx = idx[torch.randperm(dataset_size)]
            for start in range(0, dataset_size, self.cfg.MINI_BATCH):
                # 获取当前 mini-batch 的索引
                sl = idx[start:start + self.cfg.MINI_BATCH]
                
                # 评估当前策略和价值
                logprobs, state_values, entropy = self.net.evaluate(states[sl], actions[sl])
                
                # 计算重要性采样比率
                ratios = torch.exp(logprobs - old_logprobs[sl])
                
                # 计算 PPO-Clip 损失
                surr1 = ratios * advantages[sl]
                surr2 = torch.clamp(ratios, 1 - self.cfg.EPS_CLIP, 1 + self.cfg.EPS_CLIP) * advantages[sl]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失 (使用Huber损失提高鲁棒性)
                critic_loss = nn.functional.huber_loss(state_values, returns[sl], delta=1.0)
                
                # 计算熵损失 (鼓励探索)
                entropy_loss = -entropy.mean()
                
                # 添加价值函数的KL散度惩罚，防止过拟合
                with torch.no_grad():
                    _, old_values, _ = self.net.evaluate(states[sl], actions[sl])
                value_kl = 0.5 * torch.mean((state_values - old_values) ** 2)
                
                # 总损失
                loss = (actor_loss +
                        self.cfg.VALUE_COEF * critic_loss +
                        self.cfg.ENTROPY_COEF * entropy_loss +
                        0.01 * value_kl)  # KL散度惩罚

                # 更新 Actor-Critic 网络
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                actor_grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.MAX_GRAD_NORM)
                
                # 检查梯度是否正常
                if not torch.isnan(actor_grad_norm):
                    self.optimizer.step()
                
                # 累计损失
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_loss.item()
                update_count += 1

        # 5. ===== ICM 更新 =====
        # 重新计算特征（不 detach，因为需要计算梯度）
        phi = self.icm.feature(states)
        # detach next_phi，因为 ICM 不更新编码器来预测下一个状态
        next_phi = self.icm.feature(next_states).detach()
        # 前向模型预测
        pred_next_phi = self.icm.pred_next_feature(
            phi, 
            nn.functional.one_hot(actions, self.cfg.ACTION_DIM).float()
        )
        # 前向模型损失
        forward_loss = self.mse(pred_next_phi, next_phi)
        
        # 逆向模型预测 (detach phi 和 next_phi，只更新逆向模型本身)
        pred_action_logits = self.icm.pred_action_logits(phi.detach(), next_phi)
        # 逆向模型损失 (分类交叉熵)
        inverse_loss = nn.functional.cross_entropy(pred_action_logits, actions)
        
        # ICM 总损失: (1-β) * L_F + β * L_I
        icm_loss = (1 - self.cfg.ICM_BETA) * forward_loss + self.cfg.ICM_BETA * inverse_loss

        # 更新 ICM 网络
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        
        # 梯度裁剪
        icm_grad_norm = nn.utils.clip_grad_norm_(self.icm.parameters(), self.cfg.MAX_GRAD_NORM)
        
        # 检查梯度是否正常
        if not torch.isnan(icm_grad_norm):
            self.icm_optimizer.step()

        # 更新学习率和课程学习
        self.update_learning_rate()
        self.episode_count += 1
        
        # 清空 buffer
        self.buffer.clear()
        
        # 计算平均损失
        avg_actor_loss = total_actor_loss / update_count if update_count > 0 else 0.0
        avg_critic_loss = total_critic_loss / update_count if update_count > 0 else 0.0
        avg_entropy = total_entropy / update_count if update_count > 0 else 0.0
        
        return {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "icm_loss": icm_loss.item(),
            "intrinsic_reward_mean": intrinsic_reward.mean().item(),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "icm_eta": self.icm_eta
        }

    def save(self, path: str) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "icm": self.icm.state_dict()
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.icm.load_state_dict(ckpt["icm"])

# ------------------------------------------------------------
# 训练主函数
# ------------------------------------------------------------
def train(cfg: Cfg) -> None:
    # 初始化环境和智能体
    env = SnakeEnv(cfg)
    agent = PPOAgent(cfg)
    
    # 创建日志目录
    run_name = f"snake_icm_optimized_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # CSV logger
    csv_path = f"{log_dir}/log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    # 添加新的日志列
    csv_writer.writerow(["episode", "reward", "length", "score", "actor_loss", "critic_loss", 
                       "entropy", "icm_loss", "intrinsic_reward_mean", "learning_rate", "icm_eta"])

    # Pygame 初始化 (用于可视化)
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_W, cfg.WINDOW_H))
    pygame.display.set_caption("Snake PPO+ICM Improved Training")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    # 记录最近 100 个 episode 的平均奖励
    running_reward: Deque[float] = deque(maxlen=100)
    best_score = -float("inf")
    global_step = 0

    for ep in range(1, cfg.MAX_EPISODES + 1):
        # 更新课程学习
        agent.update_curriculum(ep)
        
        # 如果使用课程学习，动态调整环境大小
        if cfg.CURRICULUM_LEARNING:
            # 这里可以动态调整环境参数，但为了简化，我们只调整智能体的感知
            pass
        
        state = env.reset()
        ep_reward, ep_len = 0.0, 0
        score = 0 # 初始长度为 1，所以分数是蛇身长度 - 1

        while True:
            # 选择动作 (训练阶段始终使用随机策略)
            action, logprob, value = agent.select_action(state, deterministic=False)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 存储经验 (包括 next_state 用于 ICM)
            agent.store(state, action, logprob, reward, value, done, next_state)
            
            state = next_state
            ep_reward += reward
            ep_len += 1
            global_step += 1
            
            # 更新当前分数
            score = len(env.snake) - 1

            # 渲染 (每隔一定 episode)
            if ep % cfg.RENDER_EVERY == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save(f"{log_dir}/interrupt.pt")
                        pygame.quit()
                        sys.exit()
                env.render(screen)
                # 显示更多信息
                info_text = f"Ep:{ep} Score:{score} Len:{len(env.snake)} LR:{agent.optimizer.param_groups[0]['lr']:.6f}"
                txt = font.render(info_text, True, Color.TEXT)
                screen.blit(txt, (10, 10))
                pygame.display.flip()
                clock.tick(cfg.FPS)

            if done:
                break

        running_reward.append(ep_reward)

        # 如果 buffer 中的经验足够多，则进行更新
        if len(agent.buffer.rewards) >= cfg.BATCH_SIZE:
             # 获取 episode 结束时的最后一个状态的价值 (用于 GAE 计算)
            with torch.no_grad():
                last_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                _, last_value = agent.net(last_state)
            
            # 执行 PPO 和 ICM 更新
            loss_dict = agent.update(last_value.item(), done=True) # done=True 因为是 episode 结束
        else:
            # 如果 buffer 不够大，则不更新，loss 设为 0
            loss_dict = {
                "actor_loss": 0.0, 
                "critic_loss": 0.0, 
                "entropy": 0.0, 
                "icm_loss": 0.0,
                "intrinsic_reward_mean": 0.0,
                "learning_rate": agent.optimizer.param_groups[0]['lr'],
                "icm_eta": agent.icm_eta
            }

        # 记录 TensorBoard
        writer.add_scalar("score/episode", score, ep)
        writer.add_scalar("reward/episode", ep_reward, ep)
        writer.add_scalar("length/episode", ep_len, ep)
        writer.add_scalar("reward/running100", np.mean(running_reward), ep)
        writer.add_scalar("training/learning_rate", agent.optimizer.param_groups[0]['lr'], ep)
        writer.add_scalar("training/icm_eta", agent.icm_eta, ep)
        
        # 记录损失
        for k, v in loss_dict.items():
            writer.add_scalar(f"loss/{k}", v, ep)
        
        # 记录课程学习进度
        if cfg.CURRICULUM_LEARNING:
            writer.add_scalar("curriculum/grid_size", agent.current_grid_size, ep)

        # 记录 CSV
        csv_writer.writerow([
            ep, ep_reward, ep_len, score,
            loss_dict["actor_loss"],
            loss_dict["critic_loss"],
            loss_dict["entropy"],
            loss_dict["icm_loss"],
            loss_dict["intrinsic_reward_mean"],
            loss_dict["learning_rate"],
            loss_dict["icm_eta"]
        ])
        csv_file.flush() # 确保数据写入磁盘

        # 定期保存模型
        if ep % cfg.SAVE_FREQ == 0:
            agent.save(f"{log_dir}/ckpt_{ep}.pt")

        # 保存最佳模型 (基于分数而非奖励，更符合游戏目标)
        if score > best_score:
            best_score = score
            agent.save(f"{log_dir}/best.pt")
            print(f"新的最佳分数: {best_score}")

        # 定期打印日志
        if ep % cfg.LOG_FREQ == 0:
            print(f"Ep {ep:5d} | "
                  f"Reward {ep_reward:7.2f} | "
                  f"Running {np.mean(running_reward):7.2f} | "
                  f"Best {best_score:7.2f} | "
                  f"Score {score:3d} | "
                  f"Len {len(env.snake):3d} | "
                  f"LR {agent.optimizer.param_groups[0]['lr']:.6f} | "
                  f"ICM_η {agent.icm_eta:.3f}")
                  


    # 训练结束，保存最终模型
    agent.save(f"{log_dir}/final.pt")
    csv_file.close()
    writer.close()
    pygame.quit()
    print("Training finished.")

# ------------------------------------------------------------
# 测试/评估函数 (可选)
# ------------------------------------------------------------
def test(cfg: Cfg, model_path: str) -> None:
    """加载训练好的模型并进行测试"""
    env = SnakeEnv(cfg)
    agent = PPOAgent(cfg)
    agent.load(model_path)
    agent.net.eval() # 设置为评估模式
    agent.icm.eval()

    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_W, cfg.WINDOW_H))
    pygame.display.set_caption("Snake PPO+ICM Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    episodes = 100
    total_score = 0
    for ep in range(episodes):
        state = env.reset()
        score = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # 使用确定性策略进行测试
            action, _, _ = agent.select_action(state, deterministic=True)
            state, _, done = env.step(action)
            score = len(env.snake) - 1

            env.render(screen)
            txt = font.render(f"Test Ep:{ep+1}/{episodes}  Score:{score}", True, Color.TEXT)
            screen.blit(txt, (10, 10))
            pygame.display.flip()
            clock.tick(30) # 测试时放慢速度

            if done:
                break
        print(f"Test Episode {ep+1}: Score = {score}")
        total_score += score
    
    avg_score = total_score / episodes
    print(f"Average Score over {episodes} test episodes: {avg_score:.2f}")
    pygame.quit()


# ------------------------------------------------------------
# 入口
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Path to the model file to test")
    args = parser.parse_args()

    cfg = Cfg()
    
    if args.test:
        test(cfg, args.test)
    else:
        train(cfg)

