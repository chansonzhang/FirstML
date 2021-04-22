# -*- coding: utf-8 -*-

# Copyright 2021 Zhang, Chen. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# @since 2021/4/22 14:01
# @author Zhang, Chen (chansonzhang)
# @email ZhangChen.Shaanxi@gmail.com


import numpy as np

# 定义 T = 1000 个用户，即总共进行1000次实现
T = 1000
# 定义 N = 10 个标签，即 N 个 物品
N = 10
# 保证结果可复现，设置随机数种子
np.random.seed(888)
# 每个物品的累积点击率（理论概率）
true_rewards = np.random.uniform(low=0, high=1, size=N)
# true_rewards = np.array([0.5] * N)
# 每个物品的当前点击率
now_rewards = np.zeros(N)
# 每个物品的点击次数
chosen_count = np.zeros(N)
total_reward = 0


# 计算ucb的置信区间宽度
def calculate_delta(T, item):
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(T) / chosen_count[item])


# 计算UCB

def ucb(t, N):
    """
    :param t: current trail time
    :param N: number of items
    :return:
    """
    # ucb得分
    upper_bound_probs = [now_rewards[i] + calculate_delta(t, i) for i in range(N)]
    item = np.argmax(upper_bound_probs)
    # 模拟伯努利收益
    # reward = sum(np.random.binomial(n =1, p = true_rewards[item], size=20000)==1 ) / 20000
    reward = np.random.binomial(n=1, p=true_rewards[item])
    return item, reward


for t in range(1, T + 1):
    # 为第 t个用户推荐一个物品
    item, reward = ucb(t, N)
    # print("item is %s, reward is %s" % (item, reward))
    # 一共有多少用户接受了推荐的物品
    total_reward += reward
    chosen_count[item] += 1
    # 更新物品的当前点击率
    now_rewards[item] = (now_rewards[item] * (t - 1) + reward) / t
    # print("更新后的物品点击率为：%s" % (now_rewards[item]))
    # 输出当前点击率 / 累积点击率
    # print("当前点击率为: %s" % now_rewards)
    # print("累积点击率为: %s" % true_rewards)
    diff = np.subtract(true_rewards, now_rewards)
    print(diff[0])
    print(total_reward)
