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
# @since 2021/4/22 14:35
# @author Zhang, Chen (chansonzhang)
# @email ZhangChen.Shaanxi@gmail.com

import random
class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        """
        :param epsilon:
        :param counts:
        :param values:
        """
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
    def select_arm(self):
        if random.random() > self.epsilon:
            return self.values.index( max(self.values) )
        else:
            # 随机返回 self.values 中的一个
            return random.randrange(len(self.values))
    def reward(self):
        return 1 if random.random() > 0.5 else 0
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1)  * value + reward ) / float(n)
        self.values[chosen_arm] = round(new_value,4)
n_arms = 10
algo = EpsilonGreedy(0.1, [], [])
algo.initialize(n_arms)
for t in range(100):
    chosen_arm = algo.select_arm()
    reward = algo.reward()
    algo.update(chosen_arm, reward)

print(algo.counts)
print(algo.values)

