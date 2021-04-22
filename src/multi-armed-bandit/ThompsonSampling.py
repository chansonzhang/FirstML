import numpy as np
import random
def ThompsonSampling(wins, trials):
    pbeta = [0] * N
    for i in range(0, len(trials)):
        pbeta[i] = np.random.beta(wins[i] + 1, trials[i] - wins[i] + 1)
    choice = np.argmax(pbeta)
    trials[choice] += 1
    if random.random() > 0.5:
        wins[choice] += 1
T = 10000  # 实验次数
N = 10  # 类别个数
# 臂的选择总次数
trials = np.array([0] * N )
# 臂的收益
wins = np.array([0] * N )
for i in range(0, T):
    ThompsonSampling(wins, trials)
print(trials)
print(wins)
print(wins/trials)