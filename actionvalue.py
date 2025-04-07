import numpy as np
import matplotlib.pyplot as plt
from operator import add

np.random.seed(42)

class Bandit:
    def __init__(self, n=10, numActions=10, epsilon=0.1, steps=2000):
        self.n = n
        self.numActions = numActions
        self.epsilon = epsilon
        self.steps = steps
        self.actionCounts = np.zeros(numActions)
        self.q = np.zeros(numActions)
        self.optimalAction = 0

    def generateActionsValues(self):
        for i in range(self.numActions):
            self.q[i] = np.random.normal(0, 1)
        # return q
    
    def generateNoise(self):
        return np.random.normal(0, 1)
    
    def selectAction(self, epsilon):
        epsilon_choice = np.random.binomial(1, epsilon, 1)

        if epsilon_choice == 1:
            action = np.random.randint(0, self.numActions)
        else:
            action = np.argmax(self.q)
        return action, epsilon_choice

    def getReward(self, action):
        noise = self.generateNoise()
        reward = self.q[action] + noise
        return reward
    
    def updateQ(self, action, reward):
        self.actionCounts[action] += 1
        self.q[action] = (self.q[action] * (self.actionCounts[action] - 1) + reward) / self.actionCounts[action]
        self.optimalAction = np.argmax(self.q)
    
    def simulate(self):
        totalRewards = 0
        self.generateActionsValues()

        # For tracking rewards
        avgRewardsList = []

        # For tracking optimal action percent
        optimalActionList = []
        optimalActionCount = 0

        # optimalAction = np.argmax(self.q)

        for i in range(self.steps):
            # optimalAction = np.argmax(self.q)
            action, epsilon_choice = self.selectAction(self.epsilon)
            reward = self.getReward(action)

            # Update Q values
            if epsilon_choice == 1:
                self.updateQ(action, reward)

            # Update rewrds list
            totalRewards += reward
            avgRewardsList.append(totalRewards/ (i + 1))

            # Update optimal action list
            if action == self.optimalAction:
                optimalActionCount += 1
            optimalActionList.append((optimalActionCount / (i + 1)) * 100)
            

        return avgRewardsList, optimalActionList

steps = 2000
numActions = 10

# epsilon = 0
avgRewardsList1 = np.zeros(steps)
avgOptimalActionList1 = np.zeros(steps)

# epsilon = 0.1
avgRewardsList2 = np.zeros(steps)
avgOptimalActionList2 = np.zeros(steps)

# epsilon = 0.01
avgRewardsList3 = np.zeros(steps)
avgOptimalActionList3 = np.zeros(steps)

for n in range(10):
    bandit1 = Bandit(epsilon=0)
    avgRewards1, optimalAction1 = bandit1.simulate()

    bandit2 = Bandit(epsilon=0.1)
    avgRewards2, optimalAction2 = bandit2.simulate()

    bandit3 = Bandit(epsilon=0.01)
    avgRewards3, optimalAction3 = bandit3.simulate()

    avgRewardsList1 = list(map(add, avgRewardsList1, avgRewards1))
    avgRewardsList2 = list(map(add, avgRewardsList2, avgRewards2))
    avgRewardsList3 = list(map(add, avgRewardsList3, avgRewards3))
    avgOptimalActionList1 = list(map(add, avgOptimalActionList1, optimalAction1))
    avgOptimalActionList2 = list(map(add, avgOptimalActionList2, optimalAction2))
    avgOptimalActionList3 = list(map(add, avgOptimalActionList3, optimalAction3))

avgRewardsList1 = [x / steps for x in avgRewardsList1]
avgRewardsList2 = [x / steps for x in avgRewardsList2]
avgRewardsList3 = [x / steps for x in avgRewardsList3]
avgOptimalActionList1 = [(x / steps)*100 for x in avgOptimalActionList1]
avgOptimalActionList2 = [(x / steps)*100 for x in avgOptimalActionList2]
avgOptimalActionList3 = [(x / steps)*100 for x in avgOptimalActionList3]

plt.figure(figsize=(10, 5))
# plt.ylim(0, 0.1)
plt.plot(avgRewardsList1, label="ε=0")
plt.plot(avgRewardsList3, label="ε=0.01")
plt.plot(avgRewardsList2, label="ε=0.1")
plt.legend()
plt.grid()
plt.title(f"Performance of ε-greedy on 10-armed bandit")
plt.xlabel("Steps")
plt.ylabel("Average Reward")

plt.figure(figsize=(10, 5))
# plt.ylim(0, 0.1)
plt.plot(avgOptimalActionList1, label="ε=0")
plt.plot(avgOptimalActionList3, label="ε=0.01")
plt.plot(avgOptimalActionList2, label="ε=0.1")
plt.legend()
plt.grid()
plt.title(f"Optimal Action % of ε-greedy on 10-armed bandit")
plt.xlabel("Steps")
plt.ylabel("Optimal Action %")

plt.show()







