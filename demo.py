from Agents import AMYG4
from Agents import AINSA
from Agents import ALLIN
from Agents import AIACIMP
from ChefsHatGym.env import ChefsHatEnv

import os
import gym

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]
gameStopCriteria = 3

"""Player Parameters"""
agent1 = AMYG4.AMYG4(name="AMYG4", continueTraining=False, initialEpsilon=1, verbose=False)
agent2 = ALLIN.ALLIN(name="ALLIN", continueTraining=False, initialEpsilon=1, verbose=False)
agent3 = AINSA.AINSA(name="AINSA", continueTraining=False, initialEpsilon=1, verbose=False)
agent4 = AIACIMP.AIACIMP(name="AIACIMP", continueTraining=False, initialEpsilon=1, verbose=False)


agentNames = [agent1.name, agent2.name, agent3.name, agent4.name]
playersAgents = [agent1, agent2, agent3, agent4]

rewards = []
for r in playersAgents:
    rewards.append(r.getReward)

"""Experiment parameters"""
saveDirectory = "./playerClubTesting/"
verbose = False
saveLog = False
saveDataset = False
episodes = 1

"""Setup environment"""
env = gym.make('chefshat-v0') #starting the game Environment
env.startExperiment(rewardFunctions=rewards, gameType=gameType,stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Train Agent"""
for a in range(episodes):

    observations = env.reset()

    while not env.gameFinished:
        currentPlayer = playersAgents[env.currentPlayer]

        observations = env.getObservation()
        action = currentPlayer.getAction(observations)

        info = {"validAction":False}
        while not info["validAction"]:
            nextobs, reward, isMatchOver, info = env.step(action)

        currentPlayer.actionUpdate(observations, nextobs, action, reward, info)

        if isMatchOver:
            for p in playersAgents:
                p.matchUpdate(info)
            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")