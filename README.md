## Chef's Hat Competition Agents
The Chef's Hat Competition Agens provides four implementation agents for the Chef's Hat game.

## Instalation

The instalation can be done via pip:

```python
pip install -r requirements.txt
```

or just run setup.py

```python
python setup.py install
```

or

```python
pip install .
```


## Available Agents

There are four different implementations of Agents, all of them is based on the Karma Chameleon Club

### Dataset

We have used all the available dataset in the Chef's Hat Player Club envinroment to train the agents:

Types | Types | Types | Types | Types
------------ |------------ |------------ | ----------- | ----------- |
	lil_abcd_|lilAbsol| lilAle| lilAna| lilArkady|
         lilAuar| lilBlio1| lilBlio2| lilChu| lilDa48|
         lilDana| lilDJ| lilDomi948|lilEle|lilFael|
         lilGeo|lilIlzy|lilJba|lilLeandro|lilLena|
         lilLordelo|lilMars|lilNathalia|lilNik|lilNilay|
         lilRamsey|lilRaos|lilThecube|lilThuran|lilTisantana|
         lilToran|lilWinne|lilYves|lilYves2

We just concatenate all available dataset in only one new dataset.


### AMYG4 (Agent Mari Yves Gui 4)
For the first agent, we decided to use a custom loss function based on the Hub Loss and the following getRewarded function
```python
    def getReward(self, info, stateBefore, stateAfter):

        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        maxScore = numpy.argmax(info["score"], axis=0)
        beforeScore = self.beforeScore
        beforeInfo = self.beforeInfo

        self.beforeInfo = info
        self.beforeScore = info["score"][0]
        if matchFinished:
            if thisPlayer == 0:
                return 1
            else:
                if beforeScore is not None and info["score"][0] > beforeScore:
                    if maxScore == 0:
                        return 1
                    else:
                        return 0.001

                return -0.001

        state = numpy.concatenate((stateBefore[0:11], stateBefore[11:28]))

        reward = -0.001

        if beforeInfo is not None:
            lastActionPlayers = beforeInfo["lastActionPlayers"]
            actualActionPlayers = info["lastActionPlayers"]

            count = 0

            if len(lastActionPlayers[0]) > 0 and lastActionPlayers[0][0] == 'DISCARD':
                for x, i in enumerate(actualActionPlayers):
                    if x == 0:
                        continue
                    for val in actualActionPlayers[x]:
                        if len(actualActionPlayers[0]) > 0 and actualActionPlayers[0][0] == 'DISCARD' and (val == 'PASS' or val == ''):
                            reward += 0.05
                            count += 1

            if len(actualActionPlayers[0]) > 0 and actualActionPlayers[0][0] != 'PASS' and count == 3:
                return 1

        if beforeInfo is not None:
            if len(lastActionPlayers[0]) > 0 and lastActionPlayers[0][0] == 'DISCARD' and len(
                    actualActionPlayers[0]) > 0 and actualActionPlayers[0][0] == 'DISCARD':
                reward += 0.05

        if reward > 1:
            return 1

        return reward
```

Each agent can be instantiated using:

```python
agent = AMYG4.AMYG4(name="AMYG4", continueTraining=False, initialEpsilon=1, verbose=True)  # training agent
```

### AINSA (Agent It's Not The Same Agent)
For the our second agent implementation we just use the following getRewarded function:
```python
def getReward(self, info, stateBefore, stateAfter):

    thisPlayer = info["thisPlayerPosition"]
    matchFinished = info["thisPlayerFinished"]
    
    if matchFinished:
        if thisPlayer == 0:
            return 1
        else:
            return -0.001
    
    state = numpy.concatenate((stateBefore[0:11], stateBefore[11:28]))
    
    rewardShape = numpy.concatenate([state, info["action"]])
    rewardShape = numpy.expand_dims(numpy.array(rewardShape), 0)
    reward = self.rewardNetwork.predict([rewardShape])[0][0]
    
    if self.beforeInfo is not None:
        lastActionPlayers = self.beforeInfo["lastActionPlayers"]
        actualActionPlayers = info["lastActionPlayers"]
    
        count = 0
    
        if len(lastActionPlayers[0]) > 0 and lastActionPlayers[0][0] == 'DISCARD':
            for x, i in enumerate(actualActionPlayers):
                if x == 0:
                    continue
                for val in actualActionPlayers[x]:
                    if len(actualActionPlayers[0]) > 0 and actualActionPlayers[0][0] == 'DISCARD' and val == 'PASS':
                        reward += 0.5
                        count += 1
    
        if len(lastActionPlayers[0]) > 0 and lastActionPlayers[0][0] != 'PASS' and count == 3:
            return 1
    
    self.beforeInfo = info
    
    if reward > 1:
        return 1
    
    return reward
```

Each agent can be instantiated using:

```python
agent = AINSA.AINSA(name="AINSA", continueTraining=False, initialEpsilon=1, verbose=True)  # training agent
```

### AIACIMP (Agent It's Another Chameleon Implementation)
Third implementation is only based on the the following getRewarded function:
```python
def getReward(self, info, stateBefore, stateAfter):

    thisPlayer = info["thisPlayerPosition"]
    matchFinished = info["thisPlayerFinished"]

    if matchFinished:
        if thisPlayer == 0:
            return 1
        else:
            return -0.001

    state = numpy.concatenate((stateBefore[0:11], stateBefore[11:28]))

    rewardShape = numpy.concatenate([state, info["action"]])
    rewardShape = numpy.expand_dims(numpy.array(rewardShape), 0)
    reward = self.rewardNetwork.predict([rewardShape])[0][0]

    return reward
```

Each agent can be instantiated using:

```python
agent = AIACIMP.AIACIMP(name="AIACIMP", continueTraining=False, initialEpsilon=1, verbose=True)  # training agent
```

### ALLIN (Agent All In)
Finally, the last is rewarded only if win:
```python
def getReward(self, info, stateBefore, stateAfter):

    thisPlayer = info["thisPlayerPosition"]
    matchFinished = info["thisPlayerFinished"]

    if matchFinished:
        if thisPlayer == 0:
            return 1

    return -0.001
```

Each agent can be instantiated using:

```python
agent = ALLIN.ALLIN(name="ALLIN", continueTraining=False, initialEpsilon=1, verbose=True)  # training agent
```

### Initialization Parameters for all Agents:

Parameter | Description |
------------ | -------------
name | name of the agent, must be unique in a game
continueTraining | if the agent will learn during the game. Must have a demonstration if continue learning is True.
demonstrations | must be a npy with a list of ([state, action, possibleActions]). The default value is the dataset foleder inside the agent folder
initialEpsilon| when learning, initial exploration value
saveFolder| folder that the agent will be saved in. The default value is the same folder of the agent
verbose | verbose when learning


## Contact

Yves Galvão - ymg@ecomp.poli.br

- [Linkedin](https://www.linkedin.com/in/yvesgalvao/)
- [Google Scholar](https://scholar.google.com.br/citations?user=5ZmE50AAAAAJ&hl=pt-BR)
