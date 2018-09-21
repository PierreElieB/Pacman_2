# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0] # Pick randomly among the best

        "Add more of your code here if you want to"


        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFood_list = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
#        return successorGameState.getScore()

        def manhattanDistance(x,y):
            return abs(x[0]-y[0])+abs(x[1]-y[1])

        res1 = - min([manhattanDistance(newPos, food) for food in newFood_list]+[5000])
        if(newFood_list == []):
            res1 = 5000.
        #res1 = -sum([manhattanDistance(newPos, food) for food in newFood])
        res2 = sum([min(manhattanDistance(newPos, ghost.getPosition()), 6.) for ghost in newGhostStates])
        res3 = 2*successorGameState.getScore()
        res = res3 + res2


        if(res2<4):
            return(res1+res3)
        else:
            return(res1+res3+50)
        #res = res1+res2+res3
        return(res)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        _,action = self.value(gameState, 0, 1)
        return action

    def value(self, gameState, agent_index, depth):
        """
        A second try for this function.
        """
        #print("depth: "+str(self.depth))


        if(gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState), None

        n = gameState.getNumAgents()

        if(agent_index > 0):
            return self.min_value(gameState, agent_index, depth)

        return self.max_value(gameState, agent_index, depth)


    def min_value(self, gameState, agent_index, depth):
        mini = 99999.
        opt_action = None
        n = gameState.getNumAgents()

        stop = False
        next_depth = depth

        if(agent_index==n-1):
            next_depth+=1

            if(next_depth>self.depth):
                stop = True

        actions = gameState.getLegalActions(agent_index)

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)

            if(stop):
                next_score = self.evaluationFunction(successor)
            else:
                next_score, _ = self.value(successor, (agent_index+1)%n, next_depth)

            if(next_score < mini):
                    mini, opt_action = next_score, action

        return mini, opt_action


    def max_value(self, gameState, agent_index, depth):
        maxi = -99999.
        opt_action = None
        n = gameState.getNumAgents()

        stop = False
        next_depth = depth

        if(agent_index==n-1):
            next_depth+=1
            if(next_depth>self.depth):
                stop = True

        actions = gameState.getLegalActions(agent_index)

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            next_score = 0.

            if(stop):
                next_score = self.evaluationFunction(successor)
            else:
                next_score, _ = self.value(successor, (agent_index+1)%n, next_depth)

            if(next_score > maxi):
                maxi, opt_action = next_score, action

        return maxi, opt_action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        alpha = -99999.
        beta = 99999.
        _,action,_,_ = self.value(gameState, 0, 1, alpha, beta)
        return action

    def value(self, gameState, agent_index, depth, alpha, beta):


        if(gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState), None, alpha, beta

        n = gameState.getNumAgents()

        if(agent_index > 0):
            return self.min_value(gameState, agent_index, depth, alpha, beta)

        return self.max_value(gameState, agent_index, depth, alpha, beta)


    def min_value(self, gameState, agent_index, depth, alpha, beta):
        mini = 99999.
        opt_action = None
        n = gameState.getNumAgents()

        stop = False
        next_depth = depth

        if(agent_index==n-1):
            next_depth+=1

            if(next_depth>self.depth):
                stop = True

        actions = gameState.getLegalActions(agent_index)
        alpha2 = alpha

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)

            if(stop):
                next_score = self.evaluationFunction(successor)
            else:
                next_score, _, alpha2, beta = self.value(successor, (agent_index+1)%n, next_depth, alpha, beta)

            if(next_score < mini):
                    mini, opt_action = next_score, action

            if(agent_index > 0):
                if(mini < alpha2):
                    return mini, opt_action, alpha, beta
                beta = min(beta, mini)

        return mini, opt_action, alpha, beta


    def max_value(self, gameState, agent_index, depth, alpha, beta):
        maxi = -99999.
        opt_action = None
        n = gameState.getNumAgents()

        stop = False
        next_depth = depth

        if(agent_index==n-1):
            next_depth+=1
            if(next_depth>self.depth):
                stop = True

        actions = gameState.getLegalActions(agent_index)

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            next_score = 0.

            if(stop):
                next_score = self.evaluationFunction(successor)
            else:
                next_score, _, alpha, beta2 = self.value(successor, (agent_index+1)%n, next_depth, alpha, beta)

            if(next_score > maxi):
                maxi, opt_action = next_score, action

            if(maxi > beta2):
                return maxi,  opt_action, alpha, beta

            alpha = max(alpha, maxi)

        return maxi, opt_action, alpha, beta


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        _,action = self.value(gameState, 0, 1)
        return action

    def value(self, gameState, agent_index, depth):
        """
        A second try for this function.
        """
        #print("depth: "+str(self.depth))


        if(gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState), None

        n = gameState.getNumAgents()

        if(agent_index > 0):
            return self.min_value(gameState, agent_index, depth)

        return self.max_value(gameState, agent_index, depth)


    def min_value(self, gameState, agent_index, depth):
        mini = 99999.
        opt_action = None
        n = gameState.getNumAgents()

        stop = False
        next_depth = depth

        if(agent_index==n-1):
            next_depth+=1

            if(next_depth>self.depth):
                stop = True

        actions = gameState.getLegalActions(agent_index)
        N = len(actions)
        res = 0.

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)

            if(stop):
                next_score = self.evaluationFunction(successor)
            else:
                next_score, _ = self.value(successor, (agent_index+1)%n, next_depth)
            res+=next_score/float(N)

        return res, "Random"


    def max_value(self, gameState, agent_index, depth):
        maxi = -99999.
        opt_action = None
        n = gameState.getNumAgents()

        stop = False
        next_depth = depth

        if(agent_index==n-1):
            next_depth+=1
            if(next_depth>self.depth):
                stop = True

        actions = gameState.getLegalActions(agent_index)

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            next_score = 0.

            if(stop):
                next_score = self.evaluationFunction(successor)
            else:
                next_score, _ = self.value(successor, (agent_index+1)%n, next_depth)

            if(next_score > maxi):
                maxi, opt_action = next_score, action

        return maxi, opt_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newFood_list = newFood.asList()
    newGhostStates = successorGameState.getGhostStates()

    def manhattanDistance(x,y):
        return abs(x[0]-y[0])+abs(x[1]-y[1])


    nfood = len(newFood_list)
    res1 = -sum([manhattanDistance(newPos, food) for food in newFood_list])
    if(newFood_list == []):
        res1 = 5000.

    res2 = sum([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    res3 = 10*successorGameState.getScore()
    res = res3 + res2


    if(res2<3):
        return(res1+res3-5*res1)
    else:
        return(3.*res1+res3+500+res2-20*nfood)

    return(res)



# Abbreviation
better = betterEvaluationFunction
