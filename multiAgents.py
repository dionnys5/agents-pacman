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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        pelletPositions = successorGameState.getCapsules()
        pelletScore = 0
        if len(pelletPositions) > 0:
          pelletScore = 100 * float(min([manhattanDistance(newPos, pelletPosition) for pelletPosition  in pelletPositions]))

        min_food_dist = float(min([10000000000] + [manhattanDistance(newPos, foodPos) for foodPos  in newFood.asList()]))
        foodScore = 1/min_food_dist
        foodScore = foodScore**2

        ghostScore = 0
        score = (successorGameState.getScore() - currentGameState.getScore()) + foodScore
        for ghost in newGhostStates:  # the impact of ghost surges as distance get close
          dist_ghost = manhattanDistance(ghost.getPosition(), newPos)
          if newScaredTimes[0] > 0:
            ghostScore = (1/dist_ghost) * 100
            return score + pelletScore + ghostScore
          else:
            if dist_ghost <= 2:
              ghostScore = dist_ghost * 1000

        return score - pelletScore - ghostScore

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

  def getAction(self, gameState):
    depth = 0
    return self.get_max(gameState, depth)[1]

  def get_max(self, gameState, depth, agent = 0):
    actions = gameState.getLegalActions(agent)

    if not actions or gameState.isWin() or depth >= self.depth:
      return self.evaluationFunction(gameState), Directions.STOP

    successorCost = float('-inf')
    successorAction = None

    for action in actions:
      successor = gameState.generateSuccessor(agent, action)

      cost = self.get_min(successor, depth, agent + 1)[0]

      if cost > successorCost:
        successorCost = cost
        successorAction = action

    return successorCost, successorAction

  def get_min(self, gameState, depth, agent):
    actions = gameState.getLegalActions(agent)

    if not actions or gameState.isLose() or depth >= self.depth:
      return self.evaluationFunction(gameState), Directions.STOP

    successorCost = float('inf')
    successorAction = None

    for action in actions:
      successor = gameState.generateSuccessor(agent, action)

      cost = 0

      if agent == gameState.getNumAgents() - 1:
          cost = self.get_max(successor, depth + 1)[0]
      else:
          cost = self.get_min(successor, depth, agent + 1)[0]

      if cost < successorCost:
          successorCost = cost
          successorAction = action
    return successorCost, successorAction

class AlphaBetaAgent(MultiAgentSearchAgent):

  def getAction(self, gameState):

    depth = 0
    return self.get_max(gameState, depth)[1]
  # alpha = max best option
  # beta = min best
  def get_max(self, gameState, depth, agent = 0, alpha=float('-inf'), beta=float('inf')):
    actions = gameState.getLegalActions(agent)
    successorCost = alpha
    successorAction = Directions.STOP
    if not actions or gameState.isWin() or depth >= self.depth:
      return self.evaluationFunction(gameState), Directions.STOP

    for action in actions:
      successor = gameState.generateSuccessor(agent, action)

      cost = self.get_min(successor, depth, agent + 1, alpha=successorCost, beta=beta)[0]

      if cost > successorCost:
        successorCost = cost
        successorAction = action
        alpha = successorCost
      if successorCost >= beta: # PODA
        return successorCost, successorAction
  
    return successorCost, successorAction

  def get_min(self, gameState, depth, agent, alpha=float('-inf'), beta=float('inf')):
    actions = gameState.getLegalActions(agent)

    if not actions or gameState.isLose() or depth >= self.depth:
      return self.evaluationFunction(gameState), Directions.STOP

    successorCost = beta
    successorAction = Directions.STOP

    for action in actions:
      successor = gameState.generateSuccessor(agent, action)

      if agent == gameState.getNumAgents() - 1:
        cost = self.get_max(successor, depth + 1, alpha=alpha, beta=beta)[0]
      else:
        cost = self.get_min(successor, depth, agent + 1, alpha=alpha, beta=successorCost)[0]
      if cost < successorCost:
        successorCost = cost
        successorAction = action
        beta = successorCost
      if successorCost <= alpha: # PODA
        return successorCost, successorAction
    return successorCost, successorAction
  

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):

    depth = 0
    return self.get_max(gameState, depth)[1]
  # alpha = max best option
  # beta = min best
  def get_max(self, gameState, depth, agent = 0, alpha=float('-inf'), beta=float('inf')):
    actions = gameState.getLegalActions(agent)
    successorCost = alpha
    successorAction = Directions.STOP
    if not actions or gameState.isWin() or depth >= self.depth:
      return self.evaluationFunction(gameState), Directions.STOP

    for action in actions:
      successor = gameState.generateSuccessor(agent, action)

      cost = self.get_min(successor, depth, agent + 1, alpha=successorCost, beta=beta)[0]

      if cost > successorCost:
        successorCost = cost
        successorAction = action
        alpha = successorCost
      if successorCost >= beta: # PODA
        return successorCost, successorAction
  
    return successorCost, successorAction

  def get_min(self, gameState, depth, agent, alpha=float('-inf'), beta=float('inf')):
    actions = gameState.getLegalActions(agent)

    if not actions or gameState.isLose() or depth >= self.depth:
      return self.evaluationFunction(gameState), Directions.STOP

    successorCost = beta
    successorAction = Directions.STOP

    for action in actions:
      successor = gameState.generateSuccessor(agent, action)

      if agent == gameState.getNumAgents() - 1:
        cost = self.get_max(successor, depth + 1, alpha=alpha, beta=beta)[0]
      else:
        cost = self.get_min(successor, depth, agent + 1, alpha=alpha, beta=successorCost)[0]
      if cost < successorCost:
        successorCost = cost
        successorAction = action
        beta = successorCost
      if successorCost <= alpha: # PODA
        return successorCost, successorAction
    return successorCost, successorAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

