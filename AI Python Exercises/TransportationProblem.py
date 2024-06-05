from PriorityQueueLib import PriorityQueue #1
class Transportation(object):
    #N number of city blocks
    # weights = weights of different actions #2
    def __init__(self, N, weights):  #3
        self.N = N
        self.weights = weights #4
    def startState(self):
        return 1
    def isEnd(self, state):
        return state == self.N
    def succAndCost(self, state):
        result = []
        if state+1<=self.N:
            result.append(('walk', state+1, self.weights['walk'])) #5
        if state*2<=self.N:
            result.append(('bus', state*2, self.weights['bus'])) #6
        return result
    
def backTrackingSearch(problem):
    # Best solution found so far
    best = {'cost': float('+inf'), 'history': None}
    
    def recurse(state, history, totalCost):
        # At state, having undergone history, accumulated totalCost
        # Explore the rest of the subtree under state
        if problem.isEnd(state):
            # Update the best solution so far
            if totalCost<best['cost']:
                best['cost'] = totalCost
                best['history'] = history
            return
        # Recurse on children
        for action, newState, cost in problem.succAndCost(state):
            recurse(newState, history+[(action, newState, cost)], totalCost+cost)
    recurse(problem.startState(), history=[], totalCost=0)
    return (best['cost'], best['history'])
    
def printSolution(solution):
    totalCost, history = solution
    print(f'totalCost: {totalCost}')
    if history is None:                 #7
        print("No solution found.")
    else:                               #8
        for item in history:
            print(item)
def dynamicProgramming(problem):
    cache = {} #state -> futureCost, action, newState, cost
    
    def futureCost(state):
        #Base case
        if problem.isEnd(state):
            return 0
        if state in cache: #exponential saving
            return cache[state][0]
        result = min((cost+futureCost(newState), action, newState, cost) \
                     for action, newState, cost in problem.succAndCost(state))
        cache[state] = result
        return result[0]
    
    state = problem.startState()
    totalCost = futureCost(state)
    history = []
    while not problem.isEnd(state):
        _, action, newState, cost = cache[state]
        history.append((action, newState, cost))
        state = newState
    
    return futureCost(problem.startState()), history
    
def uniformCostSearch(problem):
    frontier = priorityQueue()
    frontier.update(problem.startState(), 0)
    while True:
        state, pastCost = frontier.removeMin()
        if problem.isEnd(state):
            return (pastCost, [])
        
        for action, newState, cost in problem.succAndCost(state):
            frontier.update(newState, pastCost+cost)
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

