class HalfingGame(object):
    def __init__(self,N):
        self.N=N
    def startState(self):
        return(+1,self.N)
       
    def isEnd(self,state):
        player,number=state
        return number == 0
       
    def utility(self,state):
        player,number=state
        assert number == 0 
        return player * float('inf')
    def actions(self,state):
        return['-','/']
    def player(self,state):
        player,number=state
        return player
    def succ(self,state,action):
        player,number=state
        if action == '-':
           return(-player,number-1)
        elif action =='/':
           return(-player,number//2)
def humanPolicy(game,state):
    while True:
        action = input('Input Action: ')
        if action in game.actions(state):
            return action