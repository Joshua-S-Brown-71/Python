import heapq
class PriorityQueue:
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {} #Map from state to priority
    
    # Insert ]state] into the heap with priority [newPriority] if [state] isn't in the
    # heap or [newPriority] is smaller than the existing priority.
    # return whether the priority queue was updated
    
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False
    
    #Return (state with minimum priority)
    # or None if the priority queue is empty
    def deleteMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left
    
    
