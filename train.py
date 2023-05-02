#!/usr/bin/env python3

import perlin as per
import rrtstar as RRT
import numpy as np
import time 
import matplotlib.pyplot as plt
import agent as agt
import random
import cython
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder

#This is where the training is to happen
SIZE = 48
EPSILON = 3.0
GAMMA = 1.5
class Train():
    
    img = np.empty(0)
    episodes = 0#times to use the same image
    trials = 0#trials of episode sets
    epochs = 0#sets of trials
    noise = per.Perlin()
    def __init__(self,episodes=10,trials=1,epochs=1):
        self.episodes = episodes
        self.trials = trials
        self.epochs = epochs
        
    def train(self):#train the model
        pathCost = 0
        pathCostA = 0
        start = time.time()
        for i in range(0,self.epochs):#TODO probably speed these loops up
            print(f"Epoch {i}")
            for j in range(0,self.trials):
                print(f"|\tTrial {j}")
                print("|\t| Generating Image")
                self.img = self.noise.getImage()#make an image
                count = 0

                weightedValue = np.zeros((SIZE,SIZE))
                weightedValue.astype(float)

                #Convert each cell into a distance vector
                for z in range(SIZE):
                    for t in range(SIZE):
                        count+=1
                        tempVar = 0
                        if(count %SIZE == 0):
                            tempVar = (1+(self.img[z][t]**2))**.5
                        else:
                            tempVar = (1+((self.img[z][t]+(self.img[z][t-1]))**2))**.5
                        #print(tempVar)
                        weightedValue[z][t] = float(tempVar)
                        #print(count)
                goals = self.noise.getGoals()
                self.rrt = RRT.RRTStar(100,SIZE,self.img,goals[0],goals[1])#make an rrt
                for e in range(0,self.episodes):
                    pathCost = 0
                    pathCostA = 0
                    print(f"|\t|\tSet {e}")
                    startingVal,endingVal = self.noise.getGoals()#get the start and the goal
                    startingx,startingy = startingVal[0],startingVal[1]

                    #Using Library Dijkstra Solution
                    dTimeStart = time.time()
                    grid = Grid(matrix=weightedValue)
                    start = grid.node(startingx,startingy) 
                    end = grid.node(endingVal[0],endingVal[1])  
                    finder = DijkstraFinder(diagonal_movement=DiagonalMovement.always)
                    path , runs = finder.find_path(start,end,grid)
                    dTimeEnd = time.time()

                    #Manual Dijkstra Solution
#                     dTimeStart = time.time()
#                     startingVal,endingVal = self.noise.getGoals()#get the start and the goal
#                     startingx,startingy = startingVal[0],startingVal[1]
#                     distmap=np.ones((SIZE,SIZE),dtype=int)*np.Infinity#initilize distance map

#                     distmap[startingx,startingy] = 0#start position is distance of 0

#                     originmap=np.ones((SIZE,SIZE),dtype=int)*np.nan
#                     visited=np.zeros((SIZE,SIZE),dtype=bool)
                    
#                     finished = False
                    
#                     x,y=startingx,startingy
#                     count=0
#                     print("|\t|\t| Starting graph conversion")
#                     while not finished:
#                         # move to x+1,y
#                         if x < SIZE-1:
#                             tmp = weightedValue[x+1,y]+distmap[x,y]
#                             if distmap[x+1,y] > tmp and not visited[x+1,y]:
#                                 distmap[x+1,y]   = tmp
#                                 originmap[x+1,y] = np.ravel_multi_index([x,y], (SIZE,SIZE))
#                         # move to x-1,y
#                         if x>0:
#                             tmp = weightedValue[x-1,y]+distmap[x,y]
#                             if distmap[x-1,y] > tmp and not visited[x-1,y]:
#                                 distmap[x-1,y]   = tmp
#                                 originmap[x-1,y] = np.ravel_multi_index([x,y], (SIZE,SIZE))
#                         # move to x,y+1
#                         if y < SIZE-1:
#                             tmp = weightedValue[x,y+1]+distmap[x,y]
#                             if distmap[x,y+1] > tmp and not visited[x,y+1]:
#                                 distmap[x,y+1]   = tmp
#                                 originmap[x,y+1] = np.ravel_multi_index([x,y], (SIZE,SIZE))
#                         # move to x,y-1
#                         if y>0:
#                             tmp = weightedValue[x,y-1]+distmap[x,y]
#                             if distmap[x,y-1] > tmp and not visited[x,y-1]:
#                                 distmap[x,y-1]   = tmp
#                                 originmap[x,y-1] = np.ravel_multi_index([x,y], (SIZE,SIZE))

#                         visited[x,y]=True# we have now checked adjacent and we can mark as visted

#                         dismaptemp = distmap
#                         dismaptemp[np.where(visited)] = np.Infinity
#   # now we find the shortest path so far
#                         minpost = np.unravel_index(np.argmin(dismaptemp),np.shape(dismaptemp))
#                         x,y=minpost[0],minpost[1]
#                         if x == endingVal[0]-1 and y == endingVal[1]-1:#reached the goal state
#                             finished=True
#                         count=count+1

# #Start backtracking to plot the path  
#                     mattemp = weightedValue.astype(float)
#                     x,y = endingVal[0]-1,endingVal[1]-1
                    
#                     path=[]
#                     mattemp[x,y]=np.nan
#                     while x != startingx or y != startingy:
#                         path.append([x,y])#add to path
#                         xxyy=np.unravel_index(int(originmap[x,y]), (SIZE,SIZE))
#                         x,y=xxyy[0],xxyy[1]#set new position
#                         mattemp[x,y]=np.nan#remove old position
                    
#                     path.append([x,y])
#                     path.insert(0,[endingVal[0],endingVal[1]])
#                     dTimeEnd = time.time()
                    
                    
                    #End of algorithm

                    self.renderMap(path,startingx,startingy,endingVal)
                    
                    #Start the agent to learn on the current environment and path
                    # startTimeModel = time.time()
                    #Uncomment above for total train time

                    ag = Agent(start=startingVal,end = endingVal)
                    episodes = 10
                    path = np.array(path)
                    timeAfterTrained = ag.Q_Learning(episodes,path,start=startingVal,end=endingVal)
                    ag.plot(episodes)
                    # ag.showValues()  

                    # Uses path instead of path taken by model since model is finding optimal path so it reduces calculations
                    for point in path:
                        pathCost += weightedValue[point[0]][point[1]]
        
                    # endTimeModel = time.time() 
                    #Uncomment above for total train time

                    print(f"|\t|\tDijkstra time usage: " + str(dTimeEnd-dTimeStart))
                    print(f"|\t|\tDijkstra path-cost found: " + str(pathCost))
                    print(f"|\t|\tDijkstra total energy consumption: " + str((pathCost*GAMMA)+(dTimeEnd-dTimeStart)*EPSILON))
                    print()

                    print(f"|\t|\tModel train time: " + str(timeAfterTrained))
                    print(f"|\t|\tModel path-cost found: " + str(pathCost))
                    print(f"|\t|\tModel total energy consumption: " + str((pathCost*GAMMA)+timeAfterTrained*EPSILON))
                    
                    
                    print()
                    startTimeA = time.time()
                    # A* implemented below using import pathfinding (more complex than that but that's the library)
                    grid = Grid(matrix=weightedValue) 
                    start = grid.node(startingx,startingy) 
                    end = grid.node(endingVal[0],endingVal[1])  
                    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
                    aPath , runs = finder.find_path(start,end,grid)
                    for pointA in aPath:
                        pathCostA += weightedValue[pointA[0]][pointA[1]]

                    endTimeA = time.time()
                    print(f"|\t|\tA* time usage: " + str(endTimeA-startTimeA))
                    print(f"|\t|\tA* path-cost found: " + str(pathCostA))
                    print(f"|\t|\tA* total energy consumption: " + str((pathCostA*GAMMA)+(endTimeA-startTimeA)*EPSILON))
                    
                    rTime = time.time()
                    self.rrt.bestPath(startingVal,endingVal)#Generate best paths for rrt
                    rTime = time.time() - rTime
                    rcost = self.RRTFind(weightedValue)
                    print()
                    print(f"|\t|\tRRT time usage: " + str(rTime))
                    print(f"|\t|\tRRT path-cost found: " + str(rcost))
                    print(f"|\t|\tRRT total energy consumption: " + str((rcost*GAMMA)+(rTime*EPSILON)))
                    # print('operations:', runs, 'path length:', len(aPath))
                    # print(grid.grid_str(path=aPath, start=start, end=end))   
                    
                   


                    
        print(f"{time.time()-start}")
    

    def RRTFind(self,weightedValue):
        path = self.rrt.simplified
        start = path[0]
        cost = 0.0
        for i in range(1,len(path)):
            goal = path[i]
            dumb = set()
            nodes = []
            for a in range(0,SIZE+1):
                a /= SIZE
                #lerp
                tmp = (1-a)*start + a*goal
                tmp = tmp.astype(int)
                tmp = tuple(tmp)
                if tmp not in dumb:
                    nodes.append(tmp)
                    dumb.add(tmp)
            prevNode = nodes[0]
            for node in nodes:
                prevVal = weightedValue[prevNode[0],prevNode[1]]
                tmp = weightedValue[node[0],node[1]]
                prevNode = node
                cost += tmp
        return cost


                
            
    def renderMap(self,path,startingx,startingy,endingVal):#render a path along the current map
        # return
        fig = plt.figure(1)#path map
        ax = fig.add_subplot()
        ax.imshow(self.img,cmap="gray")
        xp, yp = zip(*path)#unwrap the path into x and y values
        ax.scatter(xp,yp,color="blue")#plot them
        ax.scatter([startingx,endingVal[0]],[startingy,endingVal[1]],color="red")
        plt.title("Height Map")
        plt.pause(1)
        
        '''
        ax.imshow(self.img,cmap="gray")
        ax.scatter([startingx,endingVal[0]],[startingy,endingVal[1]],color="red")
        plt.pause(2)
        '''
        
class State:
    def __init__(self, start,end):
        #initalise the state to start and end to false
        self.state = start
        self.isEnd = False    
        self.end = end 
        self.start = start  

    def getReward(self,path,end):
        state = self.state
        dist = np.linalg.norm(state - self.end)
        if state not in path:
            #print("|\t|\t| Not in path           ",end="\r")
            return -1
        elif np.array_equal(state,end):
            return 2
        else:
            #print("|\t|\t| Made it to path value",end="\r")
            return -.05
        #give the rewards for each state -.2 for on path, +10 for win, -1 for others

    def isEndFunc(self,end):
        #set state to end if win/loss
        if np.array_equal(self.state,end):
            self.isEnd = True
            
    def nxtPosition(self, action):     
        #set the positions from current action - up, down, left, right
        if action == 0:                
            nxtState = (self.state[0] - 1, self.state[1]) #up             
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1]) #down
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1) #left
        elif action == 3:
            nxtState = (self.state[0]-1,self.state[1]+1) #Diagonal Right Up
        elif action == 4:
            nxtState = (self.state[0]-1,self.state[1]-1) #Diagonal Left Up
        elif action == 5:
            nxtState = (self.state[0]+1,self.state[1]-1) #Diagonal Left Down
        elif action == 6:
            nxtState = (self.state[0]+1,self.state[1]+1) #Diagonal right Down
        else:
            nxtState = (self.state[0], self.state[1] + 1) #right


        #check if next state is possible
        if (nxtState[0] >= 0) and (nxtState[0] < SIZE):
            if (nxtState[1] >= 0) and (nxtState[1] < SIZE):    
                    #if possible change to next state                
                    return nxtState 
        #Return current state if outside grid     
        return self.state 



        
#class agent to implement reinforcement learning through grid  
class Agent:

    def __init__(self,start,end):
        #inialise states and actions 
        self.states = []
        self.actions = [0,1,2,3,4,5,6,7]    # up, down, left, right
        self.State = State(start,end)
        #set the learning and greedy values
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd

        # array to retain reward values for plot
        self.plot_reward = []
        
        #initalise Q values as a dictionary for current and new
        self.Q = {}
        self.new_Q = {}
        #initalise rewards to 0
        self.rewards = 0
        
        #initalise all Q values across the board to 0, print these values
        for i in range(SIZE):
            for j in range(SIZE):
                for k in range(len(self.actions)):
                    self.Q[(i, j, k)] =0
                    self.new_Q[(i, j, k)] = 0
        
        #print(self.Q)
        
    

    #method to choose action with Epsilon greedy policy, and move to next state
    def Action(self):
        #random value vs epsilon
        rnd = random.random()
        #set arbitraty low value to compare with Q values to find max
        mx_nxt_reward =-10
        action = None
        
        #9/10 find max Q value over actions 
        if(rnd >self.epsilon) :
            #iterate through actions, find Q  value and choose best 
            action = 0
            for k in self.actions:
                
                i,j = self.State.state
                
                nxt_reward = self.Q[(i,j, k)]
                
                if nxt_reward >= mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
                    
        #else choose random action
        else:
            action = np.random.choice(self.actions)
        
        #select the next state based on action chosen
        position = self.State.nxtPosition(action)
        return position,action
    
    
    #Q-learning Algorithm
    def Q_Learning(self,episodes,path,start,end):
        print("|\t|\t| Starting Learning")
        lastTimeStart = 0
        #iterate through best path for each episode
        x = 0
        count = 0
        print(f"|\t|\t|\t| Episode {x}")
        while(x < episodes):
            count += 1
            if(x == episodes-1):
                lastTimeStart = time.time()
            #check if state is end
            if self.isEnd or count > len(path)*10:#reached end or timed out
                if self.isEnd:
                    reward = 1
                else:
                    reward = 0
                #get current reward and add to array for plot
                self.rewards += reward
                self.plot_reward.append(self.rewards)
                #print(f"|\t|\t|\t|\t| Reward {self.rewards}")
                
                #get state, assign reward to each Q_value in state
                i,j = self.State.state
                for a in self.actions:
                    self.new_Q[(i,j,a)] = round(reward,3)
                    
                #reset state
                self.State = State(start,end)
                self.isEnd = self.State.isEnd
                
                #set rewards to zero and iterate to next episode
                self.rewards = 0
                x+=1
                count = 0
                if(x%100 == 0):
                    print(f"|\t|\t|\t| Episode {x}")
            else:
                #set to arbitrary low value to compare net state actions
                mx_nxt_value = -10
                #get current state, next state, action and current reward
                next_state, action = self.Action()
                i,j = self.State.state
                reward = self.State.getReward(path,end)
                #add reward to rewards for plot
                self.rewards +=reward
                
                #iterate through actions to find max Q value for action based on next state action
                for a in self.actions:
                    nxtStateAction = (next_state[0], next_state[1], a)
                    q_value = (1-self.alpha)*self.Q[(i,j,action)] + self.alpha*(reward + self.gamma*self.Q[nxtStateAction])
                
                    #find largest Q value
                    if q_value >= mx_nxt_value:
                        mx_nxt_value = q_value
                
                #next state is now current state, check if end state
                self.State = State(start=next_state,end=end)
                self.State.isEndFunc(end)
                self.isEnd = self.State.isEnd
                
                #update Q values with max Q value for next state
                self.new_Q[(i,j,action)] = round(mx_nxt_value,3)
            
            #copy new Q values to Q table
            self.Q = self.new_Q.copy()
        lastTimeEnd = time.time()
        return lastTimeEnd-lastTimeStart
        
    #plot the reward vs episodes
    def plot(self,episodes):
        fig = plt.figure(2)
        plt.clf()
        ax = fig.add_subplot()
        ax.plot(self.plot_reward)
        plt.title("Reward Per Episode")
        plt.pause(1)

        
        
    #iterate through the board and find largest Q value in each, print output
    def showValues(self):
        for i in range(0, SIZE):
            print('-----------------------------------------------')
            out = '| '
            for j in range(0, SIZE):
                mx_nxt_value = -10
                for a in self.actions:
                    nxt_value = self.Q[(i,j,a)]
                    if nxt_value >= mx_nxt_value:
                        mx_nxt_value = nxt_value
                out += str(mx_nxt_value).ljust(6) + ' | '
            print(out)
        print('-----------------------------------------------')

if __name__ == "__main__":
    t = Train()
    t.train()
