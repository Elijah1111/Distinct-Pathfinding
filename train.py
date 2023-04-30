#!/usr/bin/env python3

import perlin as per
import numpy as np
import time #this could be removed
import matplotlib.pyplot as plt
import agent as agt
import random
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder

#This is where the training is to happen
SIZE = 48
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
                for z in range(SIZE):
                    for t in range(SIZE):
                        count+=1
                        tempVar = 0
                        if(count %SIZE == 0):
                            tempVar = (1+(self.img[z][t]**2))**.5
                        else:
                            tempVar = (1+((self.img[z][t]+(self.img[z][t]-1))**2))**.5
                        #print(tempVar)
                        weightedValue[z][t] = float(tempVar)
                        #print(count)
                for e in range(0,self.episodes):
                    print(f"|\t|\tEpisode {e}")
                    #Dijkstra's algorithm
                    startingVal,endingVal = self.noise.getGoals()#get the start and the goal
                    startingx,startingy = startingVal[0],startingVal[1]
                    grid = Grid(matrix=weightedValue) 
                    start = grid.node(startingx,startingy) 
                    end = grid.node(endingVal[0],endingVal[1])  
                    finder = DijkstraFinder(diagonal_movement=DiagonalMovement.never)
                    dPath , runs = finder.find_path(start,end,grid)
                    #End of algorithm

                    self.renderMap(dPath,startingx,startingy,endingVal)

                    #Start the agent to learn on the current environment and path
                    ag = Agent(start=(startingVal[0],startingVal[1]),end = (endingVal[0],endingVal[1]))
                    episodes = 1000
                    ag.Q_Learning(episodes,dPath,start=(startingVal[0],startingVal[1]),end=(endingVal[0],endingVal[1]))
                    ag.plot(episodes)
                    # ag.showValues()  

                    # A* implemented below using import pathfinding (more complex than that but that's the library)
                    grid = Grid(matrix=weightedValue) 
                    start = grid.node(startingx,startingy) 
                    end = grid.node(endingVal[0],endingVal[1])  
                    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
                    aPath , runs = finder.find_path(start,end,grid)
                    print('operations:', runs, 'path length:', len(aPath))
                    print(grid.grid_str(path=aPath, start=start, end=end))   
                    #End of A* 


                    
        print(f"{time.time()-start}")

    def renderMap(self,path,startingx,startingy,endingVal):#render a path along the current map
        # return
        fig = plt.figure(1)#path map
        ax = fig.add_subplot()
        ax.imshow(self.img,cmap="gray")
        xp, yp = zip(*path)#unwrap the path into x and y values
        ax.scatter(xp,yp,color="blue")#plot them
        ax.scatter([startingx,endingVal[0]],[startingy,endingVal[1]],color="red")
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
        val1, val2 = self.state
        current_state = [val1,val2]
        if current_state not in path:
            print("|\t|\t| Not in path           ",end="\r")
            return -.1
        elif self.state == end:
            return 2
        else:
            print("|\t|\t| Made it to path value",end="\r")
            return -.05
        #give the rewards for each state -.2 for on path, +10 for win, -1 for others

    def isEndFunc(self,end):
        #set state to end if win/loss
        if (self.state == end):
            self.isEnd = True
            
    def nxtPosition(self, action):     
        #set the positions from current action - up, down, left, right
        if action == 0:                
            nxtState = (self.state[0] - 1, self.state[1]) #up             
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1]) #down
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1) #left
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
        self.actions = [0,1,2,3]    # up, down, left, right
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
        x = 0
        #iterate through best path for each episode
        while(x < episodes):
            #check if state is end
            if self.isEnd:
                #get current rewrard and add to array for plot
                reward = self.State.getReward(path,end)
                self.rewards += reward
                self.plot_reward.append(self.rewards)
                
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
        
    #plot the reward vs episodes
    def plot(self,episodes):
        plt.plot(self.plot_reward)
        plt.title("Reward Per Episode")
        plt.show()

        
        
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