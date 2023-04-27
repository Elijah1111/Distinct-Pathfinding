#!/usr/bin/env python3

import perlin as per
import numpy as np
import time #this could be removed
import matplotlib.pyplot as plt
import agent as agt

#This is where the training is to happen
SIZE = 128
class Train():
    
    img = np.empty(0)
    episodes = 0#times to use the same image
    trials = 0#trials of episode sets
    epochs = 0#sets of trials
    noise = per.Perlin()
    def __init__(self,episodes=10,trials=1,epochs=2):
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
                    startingVal,endingVal = self.noise.getGoals()#get the start and the goal
                    startingx,startingy = startingVal[0],startingVal[1]
                    distmap=np.ones((SIZE,SIZE),dtype=int)*np.Infinity#initilize distance map

                    distmap[startingx,startingy] = 0#start position is distance of 0

                    originmap=np.ones((SIZE,SIZE),dtype=int)*np.nan
                    visited=np.zeros((SIZE,SIZE),dtype=bool)
                    
                    finished = False
                    
                    x,y=startingx,startingy
                    count=0
                    print("|\t|\t| Starting graph conversion")
                    while not finished:
                        # move to x+1,y
                        if x < SIZE-1:
                            tmp = weightedValue[x+1,y]+distmap[x,y]
                            if distmap[x+1,y] > tmp and not visited[x+1,y]:
                                distmap[x+1,y]   = tmp
                                originmap[x+1,y] = np.ravel_multi_index([x,y], (SIZE,SIZE))
                        # move to x-1,y
                        if x>0:
                            tmp = weightedValue[x-1,y]+distmap[x,y]
                            if distmap[x-1,y] > tmp and not visited[x-1,y]:
                                distmap[x-1,y]   = tmp
                                originmap[x-1,y] = np.ravel_multi_index([x,y], (SIZE,SIZE))
                        # move to x,y+1
                        if y < SIZE-1:
                            tmp = weightedValue[x,y+1]+distmap[x,y]
                            if distmap[x,y+1] > tmp and not visited[x,y+1]:
                                distmap[x,y+1]   = tmp
                                originmap[x,y+1] = np.ravel_multi_index([x,y], (SIZE,SIZE))
                        # move to x,y-1
                        if y>0:
                            tmp = weightedValue[x,y-1]+distmap[x,y]
                            if distmap[x,y-1] > tmp and not visited[x,y-1]:
                                distmap[x,y-1]   = tmp
                                originmap[x,y-1] = np.ravel_multi_index([x,y], (SIZE,SIZE))

                        visited[x,y]=True# we have now checked adjacent and we can mark as visted

                        dismaptemp = distmap
                        dismaptemp[np.where(visited)] = np.Infinity
  # now we find the shortest path so far
                        minpost = np.unravel_index(np.argmin(dismaptemp),np.shape(dismaptemp))
                        x,y=minpost[0],minpost[1]
                        if x == endingVal[0]-1 and y == endingVal[1]-1:#reached the goal state
                            finished=True
                        count=count+1

#Start backtracking to plot the path  
                    mattemp = weightedValue.astype(float)
                    x,y = endingVal[0]-1,endingVal[1]-1
                    
                    path=[]
                    mattemp[x,y]=np.nan
                    while x != startingx or y != startingy:
                        path.append([x,y])#add to path
                        xxyy=np.unravel_index(int(originmap[x,y]), (SIZE,SIZE))
                        x,y=xxyy[0],xxyy[1]#set new position
                        mattemp[x,y]=np.nan#remove old position
                    
                    path.append([x,y])
                    
                    self.renderMap(path,startingx,startingy,endingVal)
                    rewardMap = weightedValue.copy()
                    for row in range(SIZE):
                        for col in range(SIZE):
                            if rewardMap[row][col] not in path:
                                rewardMap[row][col] *= -1
                            else:
                                print("Path Value")
                    print(rewardMap)                   
    #                 env = agt.GridWorld(128, 128)
    #                 env.render()
    # # model hyperparameters
    #                 ALPHA = 0.1
    #                 GAMMA = 1.0
    #                 EPS = 1.0

    #                 Q = {}
    #                 for state in env.stateSpacePlus:
    #                     for action in env.possibleActions:
    #                         Q[state, action] = 0

    #                 numGames = 10000
    #                 totalRewards = np.zeros(self.episodes)
    #                 for i in range(self.epis):
    #                     if i % 5000 == 0:
    #                         print('starting game ', i)
    #                     done = False
    #                     epRewards = 0
    #                     observation = env.reset()
    #                     while not done:
    #                         rand = np.random.random()
    #                         action = agt.maxAction(Q,observation, env.possibleActions) if rand < (1-EPS) \
    #                                                 else env.actionSpaceSample()
    #                         observation_, reward, done, info = env.step(action)
    #                         epRewards += reward

    #                         action_ = agt.maxAction(Q, observation_, env.possibleActions)
    #                         Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
    #                                     GAMMA*Q[observation_,action_] - Q[observation,action])
    #                         observation = observation_
    #                     if EPS - 2 / numGames > 0:
    #                         EPS -= 2 / numGames
    #                     else:
    #                         EPS = 0
    #                     totalRewards[i] = epRewards

    #                 plt.plot(totalRewards)
    #                 plt.show()
                    #Initial state will be the graph/method designed for traversing the noise graph with initial position and goal position 
                    #Goal state and reward modeling will be based on dikjstra's path x
                    #First step will be to get dikjstra's path on the given array x
                    #Then we need to create a directed graph based on that path (using distance vectors) x
                    #Create a way of traversing the provided image x
                    #Then set rewards to be -1 for moving back on itself, 0 for an action that doesn't matter, 1 for taking a step towards the correct path
                    #Reward
        print(f"{time.time()-start}")

    def renderMap(self,path,startingx,startingy,endingVal):#render a path along the current map
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
        

if __name__ == "__main__":
    t = Train()
    t.train()
