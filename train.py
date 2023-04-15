#!/usr/bin/env python3

import perlin as per
import numpy as np
import time #this could be removed

#This is where the training is to happen
SIZE = 128
class Train():
    
    img = np.empty(0)
    episodes = 0#times to use the same image
    trials = 0#trials of episode sets
    epochs = 0#sets of trials
    noise = per.Perlin()
    def __init__(self,episodes=1,trials=1,epochs=1):
        self.episodes = episodes
        self.trials = trials
        self.epochs = epochs
    def train(self):#train the model
        start = time.time()
        for i in range(0,self.epochs):#TODO probably speed these loops up
            print(f"Epoch {i}")
            for j in range(0,self.trials):
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
                        print(tempVar)
                        weightedValue[z][t] = float(tempVar)
                        print(count)
                for e in range(0,self.episodes):
                    startAndEndGoals = self.noise.getGoals()
                    startingVal = startAndEndGoals[0]
                    endingVal = startAndEndGoals[1]
                    startingx,startingy = startingVal[0][0],startingVal[0][1]
                    distmap=np.ones((SIZE,SIZE),dtype=int)*np.Infinity
                    distmap[startingx,startingy]=0
                    originmap=np.ones((SIZE,SIZE),dtype=int)*np.nan
                    visited=np.zeros((SIZE,SIZE),dtype=bool)
                    finished = False
                    x,y=startingx,startingy
                    count=0

                    while not finished:
  # move to x+1,y
                        if x < SIZE-1:
                            if distmap[x+1,y]>self.img[x+1,y]+distmap[x,y] and not visited[x+1,y]:
                                distmap[x+1,y]=self.img[x+1,y]+distmap[x,y]
                                originmap[x+1,y]=np.ravel_multi_index([x,y], (SIZE,SIZE))
  # move to x-1,y
                        if x>0:
                            if distmap[x-1,y]>self.img[x-1,y]+distmap[x,y] and not visited[x-1,y]:
                                distmap[x-1,y]=self.img[x-1,y]+distmap[x,y]
                                originmap[x-1,y]=np.ravel_multi_index([x,y], (SIZE,SIZE))
  # move to x,y+1
                        if y < SIZE-1:
                            if distmap[x,y+1]>self.img[x,y+1]+distmap[x,y] and not visited[x,y+1]:
                                distmap[x,y+1]=self.img[x,y+1]+distmap[x,y]
                                originmap[x,y+1]=np.ravel_multi_index([x,y], (SIZE,SIZE))
  # move to x,y-1
                        if y>0:
                            if distmap[x,y-1]>self.img[x,y-1]+distmap[x,y] and not visited[x,y-1]:
                                distmap[x,y-1]=self.img[x,y-1]+distmap[x,y]
                                originmap[x,y-1]=np.ravel_multi_index([x,y], (SIZE,SIZE))

                        visited[x,y]=True
                        dismaptemp=distmap
                        dismaptemp[np.where(visited)]=np.Infinity
  # now we find the shortest path so far
                        minpost=np.unravel_index(np.argmin(dismaptemp),np.shape(dismaptemp))
                        x,y=minpost[0],minpost[1]
                        if x==SIZE-1 and y==SIZE-1:
                            finished=True
                        count=count+1

#Start backtracking to plot the path  
                    mattemp=self.img.astype(float)
                    x,y=SIZE-1,SIZE-1
                    path=[]
                    mattemp[x,y]=np.nan

                    while x>startingx or y>startingy:
                        path.append([x,y])
                        xxyy=np.unravel_index(int(originmap[x,y]), (SIZE,SIZE))
                        x,y=xxyy[0],xxyy[1]
                        mattemp[x,y]=np.nan
                    path.append([x,y])
                    # finished = False
                    # visited=np.zeros((SIZE,SIZE),dtype=bool)
                    # x,y=np.int(0),np.int(0)
                    # count = 0
                    
                    # while not finished:
                    #     if x < SIZE-1:
                    #         if weightedValue[x+1,y] > 
                    #DO THE MODEL
                    #Initial state will be the graph/method designed for traversing the noise graph with initial position and goal position
                    #Goal state and reward modeling will be based on dikjstra's path
                    #First step will be to get dikjstra's path on the given array
                    #Then we need to create a directed graph based on that path (using distance vectors)
                    #Create a way of traversing the provided image
                    #Then set rewards to be -1 for moving back on itself, 0 for an action that doesn't matter, 1 for taking a step towards the correct path
                    #Reward
        print(f"{time.time()-start}")

if __name__ == "__main__":
    t = Train()
    t.train()
