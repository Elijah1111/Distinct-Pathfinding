#!/usr/bin/env python3

import perlin as per
import numpy as np
import time #this could be removed

#This is where the training is to happen

class Train():
    img = np.empty(0)
    episodes = 0#times to use the same image
    trials = 0#trials of episode sets
    epochs = 0#sets of trials
    noise = per.Perlin()
    def __init__(self,episodes=10,trials=100,epochs=3):
        self.episodes = episodes
        self.trials = trials
        self.epochs = epochs
    def train(self):#train the model
        start = time.time()
        for i in range(0,self.epochs):#TODO probally speed these loops up
            print(f"Epoch {i}")
            for j in range(0,self.trials):
                self.img = self.noise.getImage()#make an image
                for e in range(0,self.episodes):
                    x,y = self.noise.getGoals()
                    #DO THE MODEL
                    #Reward
        print(f"{time.time()-start}")

if __name__ == "__main__":
    t = Train()
    t.train()
