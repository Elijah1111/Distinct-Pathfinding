#!/usr/bin/env python3

import numpy as np
import perlin as per
import matplotlib.pyplot as plt
from ompl import Algorithm, LightningDB, LightningPlanner, Planner, set_ompl_random_seed, turn_off_logger

EPS = 0.1#Epsilon for max graident allowed

class RRTStar():
    f = 0
    def is_valid(self,pos):
        if (np.array_equal(pos,self.start) or np.array_equal(pos,self.end)):
            return True#start and end should be valid points just in case
        x = int(pos[0])
        y = int(pos[1])
        grad = max(abs(self.gradients[0][x][y]),
                   abs(self.gradients[1][x][y]))
        
        if grad > EPS:
            self.f += 1
            return False
        return True


    def sample(self):#generate a random sample
        while True:
            x = np.random.randint(0,self.SIZE,size=2)
            if self.is_valid(x):
                return x

    def __init__(self,iterations,SIZE,img,start,goal):
        turn_off_logger()
        self.SIZE = SIZE
        self.start = start
        self.end = goal
        self.gradients = np.gradient(img)#find the graident
        self.database = LightningDB(2)
        
        #create a planner object
        self.rrt = Planner([0,0],[SIZE,SIZE],self.is_valid,1000, [1,1], Algorithm.RRTstar)
        
        self.genTrees(iterations)
        
    def genTrees(self,iterations):#generate the random trees
        for _ in range(iterations):
            tmp = self.rrt.solve(self.start,self.sample())#make a tree
            if tmp != None:
                self.database.add_experience(np.array(tmp))#save the trees
        #maybe save the database?
        self.database.save("tmp.db")
    
    def bestPath(self,start,end):#generate the best path
        self.start = start
        self.end = end#reset the goals

        db_again = LightningDB(2)
        db_again.load("tmp.db")#load the paths
        
        lightning = LightningPlanner(db_again,[0,0],[self.SIZE,self.SIZE],
                                     self.is_valid,1000, [1,1], Algorithm.RRTstar)

        self.lightPath = lightning.solve(start,end)#find the best current path
        self.simplified = lightning.solve(start,end,simplify=True)#simplify that path
    


def plot_trajectory(ax, points, color,size=0.5):
    points = np.array(points)
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=size)
    ax.scatter(points[:, 0], points[:, 1], c="black", s=0.3)


if __name__ == "__main__":
    noise = per.Perlin()
    goals = noise.getGoals()
    img = noise.getImage()
    
    rrt = RRTStar(100,48,img,goals[0],goals[1])#make an rrt

    rrt.bestPath(goals[0],goals[1])#find the best path

    fig, ax = plt.subplots()
    plt.imshow(img,cmap="gray")
    paths = rrt.database.get_experienced_paths()#get all the possible trees generated

    for i in range(0,len(paths)):#only render some of the paths
        path = paths[i]
        plot_trajectory(ax, path, "red")
    
    if(rrt.lightPath != None):
        plot_trajectory(ax, rrt.lightPath, "blue",size=1.2)#grab the best path
    if(rrt.simplified != None):
        plot_trajectory(ax, rrt.simplified, "green")#grab the simplified best path
    
    plt.scatter(x=[goals[0][0],goals[1][0]],y=[goals[0][1],goals[1][1]],c="y",s=20)
    plt.title("RRT Graph")
    plt.show()
