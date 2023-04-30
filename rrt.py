#!/usr/bin/env python3

import numpy as np
import perlin as per
import matplotlib.pyplot as plt
from ompl import Algorithm, LightningDB, LightningPlanner, Planner, set_ompl_random_seed


class RRT():
    def is_valid(self,x):
        return True#TODO take a graident of the surrounding
    def sample(self,SIZE):
        while True:
            x = np.random.randint(0,SIZE,size=2)
            if self.is_valid(x):
                return x

    def __init__(self,iterations,SIZE,img,start,goal):
        self.img = img      
        self.database = LightningDB(2)
        
        #create a planner object
        self.rrt = Planner([0,0],[SIZE,SIZE],self.is_valid,1000, [1,1], Algorithm.RRTConnect)
        
        self.genTrees(SIZE,iterations)
        self.bestPath(SIZE,start,goal)
        


    def genTrees(self,SIZE,iterations):#generate the random trees
        for _ in range(iterations):
            tmp = self.rrt.solve(self.sample(SIZE),self.sample(SIZE))#make a tree
            self.database.add_experience(np.array(tmp))#save the trees
        #maybe save the database?
        self.database.save("tmp.db")
    
    def bestPath(self,SIZE,start,end):
        db_again = LightningDB(2)
        db_again.load("tmp.db")

        lightning = LightningPlanner(db_again,[0,0],[SIZE,SIZE],
                                     self.is_valid,1000, [1,1], Algorithm.RRTConnect)
        self.lightPath = lightning.solve(start,end)#find the best current path
        self.simplified = lightning.solve(start,end,simplify=True)#simplify that path
    


def plot_trajectory(ax, points, color):
    points = np.array(points)
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=0.5)
    ax.scatter(points[:, 0], points[:, 1], c="black", s=0.3)


if __name__ == "__main__":
    noise = per.Perlin()
    goals = noise.getGoals()
    img = noise.getImage()
    
    rrt = RRT(1000,128,img,goals[0],goals[1])#make an rrt

    fig, ax = plt.subplots()
    plt.imshow(img,cmap="gray")
    paths = rrt.database.get_experienced_paths()

    for i in range(0,len(paths),100):#only render some of the paths
        path = paths[i]
        plot_trajectory(ax, path, "red")
    
    if(rrt.lightPath != None):
        plot_trajectory(ax, rrt.lightPath, "blue")
    if(rrt.simplified != None):
        plot_trajectory(ax, rrt.simplified, "green")
    
    plt.scatter(x=[goals[0][0],goals[1][0]],y=[goals[0][1],goals[1][1]],c="y",s=20)
    plt.show()
