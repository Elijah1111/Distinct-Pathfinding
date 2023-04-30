#!/usr/bin/env python3
import pyfastnoisesimd as fns
import matplotlib.pyplot as plt#just for displaying the image
import numpy as np
SIZE = 48
class Perlin():# a simple class to handle the noise generation
    
    shape = [SIZE,SIZE]#the output shape of the image
    seed  = np.random.randint(2**31)#a random starting seed
    N_threads = 8#TODO this can be changed to match how mnay cores your cpu has

    def __init__(self):
        self.perlin = fns.Noise(seed=self.seed, numWorkers=self.N_threads)#create a noise instance
        self.perlin.frequency = 0.05
        self.perlin.noiseType = fns.NoiseType.Perlin#srt type to perlin, could also be simplex
        self.perlin.fractal.octaves = 8 #how many sub-generations are created for each image
        self.perlin.fractal.lacunarity = 1.0#does not seem to be implemented
        self.perlin.perturb.perturbType = fns.PerturbType.GradientFractal#good middle ground for making things wavey and not straight

    def randomVals(self):#randomize the generation values
        self.perlin.frequency = np.random.rand()*(0.05 -0.01)+0.01
        self.perlin.fractal.gain = np.random.rand()+2.1 #2.1-3
    
    def getImage(self):#generate a new image
        self.randomVals()#
        img = self.perlin.genAsGrid(self.shape)#generate the image
        img = img/2 + 0.5#make it 0-1 instead of -1 - 1
        return img

    def getGoals(self):#generate 2 points
        return np.random.randint(0,SIZE,(2)), np.random.randint(0,SIZE,(2))


if __name__ =="__main__":
    noise = Perlin()#invoke the noise
    plt.imshow(noise.getImage(),cmap="gray")
    while True:#just keep showing images
        plt.imshow(noise.getImage(),cmap="gray")
        print(f"Frequency: {noise.perlin.frequency} Gain: {noise.perlin.fractal.gain}")
        x,y = noise.getGoals()
        plt.scatter(x=x, y=y, c='r', s=40)
        plt.show()
