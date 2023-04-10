#!/usr/bin/env python3
import pyfastnoisesimd as fns
import matplotlib.pyplot as plt#just for displaying the image
import numpy as np
class Perlin():
    shape = [1028,1028]
    seed  = np.random.randint(2**31)
    N_threads = 4#TODO this can be changed to match how mnay cores your cpu has
    def __init__(self):
        self.perlin = fns.Noise(seed=self.seed, numWorkers=self.N_threads)
        self.perlin.frequency = 0.005
        self.perlin.noiseType = fns.NoiseType.Perlin
        self.perlin.fractal.octaves = 8
        self.perlin.fractal.lacunarity = 1.0#does not seem to be implemented
        self.perlin.perturb.perturbType = fns.PerturbType.GradientFractal
        self.randomVals()

    def randomVals(self):
        #self.perlin.frequency = 0.001
        self.perlin.frequency = np.random.rand()*(0.005 - 0.001)+0.001
        self.perlin.fractal.gain = np.random.rand()+2.1 #2.1-3
        self.perlin.fractal.gain = 3 #2.1-3

    
    def getImage(self):
        img = self.perlin.genAsGrid(self.shape)
        img = img/2 + 0.5
        return img

    def getGoals(self):
        return (np.random.randint(0,1028,(1,2)), np.random.randint(0,1028,(1,2)))
if __name__ =="__main__":
    noise = Perlin()
    plt.imshow(noise.getImage(),cmap="gray")
    x,y = noise.getGoals()
    plt.scatter(x=x, y=y, c='r', s=40)
    plt.show()
