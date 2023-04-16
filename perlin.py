#!/usr/bin/env python3
import pyfastnoisesimd as fns
import matplotlib.pyplot as plt#just for displaying the image
import numpy as np
class Perlin():
    shape = [128,128]
    seed  = np.random.randint(2**31)
    N_threads = 4#TODO this can be changed to match how mnay cores your cpu has
    def __init__(self):
        self.perlin = fns.Noise(seed=self.seed, numWorkers=self.N_threads)
        self.perlin.frequency = 0.05
        self.perlin.noiseType = fns.NoiseType.Perlin
        self.perlin.fractal.octaves = 8
        self.perlin.fractal.lacunarity = 1.0#does not seem to be implemented
        self.perlin.perturb.perturbType = fns.PerturbType.GradientFractal

    def randomVals(self):
        #self.perlin.frequency = 0.001
        self.perlin.frequency = np.random.rand()*(0.05 -0.01)+0.01
        self.perlin.fractal.gain = np.random.rand()+2.1 #2.1-3
    
    def getImage(self):
        self.randomVals()
        img = self.perlin.genAsGrid(self.shape)
        img = img/2 + 0.5
        return img

    def getGoals(self):
        return (np.random.randint(0,128,(1,2)), np.random.randint(0,128,(1,2)))

if __name__ =="__main__":
    noise = Perlin()
    plt.imshow(noise.getImage(),cmap="gray")
    x,y = noise.getGoals()
    while True:
        plt.imshow(noise.getImage(),cmap="gray")
        print(f"Frequency: {noise.perlin.frequency} Gain: {noise.perlin.fractal.gain}")
        plt.scatter(x=x, y=y, c='r', s=40)
        plt.show()
