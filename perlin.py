#!/usr/bin/env python3
import random
import numpy as np
class Perlin():
    gradients = np.array(0)#graident vectors
    def __init__(self,n=256):
        self.makeGraidents(n)
    
    def makeGraidents(self,n):#TODO this may need wrapped in a try case very rarely it might fail with a division by 0
        graidents = np.random.random((n,n,3))*2 -1;#make a new matrix of graidents
        normals = np.sqrt(np.einsum('...i,...i', graidents, graidents));#super fast, unsafe if magnitude is too large, should be fine because max would be sqrt(3)
        self.graidents = graidents/normals[:,:,np.newaxis]#normalize the graidents

    
#make some noise
if __name__ == "__main__":
    noise = Perlin(1000)

'''
#works but very slow at large sizes
import matplotlib.pyplot as plt#just for displaying the image
from perlin_noise import PerlinNoise

noise1 = PerlinNoise(octaves=3)
noise2 = PerlinNoise(octaves=6)
noise3 = PerlinNoise(octaves=12)
noise4 = PerlinNoise(octaves=24)

xpix, ypix = 200, 200
pic = []
for i in range(xpix):
    row = []
    for j in range(ypix):
        noise_val = noise1([i/xpix, j/ypix])
        noise_val += 0.5 * noise2([i/xpix, j/ypix])
        noise_val += 0.25 * noise3([i/xpix, j/ypix])
        noise_val += 0.125 * noise4([i/xpix, j/ypix])

        row.append(noise_val)
    pic.append(row)

plt.imshow(pic, cmap='gray')
plt.show()
'''
