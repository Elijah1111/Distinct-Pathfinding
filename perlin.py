#!/usr/bin/env python3
import random
import numpy as np
import matplotlib.pyplot as plt#just for displaying the image
class Perlin():
    graidents = np.array(0)#graident vectors
    image = np.array(0)
    def __init__(self,n=100, octaves=2):
        self.makeGraidents(n)
        self.makeImage(n,octaves) 

    def makeGraidents(self,n):#TODO this may need wrapped in a try case very rarely it might fail with a division by 0
        graidents = np.random.random((n,n,2))*2 -1;#make a new matrix of graidents
        normals = np.sqrt(np.einsum('...i,...i', graidents, graidents));#super fast, unsafe if magnitude is too large, should be fine because max would be sqrt(3)
        self.graidents = graidents/normals[:,:,np.newaxis]#normalize the graidents

    def dotProd(self, ix, iy, x, y): #compute dot product of distance and graident vector
        vec = self.graidents[ix][iy];

        dx = x - ix#distance vec unwrapped
        dy = y - iy

        return dx*vec[0] + dy*vec[1]

    def lerp(self, a, b, val):#interpolate two vectors
        return val*b + (1-val) * a
        
    def perlin(self, x, y, scale):#do the actual perlin calculation
        x *= scale
        y *= scale
        
        #Grab the 4 nearest points
        x0 = (int(np.floor(x)))%100#TODOthis should be fixed
        x1 = (x0 + 1)%100
        y0 = (int(np.floor(y)))%100
        y1 = (y0 + 1)%100
        
        #interpolate those vectors
        sx = x - x0
        sy = y - y0

        vec0 = self.dotProd(x0,y0,x,y)
        vec1 = self.dotProd(x1,y0,x,y)
        ix0 = self.lerp(vec0,vec1,sx)
        
        vec0 = self.dotProd(x0,y1,x,y)
        vec1 = self.dotProd(x1,y1,x,y)
        ix1 = self.lerp(vec0,vec1,sx)
        
        return self.lerp(ix0,ix1,sy)



    def makeImage(self, n, octs):
        image = np.zeros((n*10,n*10))
        for i in range(0,n*9):#TODO this needs to be hella faster
            for j in range(0,n*9):
                tmp = 0
                for o in range(0,octs):
                    tmp += self.perlin(i/10,j/10,2**o)*(2**-o)
                if(tmp <=0.001):
                    continue
                tmp = (np.log(tmp) - np.log(1e-2)) / (np.log(1) - np.log(1e-2))
                image[i][j] = tmp
        self.image = image
    
#make some noise
if __name__ == "__main__":
    noise = Perlin()
    plt.imshow(noise.image, cmap='gray')
    plt.show()

'''
#works but very slow at large sizes
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
