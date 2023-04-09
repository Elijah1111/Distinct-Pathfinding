#!/usr/bin/env python3
import pyfastnoisesimd as fns
import matplotlib.pyplot as plt#just for displaying the image
import time
import numpy as np
tmp   = 1028
shape = [tmp,tmp]
seed  = np.random.randint(2**31)
N_threads = 4

perlin = fns.Noise(seed=seed, numWorkers=N_threads)
perlin.frequency = 0.02
perlin.noiseType = fns.NoiseType.Perlin
perlin.fractal.octaves = 4
perlin.fractal.lacunarity = 2.1
perlin.fractal.gain = 0.45
perlin.perturb.perturbType = fns.PerturbType.NoPerturb

start = time.time()
img = perlin.genAsGrid(shape)
end = time.time()
print(f"{end-start}")
plt.imshow(img, cmap='gray')
plt.show()
