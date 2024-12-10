from main import KMeansClustering
import numpy as np
from time import perf_counter

ED = KMeansClustering.euclidean_distance
points = np.asarray(np.random.randint(0,10, size=(4,2)))
a, b = np.array([0,0]), np.array([3,4])

print(ED(a, b))
print(b.ndim)
inix, iniy = 0,0

timer1 = perf_counter()
ED(a, b)

timer2 = perf_counter()
print(timer2 - timer1)