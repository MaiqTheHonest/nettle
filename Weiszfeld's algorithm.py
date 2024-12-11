from main import KMeansClustering
import numpy as np
from time import perf_counter

ED = KMeansClustering.euclidean_distance

points = np.asarray(np.random.randint(0,10, size=(3,2)))
#a, b = np.array([0,0]), np.array([3,4])
center = np.asarray([2,2])
print(points)
#ini = points[points.shape[0] // 2]
ini = [0,0]


k=1

def TC(centroid, points) -> float:
    return np.sum(ED(centroid, points)) + k*ED(centroid, center)

step = 1
TC_now = TC(ini, points)

while step > 0.00001:
    print(TC_now)
    
    f=False
    for dx, dy in [(-1,0), (1,0), (0,1), (0, -1)]:
        cur = [ini[0]+step*dx, ini[1]+step*dy]
        TC_new = TC(cur, points=points)
        if TC_new < TC_now:
            TC_now = TC_new
            ini = cur
            f=True
    print(cur)        
    if f==False:
        step = step/10
print(TC_now, cur)
# timer1 = perf_counter()
# ED(a, b)
# timer2 = perf_counter()
# print(timer2 - timer1)