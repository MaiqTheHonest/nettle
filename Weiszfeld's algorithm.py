from main import KMeansClustering
import numpy as np
from time import perf_counter


ED = KMeansClustering.euclidean_distance


timer1 = perf_counter()


# points = np.asarray(np.random.randint(0,10, size=(3,2)))
points = np.asarray([[1,5],[3,2],[7,3]])

#a, b = np.array([0,0]), np.array([3,4])
center = np.asarray([7,7])
#print(points)
#ini = points[points.shape[0] // 2]
ini = [0,0]


k=1

def TC(centroid, points) -> float:
    return np.sum(ED(centroid, points)) #+ k*ED(centroid, center)

step = 1
TC_now = TC(ini, points)

while step > 0.000001:
    #print(TC_now)
    
    f=False
    for dx, dy in [(-1,0), (1,0), (0,1), (0, -1)]:
        cur = [ini[0]+step*dx, ini[1]+step*dy]
        TC_new = TC(cur, points=points)
        if TC_new < TC_now:
            TC_now = TC_new
            ini = cur
            f=True
    #print(cur)        
    if f==False:
        step = step/10
print(TC_now, cur)

timer2 = perf_counter()
print(timer2 - timer1)