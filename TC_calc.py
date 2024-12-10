from main import KMeansClustering
import numpy as np

k = 5
ED = KMeansClustering.euclidean_distance
center = np.asarray([2,2])
points = np.asarray(np.random.randint(0,10, size=(4,2)))
centroid = np.asarray([np.mean(points[:, 0]), np.mean(points[:, 1])])
print(points)
print(centroid)
TC_standard = np.sum(ED(centroid, points)) + k*np.sqrt(np.sum((centroid - center)**2))


line = np.linspace(centroid, center, 100)

funcvalues = []
for point in line:
    TC = np.sum(ED(point, points)) + k*(np.sqrt(np.sum((point - center)**2)))
    funcvalues.append(TC)
print(f"min is {min(funcvalues)}")



#points = np.vstack((points, center))
centroid2_x = (np.sum(points[:, 0]) + k*center[0])/(len(points) + k)
centroid2_y = (np.sum(points[:, 1]) + k*center[1])/(len(points) + k)
print(centroid2_x, centroid2_y)
centroid2 = np.asarray([centroid2_x, centroid2_y])
TC_new = np.sum(ED(centroid2, points)) + k*(np.sqrt(np.sum((centroid2 - center)**2, axis=1)))
print(points)
print(TC_new)

print(center[1])
# grad = (centroid[1] - center[1]) / (centroid[0] - center[0])
# intercept = centroid[1] - grad*(centroid[0])

# print(f"{grad}*x + {intercept}")
