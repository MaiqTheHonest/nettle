
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

start_time = perf_counter()

#test_data = np.random.randint(0, 100, (100, 2))
test_data = np.concatenate([np.random.normal(0, 5, size=(200, 2)), 
    np.random.normal(5, 3, size=(200, 2))]) # bimodal normal draw


class KMeansClustering:

    def __init__(self, k=3):
        self.k = k
        self.centroids = None
        self.perfect_cent = None
        self.fig = plt.figure()
        self.scatter1 = plt.scatter(test_data[:, 0], test_data[:, 1], label='points')
        #self.scatter1.set_cmap("tab20") # HAS to be called separately outside of scatter1 init
        self.scatter2 = plt.scatter([], [], c='red', marker="*", s=96,  label='centroids')
        self.scatter3 = plt.scatter([], [], c='green', label='perfect centre')
    
    @staticmethod
    def euclidean_distance(data_point, centroids: np.array): # np.atleast_2d version is slower than this ugly [[]] / [] case handling
        if centroids.ndim == 1:
            return np.sqrt(np.sum((centroids - data_point)**2, axis=0))
        
        blarg = np.sqrt(np.sum((centroids - data_point)**2, axis=1))

        return blarg[0] if centroids.shape[0] == 1 else blarg

            # OR
            # if np.size(centroids) == 2:
            #     return blarg[0]
            # else:
            #     return blarg

        

    
    def update_graph(self, labs=None):
        self.scatter1.set_array(labs)
        self.scatter2.set_offsets(self.centroids)
        plt.pause(0.1)
        plt.draw()
        print(".")

    def fit(self, X, max_iterations=200): # X is data
        self.perfect_cent = [np.mean(X[:, 0]), np.mean(X[:, 1])]
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), 
                                           size=(self.k, X.shape[1]))    # get the new centroids within dimension range
        print(np.shape(self.centroids), type(self.centroids))
        print(self.centroids)
        for _ in range(max_iterations):
            y = []

            for data_point in X:
                
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)   # finds index of the nearest (i.e. smallest distance) centroid
                
                y.append(cluster_num)
            
            y = np.array(y)

            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
            
            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])  # readjusts centroids to mean of new cluster

            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)
            
            
            self.update_graph(labs=y)
            
        return y

if __name__ == "__main__":
    kmeans = KMeansClustering(k=5)
    labels = kmeans.fit(test_data)
    print(f"script finished in {perf_counter() - start_time} seconds")
