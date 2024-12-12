
import numpy as np 
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from TC_no_numpy import weighted_distance # import own Weiszfeld's algorithm

start_time = perf_counter()

class GeomMedianClustering:

    def __init__(self, k=3):
        self.k = k
        self.centroids = None
        self.perfect_cent = None
        self.fig = plt.figure()
        self.scatter1 = plt.scatter(test_data[:, 0], test_data[:, 1], label='points')
        #self.scatter1.set_cmap("tab20") # HAS to be called separately outside of scatter1 init
        self.scatter2 = plt.scatter([], [], c='red', marker="*", s=96,  label='centroids')
        self.scatter3 = plt.scatter([2], [2], c='green', label='perfect centre', marker="*", s=96)
    
    @staticmethod
    def euclidean_distance(data_point, centroids: np.array): # np.atleast_2d version is slower than this ugly [[]] / [] case handling
        
        if centroids.ndim == 1:
            return np.sqrt(np.sum((centroids - data_point)**2, axis=0))

        blarg = np.sqrt(np.sum((centroids - data_point)**2, axis=1))

        return blarg[0] if centroids.shape[0] == 1 else blarg   # returns array of distances to each centroid in centroids, not their sum


        
    def update_graph(self, labs=None):
        self.scatter1.set_array(labs)
        self.scatter2.set_offsets(self.centroids)

        plt.pause(0.1)
        plt.draw()
        print(".")

    def fit(self, X, center=None, weight=1, max_iterations=200): # X is data

        network_total_cost = 0
        self.perfect_cent = weighted_distance([X.tolist()])[:2] if center is not None else center   # if center was not provided, use geometric median
        
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), 
                                           size=(self.k, X.shape[1]))    # get the new centroids within dimension range



        for _ in range(max_iterations):
            y = []

            for data_point in X:
                
                distances = GeomMedianClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)   # finds index of the nearest (i.e. smallest distance) centroid
                
                y.append(cluster_num)
            
            y = np.array(y)



            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))     # i.e. which indices belong to cluster k, for each k?



            cluster_centers = []

            for count, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[count])
                else:
                    total_cost = weighted_distance(X[indices].tolist(), center=self.perfect_cent, weight=weight)
                    network_total_cost += total_cost[2]
                    cluster_centers.append(total_cost[:2])  # readjusts each cluster centroid to geometric median of points belonging to it [x, y, TC]

            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)
            
            
            self.update_graph(labs=y)
 
        return network_total_cost

if __name__ == "__main__":
    #test_data = np.random.randint(0, 100, (100, 2))
    np.random.seed(12345)
    test_data = np.concatenate([np.random.normal(0, 5, size=(200, 2)), 
         np.random.normal(5, 3, size=(200, 2))]) # bimodal normal draw
    
    gmeans = GeomMedianClustering(k=2)
    results = gmeans.fit(test_data, weight=5)
    print(f"total network cost is {results} units")
    print(f"script finished in {perf_counter() - start_time} seconds")
