
import numpy as np 
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from Weiszfelds_algorithm import weighted_distance # import own Weiszfeld's algorithm
from matplotlib.collections import LineCollection

start_time = perf_counter()


class GeomMedianClustering:

    def __init__(self, fig, ax1, ax2, k=3):

        self.k = k
        self.centroids = None
        self.perfect_cent = [None, None]  # filler for EI
        self.fig = fig
        self.ax1 = ax1
        #self.ax2 = ax2
        self.scatter1 = None
        #self.scatter1.set_cmap("tab20") # HAS to be called separately outside of scatter1 init
        self.scatter2 = None
        self.scatter3 = None



    def initialize_scatters(self):      # this is not in __init__ because scatters overlap when >1 class instances are created 
        self.scatter1 = self.ax1.scatter(test_data[:, 0], test_data[:, 1], label='points', s=48, ec='black', lw=0.05)
        self.scatter1.set_cmap("Accent") # HAS to be called separately outside of scatter1 init
        self.scatter2 = self.ax1.scatter([], [], c='black', marker="*", s=96,  label='centroids')
        self.scatter3 = self.ax1.scatter(0, 0, c='red', label='perfect centre', marker="*", s=96)

    def uninitialize_scatters(self):    # again, avoids overlap with multiple instances
        self.scatter1.remove()
        self.scatter2.remove()
        self.scatter3.remove()
    

    @staticmethod
    def euclidean_distance(data_point, centroids: np.array): # np.atleast_2d version is slower than this ugly [[]] / [] case handling
        
        if centroids.ndim == 1:
            return np.sqrt(np.sum((centroids - data_point)**2, axis=0))

        blarg = np.sqrt(np.sum((centroids - data_point)**2, axis=1))

        return blarg[0] if centroids.shape[0] == 1 else blarg   # returns array of distances to each centroid in centroids, not their sum


        
    def update_graph(self, labs=None, dat=None, clear=True, add=True):

        self.scatter1.set_array(labs)
        self.scatter2.set_offsets(self.centroids)
        point_segments = np.stack([dat, self.centroids[labs]], axis=1)

        centroid_segments = np.stack([self.centroids, [self.perfect_cent]*self.k], axis=1)

        if add is True:

            self.lca = LineCollection(point_segments, color='black', alpha=0.1)
            self.lcb = LineCollection(centroid_segments, linewidth=2, linestyle="dashed", color='#3d4547', alpha=0.8)
            self.ax1.add_collection(self.lca)
            self.ax1.add_collection(self.lcb)

        plt.draw()
        plt.pause(0.05)

        if clear is True:
            self.lca.remove()
            self.lcb.remove()
        print(".")



    def fit(self, X, center=None, weight=1, max_iterations=200): # X is data

        self.initialize_scatters()
        network_total_cost = 0

        self.perfect_cent = center if center is not None else weighted_distance(points=X.tolist())[:2]   # if center was not provided, use geometric median
        self.scatter3.set_offsets(self.perfect_cent)

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
                    wittle = (np.squeeze(X[indices])).tolist()                          # squeeze turns [[]] into [] within X
                    wittle = [wittle] if not isinstance(wittle[0], list) else wittle    # deals with edge case of only 1 point in a given cluster, which breaks weighted_distance
                    total_cost = weighted_distance(wittle, center=self.perfect_cent, weight=weight) 
                    network_total_cost += total_cost[2]
                    cluster_centers.append(total_cost[:2])  # readjusts each cluster centroid to geometric median of points belonging to it [x, y, TC]



            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                self.centroids = np.array(cluster_centers)
                break
            # else:
            self.centroids = np.array(cluster_centers)
            
            
            self.update_graph(labs=y, dat=X)

        

        self.update_graph(labs=y, dat=X, clear=False, add=True)
        plt.pause(0.3)
        self.update_graph(labs=y, dat=X, clear=True, add=False)
        self.uninitialize_scatters()
        

        return cluster_centers, network_total_cost


def main():
    
    fig = plt.figure(figsize=(10,8))

    gs0 = fig.add_gridspec(4, 4, wspace=0.25, right=0.90)
    
    main_ax = fig.add_subplot(gs0[0:4, 0:3])
    
    sub_ax = fig.add_subplot(gs0[0,3])
    sub_ax.set_yticklabels([])
    sub_ax.yaxis.set_label_position("right")
    klip, = sub_ax.plot(0, 0)
    
    sub_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))    # ensures xticks are always integers
    sub_ax.set_xlim([1, 5])
    sub_ax.set_xlabel('centroid count, k')
    sub_ax.set_ylabel('network cost')

    text_ax = fig.add_subplot(gs0[1,3])
    text_ax.set_axis_off()
    text_obj = text_ax.text(0, -0.25, f"centroid count = 0 \norigin-centroid weight = {weight} \ncentroid-point weight = 1 \nnetwork cost = 0",
                             fontsize=11, linespacing=2)

    instances = [GeomMedianClustering(k=i, fig=fig, ax1=main_ax, ax2=sub_ax) for i in range(1, limit)]

    TC = []
    OC = []
    
    for count, instance in enumerate(instances):
        
        output = instance.fit(test_data, weight=weight)
        blarg = round(output[-1], 2)
        TC.append(blarg)
        OC.append(count+1)
        text_obj.set_text(f"centroid count = {count+1} \norigin-centroid weight = {weight} \ncentroid-point weight = 1 \nnetwork cost = {blarg}")
        
        klip.remove()
        klip, = sub_ax.plot(OC, TC, marker='.',linestyle='-', c="#269eb3")
        sub_ax.set_xlim([1, count+5])



if __name__ == "__main__":
    weight = 2
    limit = 10
    np.random.seed(123)
    # test_data = np.random.randint(0, 100, (500, 2))
    test_data = np.concatenate([np.random.normal(0, 5, size=(100, 2)), 
         np.random.normal(5, 3, size=(100, 2))]) # bimodal normal draw
    main()
    

    # print(f"{results1[0]} are the optimal facility locations")
    # print(f"total network cost is {results1[1]} units")
    # print(f"script finished in {perf_counter() - start_time} seconds")



    
