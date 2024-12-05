import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from time import perf_counter
start_time = perf_counter()

class KMeansClustering:

    def __init__(self, k=3):
        self.k = k
        self.centroids = None
        self.perfect_cent = None 
    
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))



    def fit(self, X, max_iterations=200): # X is data
        self.perfect_cent = [np.mean(X[:, 0]), np.mean(X[:, 1])]
        print(self.perfect_cent)
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), 
                                           size=(self.k, X.shape[1]))    # get the new centroids within dimension range


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

        return y
    

test_data = np.random.randint(0, 100, (100, 2))

kmeans = KMeansClustering(k=3)

labels = kmeans.fit(test_data)
trace1 = go.Scatter(
    x=test_data[:, 0],
    y=test_data[:, 1],
    mode="markers",
    name="points",
    marker=dict(color=labels)
    )

trace2 = go.Scatter(
    x=kmeans.centroids[:, 0],
    y=kmeans.centroids[:, 1],
    mode="markers",
    marker=dict(size=12, color="green", symbol="circle-x-open"),
    name='centroids'
)
trace3 = go.Scatter(
    x=[kmeans.perfect_cent[0]],
    y=[kmeans.perfect_cent[1]],
    mode="markers",
    marker=dict(size=8, color="green", symbol="diamond"),
    name='perfect centre'
)

fig = go.Figure()
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.add_trace(trace3)


fig['layout'].update(height = 500, width = 600)

fig.show()

print(f"script finished in {perf_counter() - start_time} seconds")
