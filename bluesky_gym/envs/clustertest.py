import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
from matplotlib import pyplot as plt

# D_HEADING = 45
# ymin = 5
# ymax= 20
# xmin = 5
# xmax= 20

# N_AC = 20
# xy = np.array([[5,5]])
# for i in range(N_AC):
#     yrand = random.uniform(ymin, ymax)
#     xrand = random.uniform(xmin, xmax)
#     xysample = np.array([xrand,yrand])
#     xy = np.append(xy,xysample)

# kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(xy)
# kmeans.cluster_centers_


# plt.scatter(x, y)
# plt.show()

jeff = np.array([1,2,3,4,5])

print(jeff[1:3])