from numpy import dot
from numpy.linalg import norm
import numpy as np

def load_image(address):
    return np.loadtxt(address)

def cosine_sim(a,b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def euclidean_dist(a,b):
    dist = (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2
    return dist

def k_means(k,imgs):
    # randomly initialise centroids
    number_of_rows = imgs.shape[0]
    random_indices = np.random.choice(number_of_rows, size=k, replace=False)
    centroids = imgs[random_indices, :]

    #optimise over representatives
    label_dict = {}
    for i in range(number_of_rows):
        label_dict[i] = centroids[0]
        for centroid in centroids:
            if euclidean_dist(imgs[i],centroid)< euclidean_dist(imgs[i],label_dict[i]):
                label_dict[i] = centroid

    #optimise over clusters
    for centroid in centroids:


imgs = load_image("kmeans-image.txt").astype(int)
k_means(3,imgs)
