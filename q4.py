from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np

def load_image(address):
    return np.loadtxt(address)

def cosine_sim(a,b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def euclidean_dist(a,b):
    dist = (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2
    return dist

def cluster_mean(imgs,index_list):
    #given list of index, compute new cluster mean
    size = len(index_list)
    x_mean,y_mean,z_mean = 0, 0, 0
    for index in index_list:
        x_mean += imgs[index][0]
        y_mean += imgs[index][1]
        z_mean += imgs[index][2]
    x_mean, y_mean, z_mean = x_mean/size, y_mean/size, z_mean/size
    return np.array([x_mean,y_mean,z_mean])

def k_means(k,imgs,iterations):
    # randomly initialise centroids
    number_of_rows = imgs.shape[0]
    random_indices = np.random.choice(number_of_rows, size=k, replace=False)
    centroids = imgs[random_indices, :]


    while iterations>0:
        label_dict = {}
        centroid_dict = {}
        # optimise over representatives
        for i in range(number_of_rows):
            label_dict[i] = centroids[0]
            c_index = 0
            for j in range(len(centroids)):
                if euclidean_dist(imgs[i],centroids[j])< euclidean_dist(imgs[i],label_dict[i]):
                    label_dict[i] = centroids[j]
                    c_index = j
            if c_index not in centroid_dict.keys():
                centroid_dict[c_index] = [i]
            else:
                centroid_dict[c_index].append(i)

        #optimise over clusters
        for i in range(len(centroids)):
            centroids[i] = cluster_mean(imgs,centroid_dict[i])

        #report error and new centroids
        error = 0
        for i in range(len(centroids)):
            for index in centroid_dict[i]:
                error += euclidean_dist(centroids[i],imgs[index])
        print("The error is ",error,"and the new centroids are ",centroids)
        iterations -= 1
    return centroids,label_dict

imgs = load_image("kmeans-image.txt").astype(int)
imgs = np.asarray(imgs)
plt.figure()
plt.imshow(imgs, vmin=0,vmax=255)
plt.savefig('./visualize_after_replacing.png')
#centroids, label_dict = k_means(8,imgs,5)

# for i in range(len(imgs)):
#     imgs[i] = tuple(label_dict[i])
#
# plt.imshow(imgs)
# plt.show()