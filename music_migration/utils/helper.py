from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial import distance
import numpy as np

def get_num_clusters(audio_features):
    # using MeanShift to get an estimate
    bandwidth = estimate_bandwidth(audio_features,
                                    quantile=0.3,
                                    n_jobs=-1)
    ms = MeanShift(bandwidth=bandwidth,
                    bin_seeding=False,
                    n_jobs=-1,
                    max_iter=5000)

    ms.fit(audio_features)

    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(ms.labels_)
    n_clusters_ = len(labels_unique)
    
    return n_clusters_, cluster_centers

def get_cosine_distance(row, centers):
    return distance.cosine( row.drop(["remainder__name",
                                      "remainder__uri",
                                     "remainder__id",
                                     "cluster"]),
                            centers[row['cluster']])