from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np

def get_clusters(audio_features):
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