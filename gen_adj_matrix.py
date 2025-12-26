import numpy as np

cam_order = [
    "cam1", "cam2", "cam3", "cam4",
    "cam5", "cam6", "cam7", "cam8"
]

cams = {
    "cam1": (10.80071000917009, 106.66078135371208),
    "cam2": (10.801032758949276, 106.65788859128952),
    "cam3": (10.801272515703587, 106.64736360311508),
    "cam4": (10.802405429645702, 106.64152711629868),
    "cam5": (10.807016081822823, 106.63496643304825),
    "cam6": (10.803735939399289, 106.63591861724854),
    "cam7": (10.801952264581729, 106.63651406764984),
    "cam8": (10.792883550206625, 106.65360778570175),
}

cam_index = {cam: i for i, cam in enumerate(cam_order)}

adj_list = {
    "cam1": ["cam2", "cam8"],
    "cam2": ["cam3", "cam1"],
    "cam3": ["cam4", "cam2"],
    "cam4": ["cam5", "cam3"],
    "cam5": ["cam6", "cam4"],
    "cam6": ["cam7", "cam5"],
    "cam7": ["cam8", "cam6"],
    "cam8": ["cam1", "cam7"],
}

from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

import numpy as np

sigma = 0.6  # km – rất hợp nội đô HCM (0.5–0.8 là đẹp)

N = len(cam_order)
A = np.zeros((N, N), dtype=np.float32)

# self-loop
for i in range(N):
    A[i, i] = 1.0

# weighted edges theo distance
for cam, neighbors in adj_list.items():
    i = cam_index[cam]
    lat1, lon1 = cams[cam]

    for nb in neighbors:
        j = cam_index[nb]
        lat2, lon2 = cams[nb]

        d = haversine(lat1, lon1, lat2, lon2)
        weight = np.exp(-d / sigma)

        A[i, j] = weight
        A[j, i] = weight  # undirected
np.save("adj_matrix.npy", A)
print("Adjacency matrix shape:", A.shape)
print(A)
