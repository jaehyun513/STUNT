import numpy as np
from sklearn.cluster import KMeans

np.random.seed(0)

x = np.load('xtrain.npy')
y = np.load('ytrain.npy')

num_train = int(len(x) * 0.8)
idx = np.random.permutation(len(x))
train_idx = idx[:num_train]
val_idx = idx[num_train:]

train_x = x[train_idx]
val_x = x[val_idx]

np.save('train_x.npy', train_x)
np.save('val_x.npy', val_x)

model = KMeans(n_clusters = 2)
model.fit(val_x)
labels = model.predict(val_x)
np.save('pseudo_val_y.npy', labels)