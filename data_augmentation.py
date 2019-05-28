import numpy as np
import matplotlib.pyplot as plt
import skimage


#load_datasets

X_train=np.load('X_train.npy')
Y_train=np.load('Y_train.npy')
print(X_train.shape)
print(Y_train.shape)

#create new numpy arrays to double training set data

X_train_new=np.zeros((2160,64,64,3),dtype=np.float32)
Y_train_new=np.zeros((1,2160),dtype=np.int32)

'''
print(X_train.shape)
plt.imshow((X_train[228].reshape(64,64,3)).astype(np.uint8))
plt.show()
print(Y_train[0][228])
'''

id=0

#flip one_third of images

for i in range(360):
    data=X_train[i].reshape(64,64,3)
    X_train_new[id]=data
    Y_train_new[0][id]=Y_train[0][i]
    id+=1
    data=(np.flip(X_train[i],axis=1)).reshape(64,64,3)
    X_train_new[id] = data
    Y_train_new[0][id]=Y_train[0][i]
    id+=1


#rotate one_third of image by 30 degrees

for i in range(360,720):
    data = X_train[i].reshape(64, 64, 3)
    X_train_new[id] = data
    Y_train_new[0][id] = Y_train[0][i]
    id += 1
    data =skimage.transform.rotate(X_train[i], 30, preserve_range=True).astype(np.uint8)
    X_train_new[id] = data
    Y_train_new[0][id] = Y_train[0][i]
    id += 1

#rotate one_third of image by 60 degrees

for i in range(720,1080):
    data = X_train[i].reshape(64, 64, 3)
    X_train_new[id] = data
    Y_train_new[0][id] = Y_train[0][i]
    id += 1
    data = skimage.transform.rotate(X_train[i], 60, preserve_range=True).astype(np.uint8)
    X_train_new[id] = data
    Y_train_new[0][id] = Y_train[0][i]
    id += 1

save new training data

np.save('X_train_new',X_train_new)
np.save('Y_train_new',Y_train_new)

X_train_new=np.load('X_train_new.npy')
Y_train_new=np.load('Y_train_new.npy')


plt.imshow((X_train_new[2159].reshape(64,64,3)).astype(np.uint8))
plt.show()
print(Y_train_new[0][2159])
