import scipy.misc
import random
import re
import numpy as np
from PIL import Image
import imageio

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
# with open("/home/li/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/End_2_End/07012018/data.txt") as f:
with open("/home/li/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/imredd_binary_data.txt") as f:
    for line in f:
        # xs.append("driving_dataset/" + line.split()[0])
        
        # xs.append("data/End_2_End/07012018/data/" + line.split()[0])
        xs.append("/home/li/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/binary_data/img_pre-pro" + line.split()[0])

        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        
        # ys.append(float(line.split()[1]) * scipy.pi / 180)
        ys.append(float(re.split(',| ', line)[1]) * scipy.pi / 180)

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        x_out.append(np.array(Image.fromarray(imageio.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:]).resize([200, 66])) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        x_out.append(np.array(Image.fromarray(imageio.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:]).resize([200, 66])) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
