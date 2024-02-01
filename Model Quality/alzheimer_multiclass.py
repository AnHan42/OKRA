#Set up the environment and upload the data
from sklearn.preprocessing import StandardScaler
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
from PIL import Image
import seaborn as sns
from sklearn import decomposition, preprocessing, svm
sns.set()
import Kernel_lib
from sklearn.model_selection import train_test_split

#Dataset that should go with Alzheimer label
very_mild = glob('/home/sc.uni-leipzig.de/jf994pyjo/OKRA/data/Very_Mild_Demented/*')
mild = glob('/home/sc.uni-leipzig.de/jf994pyjo/OKRA/data/Mild_Demented/*')
moderate = glob('/home/sc.uni-leipzig.de/jf994pyjo/OKRA/data/Moderate_Demented/*')

#Dataset without Alzheimer
non = glob('/home/sc.uni-leipzig.de/jf994pyjo/OKRA/data/Non_Demented/*')

resized_image_array=[]
resized_image_array_label=[]


width = 256
height = 256
new_size = (width,height) #the data is just black to white 

def resizer(image_directory):
    for file in image_directory: #tried with os.listdir but could work with os.walk as well
        img = Image.open(file)
        #preserve aspect ratio
        img = img.resize(new_size)
        array_temp = np.array(img)
        shape_new = width*height
        img_wide = array_temp.reshape(1, shape_new)
        resized_image_array.append(img_wide[0])
        if image_directory == non:
            resized_image_array_label.append(0)
        elif image_directory == very_mild:
            resized_image_array_label.append(1)
        elif image_directory == mild:
            resized_image_array_label.append(2)
        else:
            resized_image_array_label.append(3)

resizer(non)
resizer(very_mild)
resizer(mild)
resizer(moderate)

#split the data to test and training
#train_x, test_x, train_y, test_y = train_test_split(resized_image_array, resized_image_array_label, test_size = 0.2)

#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
partitions, y = Kernel_lib.partition_dataset(resized_image_array, resized_image_array_label, 2)

k = 10
n_features = partitions[0].shape[1] 

# ORTHO kernel
gamma = Kernel_lib.get_gamma(n_features, 1, seed=42)
A_prime = partitions[0] @ gamma.T
B_prime = partitions[1] @ gamma.T

full = np.dot(A_prime, B_prime.T)

clf = svm.SVC(kernel = 'precomputed', probability=True)
clf.fit(Kernel_lib.polynomial_kernel(full, degree=3), y)

#full1 = np.dot(partitions[0], partitions[1].T)

clf3 = svm.SVC(kernel = 'poly', probability = True)
clf3.fit(resized_image_array, resized_image_array_label)

print('Performance on Naive Kernel: ')
Kernel_lib.metrics_micro(clf3, resized_image_array, resized_image_array_label, cv=5)

print('Performance of OKRA with Custom (Polynomial) Gram Kernel: ')
Kernel_lib.metrics_micro(clf, full, resized_image_array_label, cv=5)