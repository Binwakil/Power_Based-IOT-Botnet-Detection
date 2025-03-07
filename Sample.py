import tensorflow.keras
import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score

mat_data = scipy.io.loadmat('dataset.mat')
#print(mat_data)

# Access the variables in the .mat file
class_1_idle = mat_data['class_1_idle']
class_2_iot = mat_data['class_2_iot']  # Replace 'variable_name' with the actual variable name in the .mat file
class_3_reboot = mat_data['class_3_reboot']
class_4_mirai = mat_data['class_4_mirai']

# Convert the data to a NumPy array if needed
class_1_idle = np.array(class_1_idle)
class_1_idle = class_1_idle[:, 0:1000]
class_1_idle = class_1_idle.T
class_2_iot = np.array(class_2_iot)
class_2_iot = class_2_iot[:, 0:1000]
class_2_iot = class_2_iot.T
class_3_reboot = np.array(class_3_reboot)
class_3_reboot = class_3_reboot[:, 0:1000]
class_3_reboot = class_3_reboot.T
class_4_mirai = np.array(class_4_mirai)
class_4_mirai = class_4_mirai[:, :]
class_4_mirai = class_4_mirai.T
print(class_1_idle.shape)
print(class_2_iot.shape)
print(class_3_reboot.shape)
print(class_4_mirai.shape)

label_zero = np.zeros((3000, 1))
label_one = np.ones((3000, 1))

normal_data = np.concatenate((class_1_idle, class_2_iot, class_3_reboot), axis=0)
normal_data = np.concatenate((normal_data, label_zero), axis=1)
attack_data = np.concatenate((class_4_mirai, label_one), axis=1)
print(normal_data.shape)
print(attack_data.shape)

raw_data = np.concatenate((normal_data, attack_data))
labels = raw_data[:, -1]
data = raw_data[:, 0:-1]

print("Raw Data shape = ", raw_data.shape)

cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# Then just plot it:
disp.plot()
# And show it:
plt.show()
