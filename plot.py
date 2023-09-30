import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from plato.config import Config
from torchvision import transforms
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
afl = pd.read_csv('D:/Desktop/AFL.csv')
oort = pd.read_csv('D:/Desktop/OORT.csv')
pisces = pd.read_csv('D:/Desktop/PISCES.csv')
plt.plot(oort['round'],oort['accuracy'],label = 'oort')
plt.plot(pisces['round'],pisces['accuracy'], label = 'pisces')
plt.plot(afl['round'],afl['accuracy'], label = 'active-federated-learning')
plt.suptitle('accuracy deponds on rounds')
plt.title('accuracy', fontdict={'fontsize':15,'fontweight':'bold'})
plt.xlabel('rounds')
plt.legend()
plt.ylabel('accuracy')
plt.show()
