import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url='glaucoma.csv'
df = pd.read_csv(url)

print(df.shape)
print(df.showImages)