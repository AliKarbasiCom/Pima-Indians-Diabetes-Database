import numpy as np
import pandas as pnds
import matplotlib.pyplot as plt
import seaborn as cbrn

cbrn.set(rc={'figure.figsize': [7, 7]}, font_scale=1.2)
df = pnds.read_csv('diabetes.csv')

print(df.describe())