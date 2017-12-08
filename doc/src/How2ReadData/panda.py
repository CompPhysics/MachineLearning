import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd

data = pd.read_csv('src/Hudson_Bay.csv', delimiter=',', skiprows=1)


data_pandas = pd.DataFrame(data)
display(data_pandas)
