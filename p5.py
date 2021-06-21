import pandas as pd
import numpy as np
a = [np.array([0.1,0.2]),np.array([0.1,0.2])]
pd.DataFrame(a).to_csv("aaa.csv")