import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

n_samples = 1024017
n_features = 100
n_train_test_split = int(n_samples * 0.8)
test_index = np.arange(n_train_test_split, n_samples)
