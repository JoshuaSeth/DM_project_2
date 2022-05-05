from data.load import load_data
import os
import numpy as np

df = load_data()

print(df[df['srch_id']== 185711])
