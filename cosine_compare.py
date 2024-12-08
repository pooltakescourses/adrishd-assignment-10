import pandas as pd
import numpy as np
from scipy.spatial import distance


def compare(df, vector):
  def score_fn(x):
    x = x / np.linalg.norm(x)
    return 1 - distance.cosine(vector.squeeze(), x.squeeze())

  df2 = df[['file_name', 'embedding']]
  df2['scores'] = df2['embedding'].apply(score_fn)
  df2 = df2.sort_values(by='scores', ascending=False)

  res = dict()
  for idx, (name, score) in df2[['file_name', 'scores']].iloc[:3].iterrows():
    res[name] = score

  return res
