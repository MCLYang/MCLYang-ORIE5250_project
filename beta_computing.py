import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize.optimize import OptimizeWarning
from sklearn import cluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pdb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--NUM_CLUSTERS', type=int,default=2,help='NUM_CLUSTERS')
args = parser.parse_args()




NUM_CLUSTERS = int(args.NUM_CLUSTERS)
DATA_DIR = "data.csv"

def normalization(df,names):
  stat_dict = {}
  df_normalized = df.copy()
  for name in names:
    mean = np.array(df_normalized[[name]]).mean()
    std = np.array(df_normalized[[name]]).std()
    df_normalized[[name]] = (df_normalized[[name]]-mean)/std
    stat_dict[name] = (mean,std)

  return df_normalized, stat_dict

def unnormalization(df,names,stat_dict):
  df_unnormalized = df.copy()
  for name in names:
    mean = stat_dict[name][0]
    std = stat_dict[name][1]
    df_unnormalized[[name]] = (df_unnormalized[[name]]*std)+mean
  return df_unnormalized

#==============Compute clusters===================
df = pd.read_csv(DATA_DIR)
customers_features = ["srch_id","srch_booking_window",	"srch_adults_count",	"srch_children_count",	"srch_room_count",	"srch_saturday_night_bool"]
customers_df = df[customers_features]
customers_df = customers_df.drop_duplicates()
id = np.array(customers_df["srch_id"])
customers_df = customers_df.drop(columns=['srch_id'])
normalized_col = ["srch_booking_window",	"srch_adults_count",	"srch_children_count",	"srch_room_count","srch_saturday_night_bool"]
df_normalized, stat_dict = normalization(customers_df,normalized_col)
unnormalized_data_np = np.array(customers_df)
normalized_data_np = np.array(df_normalized)
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(normalized_data_np)
labels_set = set(kmeans.labels_)
labels_dict = {i:0 for i in labels_set}

for i in kmeans.labels_:
  labels_dict[i] = labels_dict[i]+1

# print(labels_dict)
center = kmeans.cluster_centers_
center_df = pd.DataFrame(center,columns=["srch_booking_window",	"srch_adults_count",	"srch_children_count",	"srch_room_count","srch_saturday_night_bool"])

#center
center_df = unnormalization(center_df,normalized_col,stat_dict)

labels = kmeans.labels_
id2labels = {id[i]:labels[i] for i in range(len(labels))}
#==============Compute clusters===================

#==============MNL===================

def normalization_4(df,names):
  stat_dict = {}
  df_normalized = df.copy()
  for name in names:
    mean = np.array(df_normalized[[name]]).mean()
    std = np.array(df_normalized[[name]]).std()
    df_normalized[[name]] = (df_normalized[[name]]-mean)/std
    stat_dict[name] = (mean,std)

  return df_normalized, stat_dict

df_total = pd.read_csv(DATA_DIR)
ids = df_total["srch_id"]
labels = np.array([id2labels[i] for i in ids])
df_total["labels"] = labels

names = ["prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag"]
df_total,stat_dict = normalization_4(df_total,names)

df_normalized_list = []
for i in range(NUM_CLUSTERS):
  temp = df_total[df_total["labels"] == i].drop(columns=["srch_booking_window",	"srch_adults_count",	"srch_children_count",	"srch_room_count","srch_saturday_night_bool",'labels'])
  df_normalized_list.append(temp)

id_sets = []
for c in df_normalized_list:
  temp = set(c["srch_id"])
  id_sets.append(temp)

thetas = []
sum_value = 0
for i in labels_dict: 
  sum_value = sum_value+labels_dict[i]

for i in range(NUM_CLUSTERS):
  thetas.append(labels_dict[i]/sum_value)

#==============optimization===================
class  Optimizer_function():
  def __init__(self,id_cluster):
    self.id_cluster = id_cluster
  def rosen(self,beta):
    id_cluster = self.id_cluster
    beta0 = beta[0]
    beta_rest = beta[1:]
    objective = 0
    for id in id_sets[id_cluster]:
      temp_matrix = np.array(df_normalized_list[id_cluster][df_normalized_list[id_cluster]["srch_id"] == id])
      j = np.where(temp_matrix[:,-1] == 1)[0]   
      v_j = temp_matrix[j,1:-1]
      v_p = temp_matrix[:,1:-1]
      if len(v_j) != 0:
        linear = beta0+v_j@beta_rest 
      else:
        linear = 0 
      concave = np.log(np.sum(np.exp(beta0 + v_p@beta_rest))+1)
      T = linear-concave
      # print("T:",T)
      objective = objective + T
    #max -> min
    objective = -objective
    print("objective",objective)
    print("beta",beta)
    print("********************")
    return objective







beta_list = []
for i in range(NUM_CLUSTERS):
  beta = np.array([-1.74629566,  0.41212766,  0.1057882 ,  0.1008278 ,  0.02017485,
          0.04341198, -0.06984647, -1.33103003,  0.15948702])
  f = Optimizer_function(i)
  res = minimize(f.rosen, beta, method='Powell', options={'xtol': 0.0001,'disp': True})
  beta_list.append(res.x)

pd.DataFrame(beta_list).to_csv("beta[NUM_CLUSTERS="+str(NUM_CLUSTERS)+"].csv",header=False)
#==============optimization===================