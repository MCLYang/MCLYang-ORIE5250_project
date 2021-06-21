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
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB
from itertools import product
from termcolor import colored
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--NUM_CLUSTERS', type=int,default=2,help='NUM_CLUSTERS')
args = parser.parse_args()



DATA_DIR = "data.csv"
BETA_DIR = "beta[NUM_CLUSTERS="+str(args.NUM_CLUSTERS)+"].csv"

beta_list_pd = pd.read_csv(BETA_DIR,names=["beta_"+str(i) for i in range(9)])
beta = np.array(beta_list_pd)

NUM_CLUSTERS = len(beta)
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
_,stat_dict = normalization_4(df_total,names)

theta = []
sum_value = 0
for i in labels_dict: 
  sum_value = sum_value+labels_dict[i]

for i in range(NUM_CLUSTERS):
  theta.append(labels_dict[i]/sum_value)

#===========================Processing data1234===========================

def normalization(df,names,stat_dict):
  df_normalized = df.copy()
  for name in names:
    df_normalized[[name]]
    stat_dict[name][0]
    df_normalized[[name]] = (df_normalized[[name]]-stat_dict[name][0])/stat_dict[name][1]

  return df_normalized

def return_Revenue_mix(beta,theta,canidates,price,normalized_data):
  num_item = len(canidates)
  num_groups = len(beta)
  temp_data = normalized_data[canidates]
  temp_price = price[canidates]

  V = []
  for k in range(num_groups):
    V.append(np.exp(temp_data@beta[k,1:]+beta[k,0]))
  V = np.array(V).transpose()

  r = 0
  for k,b in enumerate(beta):
    r = r + theta[k]*(temp_price@V[:,k])/(1+np.sum(V[:,k]))
    
  return r

def return_canidate(beta,normalized_data):
  v = np.exp(normalized_data@beta[1:]+beta[0])
  exp_list = []
  temp1 = np.array(normalized_data)
  length = len(temp1)
  for i in range(length):  
    v_temp = v[:i+1]
    temp2 = normalized_data[:i+1]
    temp2 = temp1[:i+1]
    p = temp2[:,-2]
    exp = (p@v_temp)/(1+np.sum(v_temp))
    exp_list.append(exp)
  c = np.argmax(exp_list)
  # pdb.set_trace()
  return(np.arange(c+1))

def return_Revenue(beta,canidates,price,normalized_data):
  num_item = len(canidates)
  num_groups = len(beta)
  temp_data = normalized_data[canidates]
  temp_price = price[canidates]

  V = np.exp(temp_data@beta[1:]+beta[0])
  exp_rev = (temp_price@V)/(1+np.sum(V))
    
  return exp_rev

def get_model(data_np,price,M,num_groups = 2):
  m2 = gp.Model('MIP')
  num_products = len(data_np)
  cartesian_prod = list(product(range(num_products),range(num_groups)))
  
  V = []
  for k in range(num_groups):
    V.append(np.exp(data_np@beta[k,1:]+beta[k,0]))

  V = np.array(V).transpose()

  X = m2.addVars(num_products, vtype=GRB.BINARY, name='X')
  Y = m2.addVars(cartesian_prod, vtype=GRB.CONTINUOUS, name='Y')
  Z = m2.addVars(num_groups, vtype=GRB.CONTINUOUS, name='Z')


  m2.update()
  
  m2.setObjective(-gp.quicksum(theta[k]*Z[k] for k in range(num_groups)), GRB.MINIMIZE)
  m2.update()

  # m2.addConstrs(Z[k] == ((np.multiply(price,X)@(np.exp(data_np@beta[k,1:]+beta[k,0]) )) / (1+(X@ (np.exp(data_np@beta[k,1:]+beta[k,0]))))) for k in range(num_groups))
  # pdb.set_trace()
  m2.addConstrs(Z[k]+ gp.quicksum([V[j,k]*Y[(j,k)] for j in range(num_products)])== gp.quicksum([price[j]*V[j,k]*X[j] for j in range(num_products)]) for k in range(num_groups))
  
  m2.addConstrs(0<=Y[(j,k)] for j in range(num_products) for k in range(num_groups))
  m2.addConstrs(Y[(j,k)]<=M*X[j] for j in range(num_products) for k in range(num_groups))
  m2.addConstrs(-M*(1-X[j])+Z[k]<=Y[(j,k)] for j in range(num_products) for k in range(num_groups))
  m2.addConstrs(Y[(j,k)]<=Z[k] for j in range(num_products) for k in range(num_groups))
  m2.update()
  return m2,X


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED   = "\033[1;31m"  
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"

def assortment_opt(data_dir):
  normalization_colomns = ["prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag"]
  df1 = pd.read_csv(data_dir)
  df1 = df1.sort_values(by=['price_usd'], inplace=False,ascending = False)
  data = df1
  data_unnormalized = np.array(data)
  data_np = np.array(normalization(data,normalization_colomns,stat_dict))
  price = data_np[:,-2]
  M = price.max()
  m_data,X = get_model(data_np,price,M,num_groups = len(theta))#theta is gobal Var
  m_data.optimize()
  print("******************************")
  print("******************************")
  print("**********Assortment OPT,"+data_dir+",IP**********")
  print("******************************")
  print("******************************")

  displaye_item = []
  for facility in X.keys():
      if (abs(X[facility].x) > 1e-6):
          print(f"displaye #{facility} item.")
          displaye_item.append(int(facility))

  price = data_unnormalized[:,-2]
  normalized_data = data_np
  canidates_mix = displaye_item
  print("Suppose unknown the customers type, the displayed canidates are:", set(canidates_mix))
  r = return_Revenue_mix(beta,theta,canidates_mix,price,normalized_data)
  sys.stdout.write(bcolors.RED)

  print("Mixture Revenue:",r)
  sys.stdout.write(bcolors.RESET)

  for i in range(NUM_CLUSTERS):
    canidates = return_canidate(beta[i],data_np)
    print("Suppose known the customers from cluster"+str(i))
    print("Cluster center:")
    print(center_df.loc[i,:])
    print("the displayed canidates are:", set(canidates))
    r = return_Revenue(beta[i],canidates,price,data_np)
    sys.stdout.write(bcolors.BLUE)
    print("Revenue:",r)
    sys.stdout.write(bcolors.RESET)
    r = return_Revenue(beta[i],canidates_mix,price,data_np)
    sys.stdout.write(bcolors.RED)
    print("Revenue[Mixture canidates]:",r)
    sys.stdout.write(bcolors.RESET)


data_dirs = ["data1.csv","data2.csv","data3.csv","data4.csv"]
for data_dir in data_dirs:
  assortment_opt(data_dir)



  # canidates = return_canidate(beta[1],data_np)
  # print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
  # r = return_Revenue(beta[1],canidates,price,data_np)
  # print("Revenue:",r)
