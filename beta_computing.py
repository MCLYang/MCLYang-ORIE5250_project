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

NUM_CLUSTERS = 5
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


# df_normalized_list = []
# stat_dict_list = []
# for i in range(NUM_CLUSTERS):
#   df_normalized_temp,stat_dict_temp = normalization_4(clusters[i],names)
#   df_normalized_list.append(df_normalized_temp)
#   stat_dict_list.append(stat_dict_temp) 

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
pdb.set_trace()
#==============optimization===================


#==============MNL===================





# #results from P1-P4
# #================================
# stat_dict = {'price_usd': (141.19362259736354, 181.7693498769592),
#  'promotion_flag': (0.14480847531844532, 0.3519076310545035),
#  'prop_accesibility_score': (0.005993111516316034, 0.07718286163824864),
#  'prop_brand_bool': (0.7357279637145527, 0.44094481187898166),
#  'prop_location_score': (2.6171728460417363, 1.3481508978402192),
#  'prop_log_historical_price': (4.27631054382422, 1.7895033851701312),
#  'prop_review_score': (3.9876804632407246, 0.9071191629160908),
#  'prop_starrating': (3.157402505734957, 0.8571822862649962)}
# normalization_colomns = ["prop_starrating","prop_review_score","prop_brand_bool","prop_location_score","prop_accesibility_score","prop_log_historical_price","price_usd","promotion_flag"]
# late_beta = np.array([-1.53953067,  0.46527555,  0.09313614 , 0.11243684, 0.08595676,   0.02554587,   -0.03388866,   -1.6961064,0.19450294])
# late_theta = 0.45690687096001914
# early_beta = np.array([-1.9180075,  0.37792879,  0.12591464,  0.09239193, -0.02085044,  0.05785425,  -0.09397025,  -1.07208896,0.13451358])
# early_theta = 0.5430931290399809

# #early first
# theta = np.array([early_theta,late_theta])
# beta = np.array([early_beta,late_beta])

# #================================


# def normalization(df,names,stat_dict):
#   df_normalized = df.copy()
#   for name in names:
#     df_normalized[[name]]
#     stat_dict[name][0]
#     df_normalized[[name]] = (df_normalized[[name]]-stat_dict[name][0])/stat_dict[name][1]

#   return df_normalized

# df1 = pd.read_csv("data1.csv")
# df2 = pd.read_csv("data2.csv")
# df3 = pd.read_csv("data3.csv")
# df4 = pd.read_csv("data4.csv")
# df1 = df1.sort_values(by=['price_usd'], inplace=False,ascending = False)
# df2 = df2.sort_values(by=['price_usd'], inplace=False,ascending = False)
# df3 = df3.sort_values(by=['price_usd'], inplace=False,ascending = False)
# df4 = df4.sort_values(by=['price_usd'], inplace=False,ascending = False)


# def return_Revenue_mix(beta,theta,canidates,price,normalized_data):
#   num_item = len(canidates)
#   num_groups = len(beta)
#   temp_data = normalized_data[canidates]
#   temp_price = price[canidates]

#   V = []
#   for k in range(num_groups):
#     V.append(np.exp(temp_data@beta[k,1:]+beta[k,0]))
#   V = np.array(V).transpose()

#   r = 0
#   for k,b in enumerate(beta):
#     r = r + theta[k]*(temp_price@V[:,k])/(1+np.sum(V[:,k]))
    
#   return r

# def return_canidate(beta,normalized_data):
#   v = np.exp(normalized_data@beta[1:]+beta[0])
#   exp_list = []
#   temp1 = np.array(normalized_data)
#   length = len(temp1)
#   for i in range(length):  
#     v_temp = v[:i+1]
#     temp2 = normalized_data[:i+1]
#     temp2 = temp1[:i+1]
#     p = temp2[:,-2]
#     exp = (p@v_temp)/(1+np.sum(v_temp))
#     exp_list.append(exp)
#   c = np.argmax(exp_list)
#   # pdb.set_trace()
#   return(np.arange(c+1))

# def return_Revenue(beta,canidates,price,normalized_data):
#   num_item = len(canidates)
#   num_groups = len(beta)
#   temp_data = normalized_data[canidates]
#   temp_price = price[canidates]

#   V = np.exp(temp_data@beta[1:]+beta[0])
#   exp_rev = (temp_price@V)/(1+np.sum(V))
    
#   return exp_rev


# def get_model(data_np,price,M,num_groups = 2):
#   m2 = gp.Model('MIP')
#   num_products = len(data_np)
#   cartesian_prod = list(product(range(num_products),range(num_groups)))
  
#   V = []
#   for k in range(num_groups):
#     V.append(np.exp(data_np@beta[k,1:]+beta[k,0]))

#   V = np.array(V).transpose()

#   X = m2.addVars(num_products, vtype=GRB.BINARY, name='X')
#   Y = m2.addVars(cartesian_prod, vtype=GRB.CONTINUOUS, name='Y')
#   Z = m2.addVars(num_groups, vtype=GRB.CONTINUOUS, name='Z')


#   m2.update()
  
#   m2.setObjective(-gp.quicksum(theta[k]*Z[k] for k in range(num_groups)), GRB.MINIMIZE)
#   m2.update()

#   # m2.addConstrs(Z[k] == ((np.multiply(price,X)@(np.exp(data_np@beta[k,1:]+beta[k,0]) )) / (1+(X@ (np.exp(data_np@beta[k,1:]+beta[k,0]))))) for k in range(num_groups))
#   # pdb.set_trace()
#   m2.addConstrs(Z[k]+ gp.quicksum([V[j,k]*Y[(j,k)] for j in range(num_products)])== gp.quicksum([price[j]*V[j,k]*X[j] for j in range(num_products)]) for k in range(num_groups))
  
#   m2.addConstrs(0<=Y[(j,k)] for j in range(num_products) for k in range(num_groups))
#   m2.addConstrs(Y[(j,k)]<=M*X[j] for j in range(num_products) for k in range(num_groups))
#   m2.addConstrs(-M*(1-X[j])+Z[k]<=Y[(j,k)] for j in range(num_products) for k in range(num_groups))
#   m2.addConstrs(Y[(j,k)]<=Z[k] for j in range(num_products) for k in range(num_groups))
#   m2.update()
#   return m2,X


# data = df1
# data_unnormalized = np.array(data)
# data_np = np.array(normalization(data,normalization_colomns,stat_dict))
# price = data_np[:,-2]
# M = price.max()
# m_data,X = get_model(data_np,price,M,num_groups = len(theta))
# m_data.optimize()
# # pdb.set_trace()
# print("**********P5,data1,IP**********")
# displaye_item = []
# for facility in X.keys():
#     if (abs(X[facility].x) > 1e-6):
#         print(f"displaye #{facility} item.")
#         displaye_item.append(int(facility))

# price = data_unnormalized[:,-2]
# normalized_data = data_np
# canidates = displaye_item
# print("Suppose unknown the customers type, the displayed canidates are:", set(canidates))
# r = return_Revenue_mix(beta,theta,canidates,price,normalized_data)
# print("Revenue:",r)
# canidates = return_canidate(early_beta,data_np)
# print("Suppose known the customers is Type1, the displayed canidates are:", set(canidates))
# r = return_Revenue(early_beta,canidates,price,data_np)
# print("Revenue:",r)
# canidates = return_canidate(late_beta,data_np)
# print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
# r = return_Revenue(late_beta,canidates,price,data_np)
# print("Revenue:",r)


# data = df2
# data_unnormalized = np.array(data)
# data_np = np.array(normalization(data,normalization_colomns,stat_dict))
# price = data_np[:,-2]
# M = price.max()
# m_data,X = get_model(data_np,price,M,num_groups = len(theta))
# m_data.optimize()
# # pdb.set_trace()
# print("**********P5,data2,IP**********")
# displaye_item = []
# for facility in X.keys():
#     if (abs(X[facility].x) > 1e-6):
#         print(f"displaye #{facility} item.")
#         displaye_item.append(int(facility))

# price = data_unnormalized[:,-2]
# normalized_data = data_np
# canidates = displaye_item
# print("Suppose unknown the customers type, the displayed canidates are:", set(canidates))
# r = return_Revenue_mix(beta,theta,canidates,price,normalized_data)
# print("Revenue:",r)
# canidates = return_canidate(early_beta,data_np)
# print("Suppose known the customers is Type1, the displayed canidates are:", set(canidates))
# r = return_Revenue(early_beta,canidates,price,data_np)
# print("Revenue:",r)
# canidates = return_canidate(late_beta,data_np)
# print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
# r = return_Revenue(late_beta,canidates,price,data_np)
# print("Revenue:",r)

# data = df3
# data_unnormalized = np.array(data)
# data_np = np.array(normalization(data,normalization_colomns,stat_dict))
# price = data_np[:,-2]
# M = price.max()
# m_data,X = get_model(data_np,price,M,num_groups = len(theta))
# m_data.optimize()
# # pdb.set_trace()
# print("**********P5,data3,IP**********")
# displaye_item = []
# for facility in X.keys():
#     if (abs(X[facility].x) > 1e-6):
#         print(f"displaye #{facility} item.")
#         displaye_item.append(int(facility))

# price = data_unnormalized[:,-2]
# normalized_data = data_np
# canidates = displaye_item
# print("Suppose unknown the customers type, the displayed canidates are:", set(canidates))
# r = return_Revenue_mix(beta,theta,canidates,price,normalized_data)
# print("Revenue:",r)
# canidates = return_canidate(early_beta,data_np)
# print("Suppose known the customers is Type1, the displayed canidates are:", set(canidates))
# r = return_Revenue(early_beta,canidates,price,data_np)
# print("Revenue:",r)
# canidates = return_canidate(late_beta,data_np)
# print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
# r = return_Revenue(late_beta,canidates,price,data_np)
# print("Revenue:",r)


# data = df4
# data_unnormalized = np.array(data)
# data_np = np.array(normalization(data,normalization_colomns,stat_dict))
# price = data_np[:,-2]
# M = price.max()
# m_data,X = get_model(data_np,price,M,num_groups = len(theta))
# m_data.optimize()
# # pdb.set_trace()
# print("**********P5,data4,IP**********")
# displaye_item = []
# for facility in X.keys():
#     if (abs(X[facility].x) > 1e-6):
#         print(f"displaye #{facility} item.")
#         displaye_item.append(int(facility))

# price = data_unnormalized[:,-2]
# normalized_data = data_np
# canidates = displaye_item
# print("Suppose unknown the customers type, the displayed canidates are:", set(canidates))
# r = return_Revenue_mix(beta,theta,canidates,price,normalized_data)
# print("Revenue:",r)
# canidates = return_canidate(early_beta,data_np)
# print("Suppose known the customers is Type1, the displayed canidates are:", set(canidates))
# r = return_Revenue(early_beta,canidates,price,data_np)
# print("Revenue:",r)
# canidates = return_canidate(late_beta,data_np)
# print("Suppose known the customers is Type2, the displayed canidates are:", set(canidates))
# r = return_Revenue(late_beta,canidates,price,data_np)
# print("Revenue:",r)




# late_beta = np.array([-1.53953067,  0.46527555,  0.09313614 , 0.11243684, 0.08595676,   0.02554587,   -0.03388866,   -1.6961064,0.19450294])
# late_theta = 0.45690687096001914
# early_beta = np.array([-1.9180075,  0.37792879,  0.12591464,  0.09239193, -0.02085044,  0.05785425,  -0.09397025,  -1.07208896,0.13451358])
# early_theta = 0.5430931290399809

# print("late_beta norm: ",np.linalg.norm(late_beta))
# print("early_beta norm: ",np.linalg.norm(early_beta))