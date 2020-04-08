#!/usr/bin/env python
# coding: utf-8

# April 8, 2020

# ## Setup
# Import packages and necessary files. 

# In[1]:


new_data_file = "200204_UKBiobank_ABCGene.train.tsv"


# In[264]:


# import packages
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression,LinearRegression,LogisticRegressionCV
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,precision_recall_curve,average_precision_score,precision_recall_fscore_support, precision_score, recall_score,roc_auc_score
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import rankdata
from inspect import signature
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
# import tensorflow as tf
#import keras


# In[265]:


#import keras


# In[6]:


# create functions for lists later
def flattest_list(input_list):
    flat_list_0 = []
    for sublist in input_list:
        for item in sublist:
            for widdle in item:
                flat_list_0.append(widdle)
    return(flat_list_0)

def flatten_list(input_list):
    flat_list_0 = []
    for sublist in input_list:
        for item in sublist:
            flat_list_0.append(item)
    return(flat_list_0)

def flattish_list(input_list):
    flat_list1 = []
    for sublist in input_list:
        flat_list1.append(sublist)
    return(flat_list1)


# In[7]:


data = pd.read_csv(new_data_file,delimiter = "\t",header='infer') 


# In[8]:


data.shape


# In[9]:


# make ids for credible sets
flat_creds = data["CredibleSet"]

def transform_into_ids(input_list):
    id_list = []
    id_number = 0
    id_dict = {}
    for element in input_list:
        if element not in id_dict:
            id_dict[element] = id_number
            id_number += 1
        id_list.append(id_dict[element])
    return id_list

creds_id = transform_into_ids(flat_creds)

data["CredId"] = creds_id


# In[10]:


ids_full_list = list(np.unique(data.CredId))


# In[11]:


cred_lens = []
for i in ids_full_list:
    cred_length = len(data[data["CredId"]==i])
    for j in range(cred_length):
        cred_lens.append(cred_length)


# In[12]:


data["CredSetLength"] = cred_lens


# For length purposes, change FMOverlap_Enriched to just Enriched, change GeneBodyDistanceToBestSNP to GeneBodyDistToBestSNP

# In[13]:


pd.read_csv(new_data_file,delimiter = "\t",header='infer').columns


# In[14]:


new_colnames_for_data = ['CredibleSet', 'TargetGene', 'Disease',
       'CodingSpliceOrPromoterVariants', 'PromoterDistanceToBestSNP',
       'IsClosestGeneToBestSNP', 'ConnectionStrengthRank', 'CellTypeCountRank',
       'ContactRank', 'DistanceRank', 'GeneBodyDistToBestSNP',
       'DistanceToGeneBodyRank', 'MaxABC', 'ContactMean',
       'MaxABC_Enriched', 'ConnectionStrengthRank_Enriched',
       'CodingPP50', 'TotalEnhancers', 'TotalBases', 'CellTypesWithPrediction',
       'TotalDistalPromoters', 'EnhancersPerCellType',
       'EnhancerBasesPerCellType', 'DistalPromotersPerCellType',
       'AverageEnhancerDistance', 'ABC9_G', 'ATAC_distal_prob', 'EDS_binary',
       'Expecto_MVP', 'Master_Regulator', 'PCHiC_binary', 'PPI_Enhancer',
       'SEG_GTEx', 'TF_genes_curated', 'Trans_Reg_genes', 'eQTL_CTS_prob',
       'pLI_genes', 'GeneCS_MaxABC', 'GeneCS_Enriched', 'powerlaw',
       'GeneCS_Distance','GeneCS_MaxABC_n5', 'GeneCS_FMOverlapEnriched_n5',
       'GeneCS_Distance_n5', 'GeneCS_MaxABC_n10',
       'GeneCS_FMOverlapEnriched_n10', 'GeneCS_Distance_n10','nNearbyTrues', 'CredId','CredSetLength']


# In[15]:


data.columns = new_colnames_for_data


# ## Train-Test Split by Credible Set

# In[16]:


train_size = int(len(ids_full_list)*.8)+1
test_size = len(ids_full_list) - train_size
print(train_size,test_size)


# In[17]:


random.seed(123)
random.shuffle(ids_full_list)
train_ids = ids_full_list[0:train_size]
test_ids = ids_full_list[train_size:len(ids_full_list)]


# In[18]:


train = data[data['CredId'].isin(train_ids)]
val = data[data['CredId'].isin(test_ids)]


# In[154]:


# feature lists
xvars_condensed = ['MaxABC_Enriched','ConnectionStrengthRank_Enriched',
               'GeneCS_Enriched','TotalEnhancers','TotalBases',
              'CellTypesWithPrediction','TotalDistalPromoters','EnhancersPerCellType',
              'EnhancerBasesPerCellType', 'DistalPromotersPerCellType',
             'CredSetLength','DistanceRank']
xvars= ['PromoterDistanceToBestSNP',
       'IsClosestGeneToBestSNP', 
         'ConnectionStrengthRank', 'CellTypeCountRank',
       'ContactRank', 'DistanceRank', 'GeneBodyDistToBestSNP',
       'DistanceToGeneBodyRank',
         'MaxABC', 'ContactMean',
       'MaxABC_Enriched', 'ConnectionStrengthRank_Enriched',
         'GeneCS_Enriched',
       'TotalEnhancers', 'TotalBases', 'CellTypesWithPrediction',
       'TotalDistalPromoters', 'EnhancersPerCellType',
       'EnhancerBasesPerCellType', 'DistalPromotersPerCellType',
       'AverageEnhancerDistance','CredSetLength','GeneCS_Distance']
xvars_nodist = ['ConnectionStrengthRank', 'CellTypeCountRank',
       'ContactRank', 
         'MaxABC', 'ContactMean',
       'MaxABC_Enriched', 'ConnectionStrengthRank_Enriched',
                'GeneCS_Enriched',
       'TotalEnhancers', 'TotalBases', 'CellTypesWithPrediction',
       'TotalDistalPromoters', 'EnhancersPerCellType',
       'EnhancerBasesPerCellType', 'DistalPromotersPerCellType',
                'CredSetLength']
xvars_abc_JME = ['MaxABC_Enriched','ConnectionStrengthRank_Enriched',
               'GeneCS_Enriched','TotalEnhancers','TotalBases',
              'CellTypesWithPrediction','TotalDistalPromoters','EnhancersPerCellType',
              'EnhancerBasesPerCellType', 'DistalPromotersPerCellType',
              'CredSetLength']
xvars_dist = ['PromoterDistanceToBestSNP',
       'IsClosestGeneToBestSNP','DistanceRank', 'GeneBodyDistToBestSNP',
       'DistanceToGeneBodyRank','AverageEnhancerDistance','GeneCS_Distance']
yvar = "CodingPP50"


# In[155]:


X_train = train[xvars]
X_val = val[xvars]
X_train_nodist = train[xvars_abc_JME]
X_val_nodist = val[xvars_abc_JME]
y_train = train[yvar]
y_val = val[yvar]
X = data[xvars]
Y = data[yvar]


# In[156]:


X_tsmote, y_tsmote = SMOTE(random_state = 2, k_neighbors=25).fit_sample(X_train, y_train)


# ## Exploratory Data Analysis

# Summary Statistics

# In[129]:


X_train.describe().transpose()


# Correlation Matrix

# In[118]:


combined_xy_train = data[xvars+[yvar]]
corrs_all = combined_xy_train.corr()
#corrs_all.to_csv (r'X_train_correlation_matrix.csv', index = True, header=True)


# In[120]:


plt.close()
plt.figure(figsize=(8,7))
plt.subplots_adjust(left=0.33, right=0.97, top=0.94, bottom=0.4)
heat_map = sns.heatmap(corrs_all)
plt.title("Correlation Heatmap")
plt.show() #savefig("final_figs/corr_heatmap.jpeg")
plt.close()


# In[121]:


xvars_uncorr = ['PromoterDistanceToBestSNP',
       'IsClosestGeneToBestSNP', 
         'ConnectionStrengthRank', 'DistanceRank', 
         'MaxABC', 'ContactMean',
         'GeneCS_Enriched',
       'TotalEnhancers', 'TotalBases',
       'EnhancersPerCellType',
       'EnhancerBasesPerCellType', 'DistalPromotersPerCellType',
       'AverageEnhancerDistance','CredSetLength','GeneCS_Distance']


# Scatterplot of Interest

# In[26]:


plt.close()
plt.figure(figsize=(5,5))
plt.scatter(train["ConnectionStrengthRank"],train["MaxABC"])
plt.xlabel("ABC Rank")
plt.ylabel("Max ABC")
#plt.savefig("final_figs/ABCrank_vs_Distrank.png")
plt.show()
plt.close()


# Histograms and KDEs

# In[27]:


for col in X.columns:
    plt.hist(X_train[col].values,bins=30)
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("final_figs/hists/"+col+".png")
    plt.close()


# In[27]:


for col in X.columns:
    ax = sns.kdeplot(X_train[col].values)
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("final_figs/kde/"+col+".png")
    plt.close()


# In[22]:


plt.close()
sns.pairplot(combined_xy_train) #, hue=yvar,kind="reg",diag_kind="kde", markers="+")    
plt.title("Pairplot")
plt.legend()
plt.savefig("final_figs/pairplot.jpeg")
#plt.show()


# In[29]:


for col in X.columns:
    ax = sns.kdeplot(data[data[yvar]==False][col].values,label="False")
    ax = sns.kdeplot(data[data[yvar]==True][col].values,label="True")
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("final_figs/kde_tf/"+col+".png")
    plt.close()


# In[106]:


plt.close()
fig, axs = plt.subplots(5, 5)
fig.set_size_inches(12, 12)
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.03, top=0.96, wspace=0.5, hspace=0.5)
first_col = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
second_col = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
i = 0
for col in X.columns:
    sns.kdeplot(data[data[yvar]==False][col].values,label="False",ax = axs[first_col[i],second_col[i]])
    sns.kdeplot(data[data[yvar]==True][col].values,label="True",ax = axs[first_col[i],second_col[i]])
    axs[first_col[i],second_col[i]].set_title(col)
    i = i+1

axs[0,0].set_xticklabels(["","0MM","0.5MM","1MM","1.5MM"])    


for ax in axs.flat:
    ax.set(xlabel='Value', ylabel='Density')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

axs[4,2].set_xlabel("Value")
axs[2,0].set_ylabel("Density")

plt.savefig("final_figs/kde_tf/all_kde_test.jpeg")


# ## Upsampling

# In[157]:


# Separate majority and minority classes
df_majority = train[train.CodingPP50==False]
df_minority = train[train.CodingPP50==True]
ratio = int(df_majority.shape[0]/df_minority.shape[0])
 
# # Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class # fix
                                 random_state=123) # reproducible results
 
# # Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# # Display new class counts
print(df_upsampled.CodingPP50.value_counts())
# True     5061
# False    5061
# Name: CodingPP50, dtype: int64

X_train_up = df_upsampled[xvars]
X_train_up_nodist = df_upsampled[xvars_nodist]
y_train_up = df_upsampled[yvar]


# In[23]:


df_minority.shape[0],df_majority.shape[0]


# In[24]:


print("Minority %: ",df_minority.shape[0]/(df_majority.shape[0]+df_minority.shape[0])*100)


# ## Baseline Models

# In[158]:


def big_ranker(rowname,frame_given,n):
    output_vect = []
    for index,row in frame_given.iterrows():
        if row[rowname] <= n:
            output_vect.append(1)
        else:
            output_vect.append(0)
    return(output_vect)

def one_or_two(rowname,frame_given):
    one_or_2 = []
    for index, row in frame_given.iterrows():
        if row[rowname] == 1:
            one_or_2.append(1)
        elif row[rowname] == 2:
            one_or_2.append(1)
        else:
            one_or_2.append(0)  
    return(one_or_2)

def just_one(rowname,frame_given):
    top_one = []
    for index, row in frame_given.iterrows():
        if row[rowname] == 1:
            top_one.append(1)
        else:
            top_one.append(0)  
    return(top_one)    


# In[159]:


def all_rank_predictions(rowname, train_up_X = X_train_up, #val_up_X = X_val_up,
                         test_X = X_val,all_X = X,
                        train_up_Y = y_train_up,#val_up_Y = y_val_up,
                         test_Y = y_val,all_Y = Y):
    
    d = {'1 train_upsampled': [accuracy_score(one_or_two(rowname,train_up_X),train_up_Y),
                                    accuracy_score(just_one(rowname,train_up_X),train_up_Y)], 
#          '2 val_upsampled': [accuracy_score(one_or_two(rowname,val_up_X),val_up_Y),
#                                     accuracy_score(just_one(rowname,val_up_X),val_up_Y)],  
         '3 test_holdout': [accuracy_score(one_or_two(rowname,test_X),test_Y),
                                    accuracy_score(just_one(rowname,test_X),test_Y)],   
         '4 full_X': [accuracy_score(one_or_two(rowname,all_X),all_Y),
                                    accuracy_score(just_one(rowname,all_X),all_Y)]}
    dic_df = pd.DataFrame(data=d)
    dic_df.index = ['Rank 1-2', 'Rank 1'] 
    return(dic_df)

def all_rank_prs(rowname, labels_list,precision_list,recall_list,
                 test_X = X_val, all_X = X,
                         test_Y = y_val, all_Y = Y):
    
    # calculate predictions from those ranks
    test_preds_one_or_two = one_or_two(rowname,test_X)
    test_preds_one = just_one(rowname,test_X)
    all_preds_one_or_two = one_or_two(rowname,all_X)
    all_preds_one = just_one(rowname,all_X)

    # calculate precision and recall for the predictions
    l_12_t = '1 PR 1 or 2, test'
    p_12_t = precision_score(test_Y,test_preds_one_or_two)
    r_12_t = recall_score(test_Y,test_preds_one_or_two)
    labels_list.append(method+' 1/2')
    precision_list.append(p_12_t)
    recall_list.append(r_12_t)
    l_1_t = '2 PR 1, test'
    p_1_t = precision_score(test_Y,test_preds_one)
    r_1_t = recall_score(test_Y,test_preds_one)
    labels_list.append(method+' 1')
    precision_list.append(p_1_t)
    recall_list.append(r_1_t)
    l_12_a = '3 PR 1 or 2, all'
    p_12_a = precision_score(all_Y,all_preds_one_or_two)
    r_12_a = recall_score(all_Y,all_preds_one_or_two)
#     labels_list.append(l_12_a)
#     precision_list.append(p_12_a)
#     recall_list.append(r_12_a)
    l_1_a = '4 PR 1, all'
    p_1_a = precision_score(all_Y,all_preds_one)
    r_1_a = recall_score(all_Y,all_preds_one)
#     labels_list.append(l_1_a)
#     precision_list.append(p_1_a)
#     recall_list.append(r_1_a)
    
    # put into dictionary to print nice
    d = {l_12_t: [p_12_t,r_12_t], 
         l_1_t: [p_1_t,r_1_t],   
         l_12_a: [p_12_a,r_12_a],     
         l_1_a:  [p_1_a,r_1_a]}
    dic_df = pd.DataFrame(data=d)
    dic_df.index = ['Precision', 'Recall'] 
    print(dic_df)
    return(labels_list,precision_list,recall_list)
    


# In[27]:


methods = ['ConnectionStrengthRank','ContactRank','DistanceRank','DistanceToGeneBodyRank','ConnectionStrengthRank_Enriched']
for method in methods:
    print(method, "accuracy scores")
    print(all_rank_predictions(method))


# In[160]:


methods = ['ConnectionStrengthRank','ContactRank','DistanceRank','DistanceToGeneBodyRank','ConnectionStrengthRank_Enriched']
labels_list = []
precision_list = []
recall_list = []
    
for method in methods:
    print(method, "PR scores")
    labels_list,precision_list,recall_list = all_rank_prs(method,labels_list,precision_list,recall_list)


# In[29]:


plt.close()
for i in range(len(precision_list)):
    plt.plot(recall_list[i],precision_list[i], 'bo')
    plt.text(recall_list[i] * (1 + 0.01),precision_list[i] * (1 + 0.01),  labels_list[i], fontsize=12)
plt.title("PR values in holdout test set")
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.show()


# Predictions for thesis writeup

# In[30]:


# naive predictions, just predict 0
#naive_preds = np.repeat(0, len(train_up_X))

print("train")
print(precision_score(y_train,np.repeat(0, len(y_train))))
print(recall_score(y_train,np.repeat(0, len(y_train))))
print(accuracy_score(y_train,np.repeat(0, len(y_train))))
print("train up")
print(precision_score(y_train_up,np.repeat(0, len(y_train_up))))
print(recall_score(y_train_up,np.repeat(0, len(y_train_up))))
print(accuracy_score(y_train_up,np.repeat(0, len(y_train_up))))
print("val")
print(precision_score(y_val,np.repeat(0, len(y_val))))
print(recall_score(y_val,np.repeat(0, len(y_val))))
print(accuracy_score(y_val,np.repeat(0, len(y_val))))


# In[31]:


#distance rank
dist_train_preds = just_one("DistanceRank",X_train)
dist_train_up_preds = just_one("DistanceRank",X_train_up)
dist_val_preds = just_one("DistanceRank",X_val)

print("train")
print(precision_score(y_train,dist_train_preds))
print(recall_score(y_train,dist_train_preds))
print(accuracy_score(y_train,dist_train_preds))
print("train up")
print(precision_score(y_train_up,dist_train_up_preds))
print(recall_score(y_train_up,dist_train_up_preds))
print(accuracy_score(y_train_up,dist_train_up_preds))
print("val")
print(precision_score(y_val,dist_val_preds))
print(recall_score(y_val,dist_val_preds))
print(accuracy_score(y_val,dist_val_preds))


# ### k Nearest Neighbors

# In[161]:


knn = KNeighborsClassifier(weights="distance",n_neighbors=25).fit(X_train,y_train)
train_fits_knn = knn.predict(X_train)
val_fits_knn = knn.predict(X_val)

# print out accuracy
print("Training Accuracy:",accuracy_score(train_fits_knn,y_train))
print("Validation Accuracy:",accuracy_score(val_fits_knn,y_val))
print("Training Precision:",precision_score(train_fits_knn,y_train))
print("Validation Precision:",precision_score(val_fits_knn,y_val))
print("Training Recall:",recall_score(train_fits_knn,y_train))
print("Validation Recall:",recall_score(val_fits_knn,y_val))


# In[162]:


knn_up = KNeighborsClassifier(weights="distance",n_neighbors=25).fit(X_train_up,y_train_up)
train_fits_knn_up = knn_up.predict(X_train_up)
val_fits_knn_up = knn_up.predict(X_val)

# print out accuracy
print("Training Accuracy:",accuracy_score(train_fits_knn_up,y_train_up))
print("Validation Accuracy:",accuracy_score(val_fits_knn_up,y_val))
print("Training Precision:",precision_score(train_fits_knn_up,y_train_up))
print("Validation Precision:",precision_score(val_fits_knn_up,y_val))
print("Training Recall:",recall_score(train_fits_knn_up,y_train_up))
print("Validation Recall:",recall_score(val_fits_knn_up,y_val))


# In[163]:


knn_smote = KNeighborsClassifier(weights="distance",n_neighbors=100).fit(X_tsmote,y_tsmote)
train_fits_knn_up = knn_smote.predict(X_tsmote)
val_fits_knn_up = knn_smote.predict(X_val)

# print out accuracy
print("Training Accuracy:",accuracy_score(train_fits_knn_up,y_train_up))
print("Validation Accuracy:",accuracy_score(val_fits_knn_up,y_val))
print("Training Precision:",precision_score(train_fits_knn_up,y_train_up))
print("Validation Precision:",precision_score(val_fits_knn_up,y_val))
print("Training Recall:",recall_score(train_fits_knn_up,y_train_up))
print("Validation Recall:",recall_score(val_fits_knn_up,y_val))


# In[35]:


X_train_train, X_test_train, y_train_train, y_test_train = train_test_split(
     X_train, y_train, test_size=0.2, random_state=42)


# In[36]:


ks = range(1, 100) # Grid of k's
scores_train = [] # R2 scores
scores_val = [] # R2 scores
for k in ks:
    knnreg = KNeighborsClassifier(weights="distance",n_neighbors=k) # Create KNN model
    knnreg.fit(X_train_train, y_train_train) # Fit the model to training data
    score_train = accuracy_score(knnreg.predict(X_train_train), y_train_train) # Calculate R^2 score
    #score_train = knnreg.score(X_train_train, y_train_train) # Calculate R^2 score
    scores_train.append(score_train)
    score_val = accuracy_score(knnreg.predict(X_test_train), y_test_train) # Calculate R^2 score
    #score_val = knnreg.score(X_test_train, y_test_train) # Calculate R^2 score
    scores_val.append(score_val)


# In[37]:


# Plot
fig, ax = plt.subplots(1,2, figsize=(20,6))
ax[0].plot(ks, scores_train,'o-')
ax[0].set_xlabel(r'$k$')
ax[0].set_ylabel(r'Precision$')
ax[0].set_title(r'Train Precision')
ax[1].plot(ks, scores_val,'o-')
ax[1].set_xlabel(r'$k$')
ax[1].set_ylabel(r'Precision')
ax[1].set_title(r'Validation Precision')
plt.show()


# ### Logistic Regression

# In[164]:


# logistic regression
lr1 = LogisticRegression(penalty='l2').fit(X_train,y_train)
train_fits_l1 = lr1.predict(X_train)
val_fits_l1 = lr1.predict(X_val)

# print out accuracy
print("Training Accuracy:",accuracy_score(train_fits_l1,y_train))
print("Validation Accuracy:",accuracy_score(val_fits_l1,y_val))


# In[39]:


labels_list.append('Logistic regression')
precision_list.append(precision_score(y_val,val_fits_l1))
recall_list.append(recall_score(y_val,val_fits_l1))


# In[165]:


# with balanced classes
# logistic regression with L2 penalty
lr2 = LogisticRegression().fit(X_train_up,y_train_up)
train_fits_l2 = lr2.predict(X_train_up)
#val_fits_l2 = lr2.predict(X_val_up)
valtest_fits_l2 = lr2.predict(X_val)


# print out accuracy
print("Training Accuracy:",accuracy_score(train_fits_l2,y_train_up))
#print("Validation Accuracy:",accuracy_score(val_fits_l2,y_val_up))
print("Validation/Test Accuracy:",accuracy_score(valtest_fits_l2,y_val))


# In[41]:


labels_list.append('Logistic regression up')
precision_list.append(precision_score(y_val,valtest_fits_l2))
recall_list.append(recall_score(y_val,valtest_fits_l2))


# ### Decision Tree

# In[42]:


# classify by depth
def treeClassifierByDepth(depth, X_traint, y_traint, cvt = 5):
    model = DecisionTreeClassifier(max_depth=depth).fit(X_traint, y_traint)
    return cross_val_score(model, X_traint, y_traint, cv = cvt)


# In[43]:


# holdout set performance
model_dec_tree = DecisionTreeClassifier(max_depth=15,random_state=1).fit(X_train, y_train)
train_predictions = model_dec_tree.predict(X_train)
train_score = accuracy_score(y_train, train_predictions)
val_dectree_predictions_test = model_dec_tree.predict(X_val)
val_score_test = model_dec_tree.score(X_val, y_val)
print("Training accuracy: ", train_score)
print("Test/Validation accuracy: ", val_score_test)


# In[44]:


labels_list.append('Decision tree')
precision_list.append(precision_score(y_val,val_dectree_predictions_test))
recall_list.append(recall_score(y_val,val_dectree_predictions_test))


# In[45]:


# holdout set performance
model_dec_tree_up = DecisionTreeClassifier(max_depth=1).fit(X_train_up, y_train_up)
train_predictions = model_dec_tree_up.predict(X_train_up)
train_score = accuracy_score(y_train_up, train_predictions)
val_dectree_predictions_test = model_dec_tree_up.predict(X_val)
val_score_test = model_dec_tree_up.score(X_val, y_val)
print("Training accuracy: ", train_score)
print("Test/Validation accuracy: ", val_score_test)


# In[46]:


labels_list.append('Decision tree up')
precision_list.append(precision_score(y_val,val_dectree_predictions_test))
recall_list.append(recall_score(y_val,val_dectree_predictions_test))


# ### Random Forest

# In[285]:


# regular
rf = RandomForestClassifier(random_state=1, class_weight = {0:1,1:50},criterion="gini").fit(X_train, y_train)
rf_train_preds = rf.predict(X_train)
rf_valtest_preds = rf.predict(X_val)
print("Training Accuracy:",accuracy_score(rf_train_preds, y_train))
print("Validation/Test Accuracy:",accuracy_score(rf_valtest_preds, y_val))
print("Training Precision:",precision_score(rf_train_preds, y_train))
print("Validation/Test Precision:",precision_score(rf_valtest_preds, y_val))
print("Training Recall:",recall_score(rf_train_preds, y_train))
print("Validation/Test Recall:",recall_score(rf_valtest_preds, y_val))


# In[48]:


labels_list.append('Random forest')
precision_list.append(precision_score(y_val,rf_valtest_preds))
recall_list.append(recall_score(y_val,rf_valtest_preds))


# In[286]:


# for upsampled
rf2 = RandomForestClassifier(random_state=5,criterion="gini").fit(X_train_up, y_train_up)
rf2_train_preds = rf2.predict(X_train_up)
rf2_valtest_preds = rf2.predict(X_val)
print("Training Accuracy:",accuracy_score(rf2_train_preds, y_train_up))
print("Validation/Test Accuracy:",accuracy_score(rf2_valtest_preds, y_val))
print("Training Precision:",precision_score(rf2_train_preds, y_train_up))
print("Validation/Test Precision:",precision_score(rf2_valtest_preds, y_val))
print("Training Recall:",recall_score(rf2_train_preds, y_train_up))
print("Validation/Test Recall:",recall_score(rf2_valtest_preds, y_val))


# In[269]:


labels_list.append('Random forest up')
precision_list.append(precision_score(y_val,rf2_valtest_preds))
recall_list.append(recall_score(y_val,rf2_valtest_preds))


# In[168]:


# for upsampled SMOTE
rf_smote = RandomForestClassifier(random_state=1,criterion="gini").fit(X_tsmote, y_tsmote)
rf_smote_train_preds = rf_smote.predict(X_tsmote)
rf_smote_valtest_preds = rf_smote.predict(X_val)
print("Training Accuracy:",accuracy_score(rf_smote_train_preds, y_tsmote))
print("Validation/Test Accuracy:",accuracy_score(rf_smote_valtest_preds, y_val))
print("Training Precision:",precision_score(rf_smote_train_preds, y_tsmote))
print("Validation/Test Precision:",precision_score(rf_smote_valtest_preds, y_val))
print("Training Recall:",recall_score(rf_smote_train_preds, y_tsmote))
print("Validation/Test Recall:",recall_score(rf_smote_valtest_preds, y_val))


# In[169]:


# for nodist
rf_abc = RandomForestClassifier(random_state=1,criterion="gini").fit(X_train_up[xvars_abc_JME], y_train_up)
rf_abc_train_preds = rf_smote.predict(X_train_up)
rf_abc_valtest_preds = rf_smote.predict(X_val)
print("Training Accuracy:",accuracy_score(rf_abc_train_preds, y_train_up))
print("Validation/Test Accuracy:",accuracy_score(rf_abc_valtest_preds, y_val))
print("Training Precision:",precision_score(rf_abc_train_preds, y_train_up))
print("Validation/Test Precision:",precision_score(rf_abc_valtest_preds, y_val))
print("Training Recall:",recall_score(rf_abc_train_preds, y_train_up))
print("Validation/Test Recall:",recall_score(rf_abc_valtest_preds, y_val))


# In[54]:


## for tuning
# Create the parameter grid based on the results of random search
param_grid = {
'max_depth': [300,500,800],
'max_features': [4, 5, 8],
'min_samples_leaf': [2, 3, 5],
'min_samples_split': [5, 8, 10],
'n_estimators': [50, 100, 200]
}
# Create a based model
rf_cv = RandomForestClassifier()
# Instantiate the grid search model
grid_search_transform = GridSearchCV(estimator = rf_cv, param_grid = param_grid,
cv = 5, n_jobs = -1, verbose = 1, scoring = 'precision', return_train_score=True)
# Fit the grid search to the data
grid_search_transform.fit(X_tsmote, y_tsmote)
grid_search_transform.best_params_


# In[170]:


rf_cv = RandomForestClassifier(max_depth= 300,
                               max_features = 4,
                               min_samples_leaf = 2,
                               min_samples_split = 5,
                               n_estimators = 200).fit(X_tsmote,y_tsmote)
rf_cv_train_preds = rf_cv.predict(X_train)
rf_cv_valtest_preds = rf_cv.predict(X_val)
print("Training Accuracy:",accuracy_score(rf_cv_train_preds, y_train))
print("Validation/Test Accuracy:",accuracy_score(rf_cv_valtest_preds, y_val))
print("Training Precision:",precision_score(rf_cv_train_preds, y_train))
print("Validation/Test Precision:",precision_score(rf_cv_valtest_preds, y_val))
print("Training Recall:",recall_score(rf_cv_train_preds, y_train))
print("Validation/Test Recall:",recall_score(rf_cv_valtest_preds, y_val))


# In[270]:


val["rf_preds"] = rf2.predict_proba(X_val)[:, 1]
val["rf_preds_smote"] = rf_smote.predict_proba(X_val)[:, 1]
val["rf_preds_abc"] = rf_abc.predict_proba(X_val[xvars_abc_JME])[:, 1]
val["rf_preds_cv"] = rf_cv.predict_proba(X_val[xvars])[:, 1]


# In[233]:


## make a function to normalize by credible set
def probs_by_credset(dataframe_cid,model_col,id_list_here):
    smartest_rf_proba = []
    for i in range(len(id_list_here)):
        current_credset = dataframe_cid[dataframe_cid["CredId"]==i]
        total_prob = 0
        for index, row in current_credset.iterrows():
            total_prob = total_prob + row[model_col]
        for index, row in current_credset.iterrows():
            smartest_rf_proba.append(row[model_col]/total_prob)
    return(smartest_rf_proba)    


# In[271]:


smartest_rf_probas = probs_by_credset(val,"rf_preds",ids_full_list)
smartest_rf_probas_smote = probs_by_credset(val,"rf_preds_smote",ids_full_list)
smartest_rf_probas_abc = probs_by_credset(val,"rf_preds_abc",ids_full_list)
smartest_rf_probas_cv = probs_by_credset(val,"rf_preds_cv",ids_full_list)


# In[272]:


val["smart_rf"] = smartest_rf_probas
val["smart_rf_smote"] = smartest_rf_probas_smote
val["smart_rf_abc"] = smartest_rf_probas_abc
val["smart_rf_cv"] = smartest_rf_probas_cv


# # Adaboost

# In[175]:


plt.close()
abc = AdaBoostClassifier(n_estimators=300,random_state=1)
abc.fit(X_train_up, y_train_up)
abc_predicts_train = list(abc.staged_score(X_train_up, y_train_up))
plt.plot(abc_predicts_train, label = "train");
# staged_score test to plot
abc_predicts_test = list(abc.staged_score(X_val,y_val))
plt.plot(abc_predicts_test, label = "test");
plt.legend()
plt.title("AdaBoost Classifier Accuracy")
plt.xlabel("Iterations")
plt.show()
#plt.savefig("final_figs/adaboost.png")
plt.close()
print("Maximum training accuracy is "+str(max(abc_predicts_train))+" at "+str(abc_predicts_train.index(max(abc_predicts_train)))+" iterations\n" + 
     "Maximum validation accuracy is "+str(max(abc_predicts_test))+" at "+str(abc_predicts_test.index(max(abc_predicts_test)))+" iterations \n")


# In[147]:


guess = abc.predict(X_val)
print(accuracy_score(guess,y_val), precision_score(guess,y_val), recall_score(guess,y_val))


# In[176]:


val["dist_preds_1"] = just_one("DistanceRank",X_val)
val["abc_preds_1"] = just_one("ConnectionStrengthRank",X_val)
val["boosted_preds"] = abc.predict_proba(X_val)[:, 1]


# ## PR curve

# In[149]:


recs_val = []
precs_val = []

for i in range(30):
    preds_i = big_ranker("DistanceRank",X_val,i)
    recs_val.append(recall_score(y_val,preds_i))
    precs_val.append(precision_score(y_val,preds_i))


# In[273]:


#### CALCULATE THESE ONCE TO MAKE THE PLOTS

# lr1
pos_probs_lr1 = lr1.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_lr1, recall_lr1, _ = precision_recall_curve(y_val, pos_probs_lr1)

## lr2
pos_probs_lr2 = lr2.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_lr2, recall_lr2, _ = precision_recall_curve(y_val, pos_probs_lr2)


## rf
pos_probs_rf = rf.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_rf, recall_rf, _ = precision_recall_curve(y_val, pos_probs_rf)


## rf2
pos_probs_rf2 = rf2.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_rf2, recall_rf2, _ = precision_recall_curve(y_val, pos_probs_rf2)

## rf2s
# calculate model precision-recall curve, smartest_rf_probas are calculated earlier.
precision_rf2s, recall_rf2s, thresholds_rf2 = precision_recall_curve(y_val, smartest_rf_probas)



## rf_smote
pos_probs_rf_smote = rf_smote.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_rf_smote, recall_rf_smote, _ = precision_recall_curve(y_val, pos_probs_rf_smote)

## rfs_smote
# calculate model precision-recall curve, smartest_rf_probas are calculated earlier.
precision_rfs_smote, recall_rfs_smote, _ = precision_recall_curve(y_val, smartest_rf_probas_smote)


## rf_nodist aka abc
pos_probs_rf_abc = rf_abc.predict_proba(X_val[xvars_abc_JME])[:, 1]
# calculate model precision-recall curve
precision_rf_abc, recall_rf_abc, _ = precision_recall_curve(y_val, pos_probs_rf_abc)

## rfs_nodist
# calculate model precision-recall curve, smartest_rf_probas are calculated earlier.
precision_rfs_abc, recall_rfs_abc, thresholds_abc = precision_recall_curve(y_val, smartest_rf_probas_abc)


## rf_nodist aka abc
pos_probs_rf_cv = rf_cv.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_rf_cv, recall_rf_cv, _ = precision_recall_curve(y_val, pos_probs_rf_cv)

## rfs_nodist
# calculate model precision-recall curve, smartest_rf_probas are calculated earlier.
precision_rfs_cv, recall_rfs_cv, thresholds_cv = precision_recall_curve(y_val, smartest_rf_probas_cv)



## ada
pos_probs_ada = abc.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_ada, recall_ada, _ = precision_recall_curve(y_val, pos_probs_ada)


## knn
pos_probs_knn = knn.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_knn, recall_knn, _ = precision_recall_curve(y_val, pos_probs_knn)

## knn_up
pos_probs_knn_up = knn_up.predict_proba(X_val)[:, 1]
# calculate model precision-recall curve
precision_knn_up, recall_knn_up, _ = precision_recall_curve(y_val, pos_probs_knn_up)


# In[274]:


plt.close()

plt.figure(figsize=(10,6))
plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)

### BASELINES

# calculate the no skill line as the proportion of the positive class
no_skill = len(Y[Y==True]) / len(Y)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill',color="grey")
# plot the point from distance
plt.scatter([0.34146],[0.34146],marker="o",linewidths=1,color="black")
plt.plot(recs_val,precs_val,'-',markersize = 2,color="black",linestyle='dashed',label='Existing Method (Linear Dist)')


### NEW MODELS


clrs = sns.color_palette('husl', n_colors=15)

plot the model precision-recall curve
plt.plot(recall_lr1, precision_lr1, marker='.', label='LogReg',color=clrs[0])
plt.plot(recall_lr2, precision_lr2, marker='.', label='LogReg up',color=clrs[1])
plt.plot(recall_knn, precision_knn, marker='.', label='kNN',color=clrs[2])
plt.plot(recall_knn_up, precision_knn_up, marker='.', label='kNN up',color=clrs[3])
plt.plot(recall_ada, precision_ada, marker='.', label='AdaBoost',color=clrs[4])
plt.plot(recall_rf, precision_rf, marker='.', label='Random Forest',color=clrs[6])
plt.plot(recall_rf2, precision_rf2, marker='.', label='Random Forest up',color=clrs[7])
plt.plot(recall_rf2s, precision_rf2s, marker='.', label='Random Forest up by Credset',color=clrs[8])
plt.plot(recall_rf_smote, precision_rf_smote, marker='.', label='Random Forest SMOTE',color=clrs[9])
plt.plot(recall_rfs_smote, precision_rfs_smote, marker='.', label='Random Forest SMOTE by Credset',color=clrs[10])
plt.plot(recall_rf_abc, precision_rf_abc, marker='.', label='Random Forest ABC',color=clrs[11])
plt.plot(recall_rfs_abc, precision_rfs_abc, marker='.', label='Random Forest ABC by Credset',color=clrs[12])
plt.plot(recall_rf_cv, precision_rf_cv, marker='.', label='Random Forest CV',color=clrs[1])
plt.plot(recall_rfs_cv, precision_rfs_cv, marker='.', label='Random Forest CV by Credset',color=clrs[2])



### HOUSEKEEPING

# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))


# title
plt.title("PR Curve Test Set")
# show the plot
plt.show()
#plt.savefig("RF_and_abc.jpeg")#savefig("final_figs/full_val_PR_test.jpeg")
plt.close()


# In[275]:


print(np.interp(0.34146,[no_skill, no_skill],[0, 1]))
print(np.interp(0.34146,[0.34146],[0.34146]))
print(np.interp(0.34146,precision_lr1,recall_lr1))
print(np.interp(0.34146,precision_lr2,recall_lr2))
print(np.interp(0.34146,precision_knn,recall_knn))
print(np.interp(0.34146,precision_knn_up,recall_knn_up))
print(np.interp(0.34146,precision_ada,recall_ada))
print(np.interp(0.34146,precision_rf,recall_rf))
print(np.interp(0.34146,precision_rf2,recall_rf2))
print(np.interp(0.34146,precision_rf2s,recall_rf2s))
print(np.interp(0.34146,precision_rf_smote,recall_rf_smote))
print(np.interp(0.34146,precision_rfs_smote,recall_rfs_smote))
print(np.interp(0.34146,precision_rf_abc,recall_rf_abc))
print(np.interp(0.34146,precision_rfs_abc,recall_rfs_abc))


# In[276]:


print(np.interp(0.34146,[0, 1],[no_skill, no_skill]))
print(np.interp(0.34146,[0.34146],[0.34146]))
print(np.interp(0.34146,np.flip(recall_lr1,0),np.flip(precision_lr1,0)))
print(np.interp(0.34146,np.flip(recall_lr2,0),np.flip(precision_lr2,0)))
print(np.interp(0.34146,np.flip(recall_knn,0),np.flip(precision_knn,0)))
print(np.interp(0.34146,np.flip(recall_knn_up,0),np.flip(precision_knn_up,0)))
print(np.interp(0.34146,np.flip(recall_ada,0),np.flip(precision_ada,0)))
print(np.interp(0.34146,np.flip(recall_rf,0),np.flip(precision_rf,0)))
print(np.interp(0.34146,np.flip(recall_rf2,0),np.flip(precision_rf2,0)))
print(np.interp(0.34146,np.flip(recall_rf2s,0),np.flip(precision_rf2s,0)))
print(np.interp(0.34146,np.flip(recall_rf_smote,0),np.flip(precision_rf_smote,0)))
print(np.interp(0.34146,np.flip(recall_rfs_smote,0),np.flip(precision_rfs_smote,0)))
print(np.interp(0.34146,np.flip(recall_rf_abc,0),np.flip(precision_rf_abc,0)))
print(np.interp(0.34146,np.flip(recall_rfs_abc,0),np.flip(precision_rfs_abc,0)))


# In[278]:


thresholds_rf2_full = np.append(thresholds_rf2,1)
thresholds__rfs_abc_full = np.append(thresholds_abc,1)


# In[279]:


print(np.interp(0.411,np.flip(recall_rf2s,0),np.flip(precision_rf2s,0)))
print(np.interp(0.4,np.flip(recall_rf2s,0),np.flip(thresholds_rf2_full,0)))
print(np.interp(0.4,np.flip(recall_rfs_abc,0),np.flip(thresholds__rfs_abc_full,0)))


# In[70]:


for i in range(len(precision_rf2s)):
    print(precision_rf2s[i], recall_rf2s[i],thresholds_rf2_full[i])


## PR curves by threshold, best model

# In[280]:


plt.close()
## rf2s
# calculate model precision-recall curve, smartest_rf_probas are calculated earlier.
precision_rf2s, recall_rf2s, thresholds = precision_recall_curve(y_val, smartest_rf_probas)


plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
plt.title("Decision Threshold Tuning")
plt.plot(thresholds, precision_rf2s[:-1], label="Precision")
plt.plot(thresholds, recall_rf2s[:-1], label="Recall")
plt.ylabel("Score")
plt.xlabel("Decision Threshold")
plt.axvline(x = 0.3413, linewidth=1, linestyle = "dashed", color='r',label="Selected Threshold")
plt.legend(loc='best')
plt.show()
#plt.savefig("final_figs/threshold_adjust_rf.jpeg") #show()


# We note that at a threshold of $0.411764705882$ we achieve a precision of $0.68$ and a recall of $0.414634146341$

# In[281]:


def use_t_to_predict(y_probas, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_probas]


# In[282]:


thresh_rf2s = np.interp(0.4,np.flip(recall_rf2s,0),np.flip(thresholds_rf2_full,0))
thresh_rfs_abc = np.interp(0.4,np.flip(recall_rfs_abc,0),np.flip(thresholds__rfs_abc_full,0))


# In[283]:


rf2s_predictions = use_t_to_predict(smartest_rf_probas,thresh_rf2s) #0.411764705882)
rfs_abc_predictions = use_t_to_predict(smartest_rf_probas_abc,thresh_rfs_abc) #0.411764705882)


# In[284]:


print(precision_score(y_val,rf2s_predictions))
print(recall_score(y_val,rf2s_predictions))
print(accuracy_score(y_val,rf2s_predictions))

print("")

print(precision_score(y_val,rfs_abc_predictions))
print(recall_score(y_val,rfs_abc_predictions))
print(accuracy_score(y_val,rfs_abc_predictions))


# # IBD test

# In[76]:


# file names
# IBD_gold_file = "/seq/lincRNA/RAP/GWAS/200304_ABCGene/191224_IBD_ABCGene.CodingAndDrugTargets.tsv"
# IBD_silver_file = "/seq/lincRNA/RAP/GWAS/200304_ABCGene/191224_IBD_ABCGene.JME_Combined.tsv"
# new_gold_file = "/seq/lincRNA/RAP/GWAS/200304_ABCGene/200220_LDLCcsTop3_ABCGene.validate.tsv"
# file names updated
IBD_gold_file = "/seq/lincRNA/RAP/GWAS/200327_ABCGene/191224_IBD_ABCGene.CodingAndDrugTargets.tsv"
IBD_silver_file = "/seq/lincRNA/RAP/GWAS/200327_ABCGene/191224_IBD_ABCGene.JME_Combined.tsv"
new_gold_file = "/seq/lincRNA/RAP/GWAS/200327_ABCGene/200220_LDLCcsTop3_ABCGene.validate.tsv"


# In[186]:


# file names
IBD_gold_file = "191224_IBD_ABCGene.CodingAndDrugTargets.tsv"
IBD_silver_file = "191224_IBD_ABCGene.JME_Combined.tsv"
new_gold_file = "200220_LDLCcsTop3_ABCGene.validate.tsv"
all_ibd_file = "191224_IBD_ABCGene.all.tsv"
all_ukbiobank_file = "200204_UKBiobank_ABCGene.all.tsv"


# In[187]:


# load files
gold = pd.read_csv(IBD_gold_file,delimiter = "\t",header='infer').dropna()
silver = pd.read_csv(IBD_silver_file,delimiter = "\t",header='infer').dropna()
new_gold = pd.read_csv(new_gold_file,delimiter = "\t",header='infer').dropna()
all_ibd = pd.read_csv(all_ibd_file,delimiter = "\t",header='infer').dropna()
all_ukbiobank = pd.read_csv(all_ukbiobank_file,delimiter = "\t",header='infer').dropna()


# In[188]:


# make ids for credible sets
gold["CredId"] = transform_into_ids(gold["CredibleSet"])
silver["CredId"] = transform_into_ids(silver["CredibleSet"])
new_gold["CredId"] = transform_into_ids(new_gold["CredibleSet"])
all_ibd["CredId"] = transform_into_ids(all_ibd["CredibleSet"])
all_ukbiobank["CredId"] = transform_into_ids(all_ukbiobank["CredibleSet"])


# In[189]:


# add credlen
def credlen_add(frame_in):
    ids_full_list = list(np.unique(frame_in.CredId))
    cred_lens = []
    for i in ids_full_list:
        cred_length = len(frame_in[frame_in["CredId"]==i])
        for j in range(cred_length):
            cred_lens.append(cred_length)
    return(cred_lens, ids_full_list)
gold["CredSetLength"], gold_id_list = credlen_add(gold)
silver["CredSetLength"], silver_id_list = credlen_add(silver)
new_gold["CredSetLength"], new_gold_id_list = credlen_add(new_gold)
all_ibd["CredSetLength"], all_ibd_id_list = credlen_add(all_ibd)
all_ukbiobank["CredSetLength"], all_ukbiobank_id_list = credlen_add(all_ukbiobank)


# In[123]:


all_ukbiobank_id_list


# In[190]:


# rename columns for shorthand
gold["MaxABC_Enriched"] = gold["MaxABC_FMOverlapEnriched"]
gold["ConnectionStrengthRank_Enriched"] = gold["ConnectionStrengthRank_FMOverlapEnriched"]
gold["GeneCS_Enriched"] = gold["GeneCS_FMOverlapEnriched"]
gold["GeneBodyDistToBestSNP"] = gold["GeneBodyDistanceToBestSNP"]
silver["MaxABC_Enriched"] = silver["MaxABC_FMOverlapEnriched"]
silver["ConnectionStrengthRank_Enriched"] = silver["ConnectionStrengthRank_FMOverlapEnriched"]
silver["GeneCS_Enriched"] = silver["GeneCS_FMOverlapEnriched"]
silver["GeneBodyDistToBestSNP"] = silver["GeneBodyDistanceToBestSNP"]
new_gold["MaxABC_Enriched"] = new_gold["MaxABC_FMOverlapEnriched"]
new_gold["ConnectionStrengthRank_Enriched"] = new_gold["ConnectionStrengthRank_FMOverlapEnriched"]
new_gold["GeneCS_Enriched"] = new_gold["GeneCS_FMOverlapEnriched"]
new_gold["GeneBodyDistToBestSNP"] = new_gold["GeneBodyDistanceToBestSNP"]
all_ibd["MaxABC_Enriched"] = all_ibd["MaxABC_FMOverlapEnriched"]
all_ibd["ConnectionStrengthRank_Enriched"] = all_ibd["ConnectionStrengthRank_FMOverlapEnriched"]
all_ibd["GeneCS_Enriched"] = all_ibd["GeneCS_FMOverlapEnriched"]
all_ibd["GeneBodyDistToBestSNP"] = all_ibd["GeneBodyDistanceToBestSNP"]
all_ukbiobank["MaxABC_Enriched"] = all_ukbiobank["MaxABC_FMOverlapEnriched"]
all_ukbiobank["ConnectionStrengthRank_Enriched"] = all_ukbiobank["ConnectionStrengthRank_FMOverlapEnriched"]
all_ukbiobank["GeneCS_Enriched"] = all_ukbiobank["GeneCS_FMOverlapEnriched"]
all_ukbiobank["GeneBodyDistToBestSNP"] = all_ukbiobank["GeneBodyDistanceToBestSNP"]


# In[191]:


# separate X and Y vars
X_gold = gold[xvars]
X_silver = silver[xvars]
X_new_gold = new_gold[xvars]
X_all_ibd = all_ibd[xvars]
X_all_ukbiobank = all_ukbiobank[xvars]
y_gold = gold["GeneList.CodingAndDrugTargets"]
y_silver = silver["GeneList.JME_Combined"]
y_new_gold = new_gold["GeneList.LDL.Khera2020.RVAS"]


# In[111]:


plt.close()
fig, axs = plt.subplots(5, 5)
fig.set_size_inches(12, 12)
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.03, top=0.96, wspace=0.5, hspace=0.5)
first_col = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
second_col = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
i = 0
for col in X_gold.columns:
    sns.kdeplot(X_gold[gold["GeneList.CodingAndDrugTargets"]==False][col].values,label="False",ax = axs[first_col[i],second_col[i]])
    sns.kdeplot(X_gold[gold["GeneList.CodingAndDrugTargets"]==True][col].values,label="True",ax = axs[first_col[i],second_col[i]])
    axs[first_col[i],second_col[i]].set_title(col)
    i = i+1

axs[0,0].set_xticklabels(["","0MM","0.5MM","1MM","1.5MM"])    


# for ax in axs.flat:
#     ax.set(xlabel='Value', ylabel='Density')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

# axs[4,2].set_xlabel("Value")
# axs[2,0].set_ylabel("Density")

plt.show() #savefig("final_figs/kde_tf/all_kde_test.jpeg")


# In[112]:


val.columns


# In[192]:


# add rf_preds
gold["rf_preds"] = rf2.predict_proba(X_gold)[:, 1]
silver["rf_preds"] = rf2.predict_proba(X_silver)[:, 1]
new_gold["rf_preds"] = rf2.predict_proba(X_new_gold)[:, 1]
all_ibd["rf_preds"] = rf2.predict_proba(X_all_ibd)[:, 1]
all_ukbiobank["rf_preds"] = rf2.predict_proba(X_all_ukbiobank)[:, 1]
gold["rf_preds_abc"] = rf_abc.predict_proba(X_gold[xvars_abc_JME])[:, 1]
silver["rf_preds_abc"] = rf_abc.predict_proba(X_silver[xvars_abc_JME])[:, 1]
new_gold["rf_preds_abc"] = rf_abc.predict_proba(X_new_gold[xvars_abc_JME])[:, 1]
gold["rf_preds_cv"] = rf_cv.predict_proba(X_gold)[:, 1]
silver["rf_preds_cv"] = rf_cv.predict_proba(X_silver)[:, 1]
new_gold["rf_preds_cv"] = rf_cv.predict_proba(X_new_gold)[:, 1]


# In[118]:


all_ukbiobank


# In[121]:


print(len(all_ukbiobank["rf_preds"]),len(probs_by_credset(all_ukbiobank,"rf_preds")))


# In[240]:


# get list IDs
gold_ids = list(np.unique(gold.CredId))
silver_ids = list(np.unique(silver.CredId))
new_gold_ids = list(np.unique(new_gold.CredId))
all_ibd_ids = list(np.unique(all_ibd.CredId))
all_ukbiobank_ids = list(np.unique(all_ukbiobank.CredId))


# In[241]:


# add predictions from "smarter rf" by credible set
gold["smart_rf"] = probs_by_credset(gold,"rf_preds",gold_ids)
silver["smart_rf"] = probs_by_credset(silver,"rf_preds",silver_ids)
new_gold["smart_rf"] = probs_by_credset(new_gold,"rf_preds",new_gold_ids)
all_ibd["smart_rf"] = probs_by_credset(all_ibd,"rf_preds",all_ibd_ids)
all_ukbiobank["smart_rf"] = probs_by_credset(all_ukbiobank,"rf_preds",all_ukbiobank_ids)
gold["smart_rf_abc"] = probs_by_credset(gold,"rf_preds_abc",gold_ids)
silver["smart_rf_abc"] = probs_by_credset(silver,"rf_preds_abc",silver_ids)
new_gold["smart_rf_abc"] = probs_by_credset(new_gold,"rf_preds_abc",new_gold_ids)
gold["smart_rf_cv"] = probs_by_credset(gold,"rf_preds_cv",gold_ids)
silver["smart_rf_cv"] = probs_by_credset(silver,"rf_preds_cv",silver_ids)
new_gold["smart_rf_cv"] = probs_by_credset(new_gold,"rf_preds_cv",new_gold_ids)


# In[85]:


def smartest(frame, id_list):
    smart_rf_preds_i = []
    for i in range(len(id_list)):
        current_credset = frame[frame["CredId"]==i]
        max_val_credset = np.max(current_credset["smart_rf"])
        for index, row in current_credset.iterrows():
            if row["smart_rf"] == max_val_credset:
                smart_rf_preds_i.append(1)
            else:
                smart_rf_preds_i.append(0)
val["smart_rf_i"] = smart_rf_preds_i

print(accuracy_score(y_val,smart_rf_preds_2))
print(precision_score(y_val,smart_rf_preds_2))
print(recall_score(y_val,smart_rf_preds_2))


# In[242]:


# insert default preds
gold["dist_preds_1"] = just_one("DistanceRank",X_gold)
gold["abc_preds_1"] = just_one("ConnectionStrengthRank",X_gold)
silver["dist_preds_1"] = just_one("DistanceRank",X_silver)
silver["abc_preds_1"] = just_one("ConnectionStrengthRank",X_silver)
new_gold["dist_preds_1"] = just_one("DistanceRank",X_new_gold)
new_gold["abc_preds_1"] = just_one("ConnectionStrengthRank",X_new_gold)
all_ibd["dist_preds_1"] = just_one("DistanceRank",all_ibd)
all_ukbiobank["dist_preds_1"] = just_one("DistanceRank",all_ukbiobank)


# In[87]:


rf2_gold_preds = rf2.predict(X_gold)
print("RF Validation/Test Accuracy Gold:",accuracy_score(rf2_gold_preds, y_gold))
rf2_silver_preds = rf2.predict(X_silver)
print("RF Validation/Test Accuracy Silver:",accuracy_score(rf2_silver_preds, y_silver))
rf2_new_gold_preds = rf2.predict(X_new_gold)
print("RF Validation/Test Accuracy New Gold:",accuracy_score(rf2_new_gold_preds, y_new_gold))
abc_gold_preds = abc.predict(X_gold)
print("ABC Validation/Test Accuracy Gold:",accuracy_score(abc_gold_preds, y_gold))
abc_silver_preds = abc.predict(X_silver)
print("ABC Validation/Test Accuracy Silver:",accuracy_score(abc_silver_preds, y_silver))
abc_new_gold_preds = abc.predict(X_new_gold)
print("ABC Validation/Test Accuracy New Gold:",accuracy_score(abc_new_gold_preds, y_new_gold))


# In[196]:


dist_preds_1 = just_one("DistanceRank",X_gold)
print("Test Accuracy 1:",accuracy_score(dist_preds_1, y_gold))
print("Prec-Rec, 1:", precision_score(y_gold,dist_preds_1), recall_score(y_gold,dist_preds_1))


# In[197]:


dist_preds_1 = just_one("DistanceRank",X_silver)
print("Test Accuracy 1:",accuracy_score(dist_preds_1, y_silver))
print("Prec-Rec, 1:", precision_score(y_silver,dist_preds_1), recall_score(y_silver,dist_preds_1))


# In[247]:


def recs_precs_maker(colname):
    recs_silv = []
    precs_silv = []

    recs_gold = []
    precs_gold = []

    recs_new_gold = []
    precs_new_gold = []

    for i in range(20):
        if (i>=1):
            preds_i = big_ranker(colname,X_silver,i)
            recs_silv.append(recall_score(y_silver,preds_i))
            precs_silv.append(precision_score(y_silver,preds_i))

            preds_i = big_ranker(colname,X_gold,i)
            recs_gold.append(recall_score(y_gold,preds_i))
            precs_gold.append(precision_score(y_gold,preds_i))

            preds_i = big_ranker(colname,X_new_gold,i)
            recs_new_gold.append(recall_score(y_new_gold,preds_i))
            precs_new_gold.append(precision_score(y_new_gold,preds_i))
        
    return(recs_silv,precs_silv,recs_gold,precs_gold,recs_new_gold,precs_new_gold)


# In[248]:


## SILVER AND GOLD and newgold DISTANCE
recs_gold = []
precs_gold = []

recs_silv = []
precs_silv = []

recs_new_gold = []
precs_new_gold = []


recs_silv,precs_silv,recs_gold,precs_gold,recs_new_gold,precs_new_gold = recs_precs_maker("DistanceRank")


# In[92]:


dist_preds_1 = just_one("DistanceRank",X_silver)
dist_preds_2 = one_or_two("DistanceRank",X_silver)
print("Test Accuracy 1:",accuracy_score(dist_preds_1, y_silver))
print("Test Accuracy 2:",accuracy_score(dist_preds_2, y_silver))
print("Prec-Rec, 1:", precision_score(y_silver,dist_preds_1), recall_score(y_silver,dist_preds_1))
print("Prec-Rec, 1or2:", precision_score(y_silver,dist_preds_2),  recall_score(y_silver,dist_preds_2))


# In[93]:


dist_preds_1 = just_one("DistanceRank",X_gold)
dist_preds_2 = one_or_two("DistanceRank",X_gold)
print("Test Accuracy 1:",accuracy_score(dist_preds_1, y_gold))
print("Test Accuracy 2:",accuracy_score(dist_preds_2, y_gold))
print("Prec-Rec, 1:", precision_score(y_gold,dist_preds_1), recall_score(y_gold,dist_preds_1))
print("Prec-Rec, 1or2:", precision_score(y_gold,dist_preds_2),  recall_score(y_gold,dist_preds_2))


# In[94]:


dist_preds_1 = just_one("DistanceRank",X_new_gold)
dist_preds_2 = one_or_two("DistanceRank",X_new_gold)
print("Test Accuracy 1:",accuracy_score(dist_preds_1, y_new_gold))
print("Test Accuracy 2:",accuracy_score(dist_preds_2, y_new_gold))
print("Prec-Rec, 1:", precision_score(y_new_gold,dist_preds_1), recall_score(y_new_gold,dist_preds_1))
print("Prec-Rec, 1or2:", precision_score(y_new_gold,dist_preds_2),  recall_score(y_new_gold,dist_preds_2))


# In[290]:


### FOR SILVER


pred_proba_val_silver = [item[1] for item in rf2.predict_proba(X_silver)]

# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_silver,pred_proba_val_silver)

rf2s_predictions_silver = use_t_to_predict(silver["smart_rf"],thresh_rf2s)


plt.close()
#posprobs
pos_probs = rf2.predict_proba(X_silver)[:, 1]
# calculate the no skill line as the proportion of the positive class
no_skill = len(y_silver[y_silver==True]) / len(y_silver)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
# precision, recall, _ = precision_recall_curve(y_silver, pos_probs)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_silver, silver["smart_rf"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest by CredSet')

precision, recall, _ = precision_recall_curve(y_silver, silver["smart_rf_abc"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest ABC by CredSet')

precision, recall, _ = precision_recall_curve(y_silver, silver["smart_rf_cv"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest CV by CredSet')

plt.scatter([0.681818181818],[0.681818181818],marker="o",linewidths=1,color="black",label='Existing Method (Linear Dist)')
plt.scatter(recall_score(y_silver,rf2s_predictions_silver),precision_score(y_silver,rf2s_predictions_silver),marker="o",linewidths=1,color="blue")

plt.plot(recs_silv,precs_silv,marker='o',markersize = 2,linestyle="dashed")
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1))
# title
plt.title("PR Curve IBD")
# show the plot
plt.show()


# In[98]:


print(precision_score(y_silver,just_one("DistanceRank",X_silver)), 
     recall_score(y_silver,just_one("DistanceRank",X_silver)),
          accuracy_score(y_silver,just_one("DistanceRank",X_silver)))
print(precision_score(y_silver,rf2s_predictions_silver),
      recall_score(y_silver,rf2s_predictions_silver),
          accuracy_score(y_silver,rf2s_predictions_silver))


# In[288]:


precision_score(y_silver,just_one("DistanceRank",X_silver))


# In[202]:


###GOLD

pred_proba_val_gold = [item[1] for item in rf2.predict_proba(X_gold)]

# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_gold,pred_proba_val_gold)

rf2s_predictions_gold = use_t_to_predict(gold["smart_rf"],thresh_rf2s)


plt.close()
#posprobs
pos_probs = rf2.predict_proba(X_gold)[:, 1]
# calculate the no skill line as the proportion of the positive class
no_skill = len(y_gold[y_gold==True]) / len(y_gold)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_gold, pos_probs)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_gold, gold["smart_rf"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest by CredSet')

precision, recall, _ = precision_recall_curve(y_gold, gold["smart_rf_abc"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest ABC by CredSet')


precision, recall, _ = precision_recall_curve(y_gold, gold["smart_rf_cv"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest CV by CredSet')


plt.scatter([0.857142857143],[0.857142857143],marker="o",linewidths=1,color="black",label='Existing Method (Linear Dist)')
plt.scatter(recall_score(y_gold,rf2s_predictions_gold),precision_score(y_gold,rf2s_predictions_gold),marker="o",linewidths=1,color="blue")


plt.plot(recs_gold,precs_gold,marker='o',markersize = 2,
         label='Existing Method (Linear Dist)')
# plt.plot(recs_gold_abc,precs_gold_abc,marker='o',markersize = 2,
#          label='ABC Rank')# axis labels
# plt.plot(recs_gold_fm,precs_gold_fm,marker='o',markersize = 2,
#          label='ABC Enriched Rank')# axis labelsplt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend(loc="lower right")
# title
plt.title("PR Curve IBD Gold")
# show the plot
plt.show()


# In[100]:


print(precision_score(y_gold,just_one("DistanceRank",X_gold)), 
     recall_score(y_gold,just_one("DistanceRank",X_gold)),
          accuracy_score(y_gold,just_one("DistanceRank",X_gold)))
print(precision_score(y_gold,rf2s_predictions_gold),
      recall_score(y_gold,rf2s_predictions_gold),
          accuracy_score(y_gold,rf2s_predictions_gold))


# In[203]:


### NEW GOLD

pred_proba_val_gold = [item[1] for item in rf2.predict_proba(X_new_gold)]

# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_new_gold,pred_proba_val_gold)

rf2s_predictions_new_gold = use_t_to_predict(new_gold["smart_rf"],0.411764705882)


plt.close()
#posprobs
pos_probs = rf2.predict_proba(X_new_gold)[:, 1]
# calculate the no skill line as the proportion of the positive class
no_skill = len(y_new_gold[y_new_gold==True]) / len(y_new_gold)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_new_gold, pos_probs)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_new_gold, new_gold["smart_rf"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest by CredSet')

precision, recall, _ = precision_recall_curve(y_new_gold, new_gold["smart_rf_abc"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest ABC by CredSet')

precision, recall, _ = precision_recall_curve(y_new_gold, new_gold["smart_rf_cv"])
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest CV by CredSet')


plt.scatter([0.875],[0.875],marker="o",linewidths=1,color="black",label='Existing Method (Linear Dist)')
plt.scatter(recall_score(y_new_gold,rf2s_predictions_new_gold),precision_score(y_new_gold,rf2s_predictions_new_gold),marker="o",linewidths=1,color="blue")



plt.plot(recs_new_gold,precs_new_gold,marker='o',markersize = 2,
         label='Existing Method (Linear Dist)')
# plt.plot(recs_gold_abc,precs_gold_abc,marker='o',markersize = 2,
#          label='ABC Rank')# axis labels
# plt.plot(recs_gold_fm,precs_gold_fm,marker='o',markersize = 2,
#          label='ABC Enriched Rank')# axis labelsplt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend(loc="lower left")
# title
plt.title("PR Curve LDLC Gold")
# show the plot
plt.show()


# In[102]:


print(precision_score(y_new_gold,just_one("DistanceRank",X_new_gold)), 
     recall_score(y_new_gold,just_one("DistanceRank",X_new_gold)),
          accuracy_score(y_new_gold,just_one("DistanceRank",X_new_gold)))
print(precision_score(y_new_gold,rf2s_predictions_new_gold),
      recall_score(y_new_gold,rf2s_predictions_new_gold),
          accuracy_score(y_new_gold,rf2s_predictions_new_gold))


# ## on all IBD

# In[204]:


all_ibd_predictions = use_t_to_predict(all_ibd["smart_rf"],0.411764705882)


# In[206]:


sum(all_ibd_predictions)


# In[207]:


print(sum(rf2s_predictions_silver))


# In[208]:


all_ibd["full_preds"] = all_ibd_predictions


# In[217]:


predicted_causal_genes = all_ibd[all_ibd["full_preds"] == 1]


# In[227]:


print(sum(predicted_causal_genes["GeneList.JME_Combined"]))
print(sum(all_ibd["GeneList.JME_Combined"]))


# In[231]:


for index, row in predicted_causal_genes.iterrows():
    print(#row["full_preds"], row["GeneList.JME_Combined"],
         row["CredibleSet"][10:],row["TargetGene"], row["Disease"])


# ## all abc

# In[249]:


all_ukbiobank_predictions = use_t_to_predict(all_ukbiobank["smart_rf"],0.411764705882)


# In[250]:


sum(all_ukbiobank_predictions)


# In[254]:


print(sum(rf2s_predictions))


# In[256]:


all_ukbiobank["full_preds"] = all_ukbiobank_predictions


# In[257]:


predicted_causal_genes_all = all_ukbiobank[all_ukbiobank["full_preds"] == 1]


# In[258]:


print(sum(predicted_causal_genes_all[yvar]))
print(sum(all_ukbiobank[yvar]))


# In[262]:


for index, row in predicted_causal_genes_all.iterrows():
    print(#row["full_preds"], row["GeneList.JME_Combined"],
         row["CredibleSet"]," & ", row["TargetGene"]," & ", row["Disease"]," \\\ ")


# ## confidence intervals

# In[178]:


def bootstrap_interval(data, percentiles=(2.5,97.5), n_boots=10000):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create empty array to fill the results
    bootstrap_diffs = []
    for ii in range(n_boots):
        # Generate random indices for data *with* replacement, then take the sample mean
        random_sample = resample(data)
        temp_prec_score_RF = precision_score(random_sample[yvar],random_sample["rf2s_predictions"])
        temp_prec_score_DIST = precision_score(random_sample[yvar],random_sample["just_one"])
        #print(temp_prec_score_RF-temp_prec_score_DIST)
        bootstrap_diffs.append(temp_prec_score_RF-temp_prec_score_DIST)
    # Compute the percentiles of choice for the bootstrapped diffs
    print(np.mean(bootstrap_diffs))
    percentiles = np.percentile(bootstrap_diffs, percentiles, axis=0)
    return percentiles, bootstrap_diffs


# In[81]:


pcts, bsds = bootstrap_interval(val)


# In[82]:


pcts


# In[179]:


def bootstrap_interval5(data, ytruth=yvar, percentiles=(2.5,97.5), n_boots=10000):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    diff_cur = precision_score(data[ytruth],data["rf2s_predictions"]) - precision_score(data[ytruth],data["just_one"])
    print(diff_cur)    
    # Create empty array to fill the results
    bootstrap_diffs = []
    for ii in range(n_boots):
        # Generate random indices for data *with* replacement, then take the sample mean
        random_sample = resample(data)
        temp_prec_score_RF = precision_score(random_sample[ytruth],random_sample["rf2s_predictions"])
        temp_prec_score_DIST = precision_score(random_sample[ytruth],random_sample["just_one"])
        #print(temp_prec_score_RF-temp_prec_score_DIST)
        bootstrap_diffs.append((temp_prec_score_RF-temp_prec_score_DIST)-diff_cur)
    # Compute the percentiles of choice for the bootstrapped diffs
    print(np.mean(bootstrap_diffs))
    percentiles = np.percentile(bootstrap_diffs, percentiles, axis=0)
    return diff_cur, percentiles, bootstrap_diffs


# In[107]:


val["rf2s_predictions"] = rf2s_predictions
val["just_one"] = just_one("DistanceRank",val)
dc2, pcts2, bsds2 = bootstrap_interval5(val)
print(dc2-pcts2[1],dc2-pcts2[0])


# In[102]:


## ROC
fpr, tpr, thresholds = metrics.roc_curve(y_val, smartest_rf_probas)
plt.close()
plt.figure(figsize=(5,5))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label = "Best Random Forest")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label = "No Skill")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="right",bbox_to_anchor=(1.55, .8))
plt.show()


# In[43]:


print(grid_search_transform)


# In[32]:


GridSearchCV(rf2, param_grid, scoring=None, n_jobs=None, iid='deprecated', 
                                           refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, 
                                           return_train_score=False)


# In[ ]:


# Separate majority and minority classes
df_majority = train[train.CodingPP50==False]
df_minority = train[train.CodingPP50==True]
ratio = int(df_majority.shape[0]/df_minority.shape[0])
 
# # Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class # fix
                                 random_state=123) # reproducible results
 
# # Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# # Display new class counts
# print(df_upsampled.CodingPP50.value_counts())
# True     5356
# False    5356
# Name: CodingPP50, dtype: int64

X_train_up = df_upsampled[xvars]
X_train_up_nodist = df_upsampled[xvars_nodist]
y_train_up = df_upsampled[yvar]


# In[103]:


val


# ### Confusion matrices

# In[53]:


conf_mat = confusion_matrix(y_true=y_val, y_pred=np.repeat(0, len(y_val)))
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# In[54]:


conf_mat = confusion_matrix(y_true=y_val, y_pred=dist_val_preds)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# In[55]:


conf_mat = confusion_matrix(y_true=y_val, y_pred=rf2s_predictions)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# In[77]:


# massive combined control set?
massive = pd.concat([silver,gold,new_gold])
# make new ids for credible sets
massive["CredId"] = transform_into_ids(massive["CredibleSet"])
## looking at dataset, weird stuff happening here. massive

mass_X = massive[xvars]
mass_y = np.array(flatten_list([y_silver.values,y_gold.values,y_new_gold.values]))


# In[78]:


massive["truth"] = mass_y


# In[79]:


smarties = pd.concat([silver["smart_rf"],gold["smart_rf"],new_gold["smart_rf"]])
massive["smarties"] = smarties
massive.to_csv (r'3.30/massive_3.31.csv', index = None, header=True)


# In[80]:


pred_proba_massive = [item[1] for item in rf2.predict_proba(mass_X)]

# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(mass_y,pred_proba_massive)

rf2s_predictions_massive = use_t_to_predict(smarties,0.411764705882)
massive["rf2s_predictions_massive"] = rf2s_predictions_massive

plt.close()
#posprobs
pos_probs = rf2.predict_proba(mass_X)[:, 1]
# calculate the no skill line as the proportion of the positive class
no_skill = len(mass_y[mass_y==True]) / len(mass_y)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(mass_y, pos_probs)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(mass_y, smarties)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Random Forest by CredSet')



plt.scatter(recall_score(mass_y,just_one("DistanceRank",mass_X)), precision_score(mass_y,just_one("DistanceRank",mass_X)),marker="o",linewidths=1,color="black",label='Existing Method (Linear Dist)')
plt.scatter(recall_score(mass_y,rf2s_predictions_massive),precision_score(mass_y,rf2s_predictions_massive),marker="o",linewidths=1,color="blue",label="New Method")



plt.plot(recs_gold_abc,precs_gold_abc,marker='o',markersize = 2,
         label='ABC Rank')# axis labels
plt.plot(recs_gold_fm,precs_gold_fm,marker='o',markersize = 2,
         label='ABC Enriched Rank')# axis labelsplt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend(loc="lower left")
# title
plt.title("PR Values Drug Dataset")
# show the plot
plt.show()


# In[81]:


no_skill = len(mass_y[mass_y==True]) / len(mass_y)
print(no_skill)
print(recall_score(mass_y,no_skill), precision_score(mass_y,no_skill))
print(recall_score(mass_y,just_one("DistanceRank",mass_X)), precision_score(mass_y,just_one("DistanceRank",mass_X)))
print(recall_score(mass_y,rf2s_predictions_massive),precision_score(mass_y,rf2s_predictions_massive))


# ## bootstrap confidence interval

# In[82]:


def bootstrap_interval2(data, percentiles=(0.05, 100), n_boots=1000):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create empty array to fill the results
    bootstrap_diffs = []
    for ii in range(n_boots):
        # Generate random indices for data *with* replacement, then take the sample mean
        random_sample = resample(data)
        temp_prec_score_RF = precision_score(random_sample["truth"],random_sample["rf2s_predictions_massive"])
        temp_prec_score_DIST = precision_score(random_sample["truth"],just_one("DistanceRank",random_sample))
        #print(temp_prec_score_RF-temp_prec_score_DIST)
        bootstrap_diffs.append(temp_prec_score_RF-temp_prec_score_DIST)
    # Compute the percentiles of choice for the bootstrapped diffs
    percentiles = np.percentile(bootstrap_diffs, percentiles, axis=0)
    return percentiles, bootstrap_diffs


# In[ ]:


def bootstrap_interval3(data, percentiles=(0.05, 100), n_boots=1000):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    cur_diff = precision_score(data["truth"],data["rf2s_predictions_massive"])-precision_score(data["truth"],just_one("DistanceRank",data))
    print(cur_diff)    
    
    # Create empty array to fill the results
    bootstrap_diffs = []
    for ii in range(n_boots):
        # Generate random indices for data *with* replacement, then take the sample mean
        random_sample = resample(data)
        temp_prec_score_RF = precision_score(random_sample["truth"],random_sample["rf2s_predictions_massive"])
        temp_prec_score_DIST = precision_score(random_sample["truth"],just_one("DistanceRank",random_sample))
        #print(temp_prec_score_RF-temp_prec_score_DIST)
        bootstrap_diffs.append(temp_prec_score_RF-temp_prec_score_DIST)
    # Compute the percentiles of choice for the bootstrapped diffs
    percentiles = np.percentile(bootstrap_diffs, percentiles, axis=0)
    return percentiles, bootstrap_diffs


# In[151]:


pects, bms = bootstrap_interval2(massive,n_boots=1000)
print(pects)


# In[85]:


pects, bms = bootstrap_interval2(massive,n_boots=1000)
print(pects)


# In[88]:


plt.hist(bms,bins=30)
plt.show()


# In[89]:


mass_fair = massive[massive["ConnectionStrengthRank"] < 100]


# In[91]:


pects, bms = bootstrap_interval2(mass_fair,n_boots=100)
plt.hist(bms,bins=30)
plt.show()


# In[124]:


pects2, bms2 = bootstrap_interval2(mass_fair,n_boots=100)
print(pects2)


# In[115]:


massive["rf2s_predictions"] = rf2s_predictions_massive
massive["just_one"] = just_one("DistanceRank",massive)
dc_mass, pcts_mass, bsds_mass = bootstrap_interval5(massive,"truth",(10,90))
print(dc_mass-pcts_mass[1],dc_mass-pcts_mass[0])

