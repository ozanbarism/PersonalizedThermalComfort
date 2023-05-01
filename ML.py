#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
from warnings import simplefilter

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import csv
from sklearn.neural_network import MLPClassifier
import time
from datetime import datetime

simplefilter(action='ignore', category=FutureWarning)
#%% 

k=0
prev_data_len={}
best_model={}
model_acc={}
importances={}
    
while True:
# In[2]:

    #to count the number of loops
    
    # Load data and print the classes
    df = pd.read_csv('enviro.csv', usecols=[0,2,3,4,6,7,8])
    
    # Rename the columns
    df.columns = ['name', 'binary', 'sensation', 'satisfaction','clo1','clo2', 'date']
    # convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.floor('min')
    df['binary'] = df['binary'].map({'Yes': 1, 'No': 0})
    
    dfs_by_name = dict(tuple(df.groupby('name')))
    
    # Get unique values in 'name' column
    unique_names = df['name'].unique()
    
    # Create dictionary of dataframes
    dfs_by_name = {}
    for name in unique_names:
        dfs_by_name[name] = df[df['name'] == name]
        
    
    
        
    #%% In[3]:
        
    temp_data={}    
    for name in unique_names:
    
        string= name.capitalize() + '_temp_hum.csv'
        # Load data and print the classes
        data = pd.read_csv(string, usecols=[0,1,2])
    # Clean up the 'temp' column by removing non-numeric characters
        if any(isinstance(value, str) for value in data['hum']):
            
            data['hum'] = data['hum'].str.replace('[^0-9\.]', '')
        data['temp'] = data['temp'].astype(float)
        data['hum'] = data['hum'].astype(float)
        data = data.rename(columns={'time': 'date'})
        data['date'] = pd.to_datetime(data['date'])
        #there was a 4 hour difference so we subtract that
        data['date'] = data['date'] - pd.Timedelta(hours=4)
        
        #error removal (values equal or smaller than zero)
        data = data[(data['temp'] > 0) & (data['hum'] > 0)]
        
        temp_data[name]=data
    
    #read the ambient temp data
    
    # Load data and print the classes
    data = pd.read_csv('Main_temp_hum.csv', usecols=[0,1,2])
    # Clean up the 'temp' column by removing non-numeric characters
    if any(isinstance(value, str) for value in data['hum']):
        
        data['hum'] = data['hum'].str.replace('[^0-9\.]', '')
        
    data['temp'] = data['temp'].astype(float)
    data['hum'] = data['hum'].astype(float)
    data = data.rename(columns={'time': 'date'})
    data['date'] = pd.to_datetime(data['date'])
    #there was a 4 hour difference from the pittsburgh time so we subtract that
    data['date'] = data['date'] - pd.Timedelta(hours=4)
    
    #error removal (values equal or smaller than zero)
    data = data[(data['temp'] > 0) & (data['hum'] > 0)]
    data = data.rename(columns={'temp': 'm_temp'})
    data = data.rename(columns={'hum': 'm_hum'})
    main_df=data
    
    #%%
    #Data preprocessing
    clo_insulation_dict = {
        "T-shirt": 0.09,
        "Dress": 0.125,
        "Long-sleeve": 0.12,
        "Jacket/Hoodie": 0.35,
        "Parka": 0.70,
        "Shorts": 0.06,
        "Jeans/Pants": 0.25,
        "Skirt": 0.18,
    }
    
    
    # Define unique clothing types
    unique_clothing_types = ['T-shirt', 'Dress', 'Long-sleeve', 'Jacket/Hoodie', 'Parka', 'Shorts', 'Jeans/Pants', 'Skirt', 'Dress']
    
    # Loop through dataframes in per_df_by_name
    for df_name in unique_names:
        # Get the dataframe by name
        df = dfs_by_name[df_name]
        # Loop through unique clothing types and update clo1 and clo2 columns
        for clo_type in unique_clothing_types:
            # Check if the clothing type is in the insulation dictionary
            if clo_type in clo_insulation_dict:
                insulation_value = clo_insulation_dict[clo_type]
                # Update the clo1 and clo2 columns
                df.loc[df['clo1'] == clo_type, 'clo1'] = insulation_value
                df.loc[df['clo2'] == clo_type, 'clo2'] = insulation_value
    
            # Add the clo column by summing clo1 and clo2
        df['clo'] = df[['clo1', 'clo2']].apply(lambda x: sum(x), axis=1)
        # Drop the clo1 and clo2 columns
        
        
    #%%
    merged_df = {}
    data_length={}
    for name in unique_names:
    # Merge the dataframes based on nearest match of 'date' column with a tolerance of 1 minute
        merged_df[name] = pd.merge_asof(dfs_by_name[name], temp_data[name], on='date', tolerance=pd.Timedelta('35minute'))
        merged_df[name] = pd.merge_asof(merged_df[name], main_df, on='date', tolerance=pd.Timedelta('35minute'))
    # Reset the index to make 'date' column a regular column
        merged_df[name].reset_index(inplace=True)
        merged_df[name]['temp'] = merged_df[name]['temp'].interpolate(method='linear')
        merged_df[name]['hum']= merged_df[name]['hum'].interpolate(method='linear')
        #print(merged_df[name])

        data_length[name]=len(merged_df[name].drop_duplicates())

    
    # In[3]:
    
    outputs=['satisfaction']
    inputs=['clo','temp', 'hum','m_temp']
    

        
    
    # In[4]:
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    import imblearn
    from imblearn.over_sampling import SMOTE
    
    # Define the SMOTE object
    smote = SMOTE(random_state=42)    
        
    


    
    if k==0:
        #print('here for the first time')
        for name in unique_names:
            prev_data_len[name]=len(merged_df[name].drop_duplicates())
    
    for output in outputs:
        #print('when the prediction is done for ', output)
        #print('------------------------------------------')
        
        for name in unique_names:

            #to check if there is new data so that we retrain

            #print(name)
            #print(data_length[name])
            #print(prev_data_len[name])
            #print(k)
            if data_length[name] > prev_data_len[name] or k==0:
                print('new data observed: training')
                prev_data_len[name] = len(merged_df[name].drop_duplicates())
                print('For the user:', name)
                loo = LeaveOneOut()
                
                X = merged_df[name][inputs].values
                Y = merged_df[name][output].values
                
                X = np.array(X)
                Y = np.array(Y)
                
                #X, Y = smote.fit_resample(X, Y)
        
                
                
                classifiers = [KNeighborsClassifier(),               SVC(kernel='poly'),             
                                            DecisionTreeClassifier(),    
                            GaussianNB()     ,        LogisticRegression(max_iter=10000000, solver='sag'),
                            MLPClassifier(max_iter=10000000)] #RandomForestClassifier(), GradientBoostingClassifier(),
                i=0
                for clf_model in classifiers:
                    accuracy = []
                    for train_index, test_index in loo.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = Y[train_index], Y[test_index]
                        clf_model.fit(X_train, y_train)
                        accuracy.append(clf_model.score(X_test, y_test))
                
                    model_acc[clf_model]=np.mean(accuracy)
                    print("Leave one trial out cross-val accuracy for",clf_model,"is", np.mean(accuracy))
                    
                    if i==2:
                        importances[name]=clf_model.feature_importances_
                    i=i+1
            best_model[name]= max(model_acc, key=model_acc.get)
            
    
    # In[3]:
        

    
    for name in unique_names:
        
        last_date = temp_data[name]['date'].iloc[-1] # Get the last date from the column
        current_time = datetime.now() # Get the current time
        time_diff = (current_time - last_date).total_seconds() / 60 # 
        #print(time_diff)
        
        if time_diff>5:
            unique_names = np.delete(unique_names, np.where(unique_names == name))
            
    
    # In[ ]
    set_points = np.arange(15, 35, 0.5)
    #here we loop through different setpoints as Main temp to see which one would make the most number of people comfortable
    comfort_range={}
    for name in unique_names:
    
        comfy_temp_list=[]
        clf_model=best_model[name]
        
        #X_train = merged_df[name][inputs].values
        #Y_train = merged_df[name][output].values
        
        #X_train = np.array(X_train)
        #Y_train = np.array(Y_train)
        
        #X_train = scaler.fit_transform(X_train)
        #print(name)
        #print(clf_model)
        temp,hum=temp_data[name].iloc[-1, [1,2]].values
        main_hum=main_df.iloc[-1, 2]
        clo1,clo2=merged_df[name].iloc[-1,[5,6]].values
        for Tset in set_points:
    
            X=np.array([clo1+clo2,temp,hum,Tset]).reshape(1, -1)
            
    
    
            pred=clf_model.predict(X)
            #print(pred)
    
            #change this if you are using a different scale: zero for satisfaction
            if pred==0:
                comfy_temp_list.append(Tset)
                
            
        comfort_range[name]=comfy_temp_list
        
    
    
    
    # In[ ]:
    from functools import reduce
    import itertools
    
    
    #this function goes through the setpoints in which people feel comfortable. It outputs the common set of comfortable
    #comfortable temperatures as a result. In the case of there being no common value among users, it results in the median in 
    #all of the setpoints.
    def find_common_values(dict):
        # Initialize with first array
        common_values = set(dict[list(dict.keys())[0]])
    
        # Loop through each array and find common values
        for key in dict.keys():
            common_values = common_values.intersection(set(dict[key]))
    
        # If no common values, find common values for maximum number of arrays
        if len(common_values) == 0:
            max_common_count = 0
            for i in range(2, len(dict.keys())+1):
                for combination in itertools.combinations(dict.keys(), i):
                    intersection = set(dict[combination[0]])
                    for key in combination[1:]:
                        intersection = intersection.intersection(set(dict[key]))
                    if len(intersection) > max_common_count:
                        max_common_count = len(intersection)
                        common_values = intersection
    
        if len(common_values) == 0:
            all_values = []
            for key in dict.keys():
                all_values += dict[key]
            all_values.sort()
            median_index = len(all_values) // 2
            common_values = [all_values[median_index]]
        
        
        return list(common_values)
    
    
    def median(lst):
        sorted_lst = sorted(lst)
        lst_len = len(lst)
        mid = lst_len // 2
        if lst_len % 2 == 0:
            return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
        else:
            return sorted_lst[mid]
    
    if len(unique_names)>0:
        common_values=find_common_values(comfort_range)
        
        
        made_up_dict={'a1':[3,4,5,6,7], 'a2':[0,1,2],'a3':[12,13,14]}
        common_val=find_common_values(made_up_dict)
        
        opt_temp=median(common_values)
        opt_temp=float(opt_temp)
        print('active users', unique_names)
        #print('list of comfortable temperatures', common_values )
        print('This is the optimal temperature set point for this zone: ', opt_temp)
        
        with open('opt_temp.csv', mode="a", newline="") as file:
            writer = csv.writer(file)
        
        
            writer.writerow([opt_temp])
    else:
        print('no active users')

        
        
    k=k+1
    time.sleep(60)