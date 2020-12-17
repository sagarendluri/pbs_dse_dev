from sklearn.metrics import make_scorer
import os
import json
import keras
import boto3
#import allel
import joblib
import sklearn
import pymysql
import tempfile
import argparse
import sqlalchemy
from os.path import splitext
import numpy as np
import pandas as pd
from math import sqrt
from io import StringIO
from sklearn.impute import SimpleImputer
from smart_open import smart_open 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from configparser import ConfigParser
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from xgboost.sklearn import XGBRegressor,XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.metrics import mean_squared_error ,mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from keras.callbacks import EarlyStopping
from keras import backend
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from dateutil.parser import parse
from scipy.stats import norm
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import confusion_matrix
import numpy
############### h2o.save_model(aml.leader, path="./product_backorders_model_bin")
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.constraints import maxnorm
from keras.layers import Dropout
from sklearn.svm import SVR,SVC
class Model():
    def __init__(self,*args,**kwargs):
        self.df1 = df1
        self.i = i
        self.sklearn = sklearn
        self.ai = ai
        self.ini=  ini
        self.test_size = 0.2
        self.Regressor = [RandomForestRegressor(),KNeighborsRegressor(),XGBRegressor(),SVR()]
        self.Classifier = [RandomForestClassifier(),KNeighborsClassifier(),XGBClassifier(),SVC()]
        self.config_object = ConfigParser()
        self.config_object.read(self.ini)
        self.Regressor_grids =[{   
                            'max_depth': [int(x) for x in np.linspace(1, 45, num = 3)],
                            'max_features': ['auto', 'sqrt'],
                            'min_samples_split': [5, 10],
                            'n_estimators': [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]},
                            {'n_neighbors': np.arange(1, 25)},
                            {'objective':['reg:linear'],
                              'learning_rate': [0.045], 
                              'max_depth': [3,4],
                              'min_child_weight': [2],
                              'silent': [1],
                              'subsample': [0.5],         
                              'n_estimators': [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]},
                              {'C' : [0.001, 0.01, 0.1, 1, 10],
                                'gamma':[0.001, 0.01, 0.1, 1],
                              'kernel': ['linear']}]
        self.Classifiers_grids = [self.Regressor_grids[0],
                                  self.Regressor_grids[1],
                                  {'model__learning_rate': [0.045], 
                                    'model__max_depth': [3,4],
                                    'model__min_child_weight': [2],
                                    'model__silent': [1],
                                    'model__subsample': [0.5],
                                    'model__n_estimators': [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]},
                                 self.Regressor_grids[3]]
        drop_cols = ['Unnamed: 0','alleles','chrom','pos', 'strand','assembly#','center','protLSID',
                   'assayLSID','panelLSID','QCcode']
        drop_cols.append(i)
        cols = df.columns.intersection(drop_cols)
        self.config_object.read(self.ini)
        self.userinfo = self.config_object["MYSQL"]
        self.password= self.userinfo["password"]
        self.user= self.userinfo["user"]
        self.host= self.userinfo["host"]
        self.db= self.userinfo["db"]
        self.access_key_id = self.userinfo["access_key_id"]
        self.secret_access_key  = self.userinfo["secret_access_key"]
        self.bucket  = self.userinfo["bucket"]
#         df1[self.i] = df1[self.i]
#         df1[self.i] = (df1[self.i].cat.codes.replace(-1, np.nan).interpolate().astype(int).astype('category')
#                         .cat.rename_categories(df1[self.i].cat.categories))
#         cols = [self.i]
#         print(self.i)
#         df1.loc[:,'Survived'] = df1.loc[:,'Survived'].ffill()
        self.x = df1.drop(cols,axis = 1)
        self.y = df1[self.i]
    def dtypes_handliing(self):
        print(self.y.unique())
        try:
            print("Data analyzing")
            # Select the bool columns
            self.bool_col = self.x.select_dtypes(include='bool')
            # select the float columns
            self.float_col = self.x.select_dtypes(include=[np.float64])
            # select the int columns
            self.int_col = self.x.select_dtypes(include=[np.int64])
        #   # select non-numeric columns
            self.cat_col = self.x.select_dtypes(include=['category', object])
            self.date_col = self.x.select_dtypes(include=['datetime64'])
            return self.bool_col,self.float_col,self.int_col,self.cat_col,self.date_col
        except:
            print('Data analyzing failed')
    def handiling_categorical(self):
        try:
            self.cat_result = []
            for col in list(self.cat_col):
                labels, levels = pd.factorize(self.cat_col[col].unique())
                if sum(labels) <= self.cat_col[col].shape[0]:
                    self.cat_result.append(pd.get_dummies(self.cat_col[col], prefix=col))
            print("Categorical Data analyzing")
            return self.cat_result
        except:
            print("Categorical Data analyzing failed")
    def handiling_int_col(self):
        try:
            print("int Data analyzing")
            self.int_result = []
            for col in self.int_col:
                labels, levels = pd.factorize(self.int_col[col].unique())
                if len(labels) == self.int_col[col].shape[0]:
                    re = self.int_col.drop([col],axis=1)
                else:
                    self.int_result.append(self.int_col[col])
            return self.int_result
        except:
            print("int Data analyzing failed")
    def concat_cat(self):
        result = [self.cat_result]        
        for fname in result:
            if fname == []:
                print('No objects to concat')
            else:
                self.data =  pd.concat([col for col in fname],axis=1)
                self.cleaned_Data_frm = pd.concat([self.data.reindex(self.y.index)], axis=1)
                print(list(self.cleaned_Data_frm.columns))
                return self.cleaned_Data_frm
    def concat_int(self):
        result2 = [self.int_result]
        for fname2 in result2:
            if fname2 == []:
                print('No int_cols to concat')
            else:
                self.data2 =  pd.concat([col for col in fname2],axis=1)
                self.cleaned_Data_frm1 = pd.concat([self.data2.reindex(self.y.index)], axis=1)
                return self.cleaned_Data_frm1
    def encoder(self):
        if(self.y.dtype == object or self.y.dtype == bool):
            self.y = pd.to_numeric(self.df1[self.i], errors='coerce').fillna(0)
            self.y = self.y.astype(int)
            print(self.y.unique())
            label_encoder = preprocessing.LabelEncoder()
            self.y= label_encoder.fit_transform(self.y.astype(str))
            self.dataset = pd.DataFrame()
            self.dataset[self.i] = self.y.tolist()
            print('Multiclass_classification')
            self.types = 'Classification_problem'
            return self.dataset[self.i]
        elif self.y.dtypes == np.int:
            self.types = 'Classification_problem'
            print('Classification_problem')
            return self.y
        else:
            print('Regression_problem')
            return self.y
    def localdb(self):
        conn = pymysql.connect(host=(', '.join(["%s" % self.host])),
                   port=int(3306),
                   user=(', '.join(["%s" % self.user])),
                   passwd=(', '.join(["%s" % self.password])),
                   db=(', '.join(["%s" % self.db])), charset='utf8mb4')
        print("db connection success")
        cursor=conn.cursor()
        return cursor ,conn
    def QC(self,cleaned_Data_frm, cleaned_Data_frm1,y,cursor ,conn):
#         try:
            print('Models Building')
            float_cols = self.float_col
            result = pd.concat([cleaned_Data_frm,cleaned_Data_frm1,y,float_cols], axis=1)
            self.data_sorted1 = result.sort_values(self.i)
            self.data_sorted = self.data_sorted1.loc[:,~self.data_sorted1.columns.duplicated()]
            print(self.data_sorted.shape)
            new_list = [list(set(self.x.columns).difference(self.data_sorted.columns))]
            uploaded_cols = []
            for self.col in list(self.data_sorted.select_dtypes(include=[np.float64])):
                if not self.col in uploaded_cols:
                    print(self.col)
                    X = self.data_sorted.drop([self.col],axis=1)
                    Y = self.data_sorted[self.col]
                    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 42)
                    X_train, X_test = train_test(X_train, X_test)
                    print(X_train.shape)
                    Modles_reuslts = []
                    Names = []    
                    target = self.col
                    print('Models Building')
                    models = ['Random Forest','KNN','XGB','SVR']
                    l=0
                    features  = []
                    print(Y.unique())
                    X= X.fillna(X.mean())
                    y = (', '.join(["%s" % self.i]))
                    print(y)
                    cols = list(self.data_sorted.columns)
                    x = cols
                    x.remove(y)
                    if  'sklearn'== (', '.join(["%s" %  self.sklearn])):
                        print("good")
                        def sklearn(X,Y,algos):
                            model = models[algos]
                            X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
                            gd = RandomizedSearchCV(self.Regressor[algos],self.Regressor_grids[algos],cv = 5, n_jobs=-1,
                                                    verbose=True,refit = True)
                            gd.fit(X_train, y_train)
                            y_pred = gd.predict(X_test)
                            random_best= gd.best_estimator_.predict(X_test)
                            errors = abs(random_best - y_test)
                            mape = np.mean(100 * (errors / y_test))
                            Accuracy = 100 - mape 
                            grid = gd.best_params_
                            estimator = gd.best_estimator_
                            print(grid)
                            if model=='KNN':
                                perm = PermutationImportance(gd, random_state=1).fit(X_train,y_train)      
                                importances = perm.feature_importances_
                                DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
                                           importances,
                                          grid,estimator,l,None,target,model)
                            elif model == 'SVR':
                                
                                weights = gd.best_estimator_.coef_
                                test=' '.join(str(weights).split())
                                
                                #replace double whitespace with comma
                                   
                                test=test.replace(" ",",")
                                #str to json loads
                                lists = json.loads(test)
                                importances = lists[0]                                             
                                DB_upload(Accuracy,X_train,X_test,y_test,
                                              y_pred, importances,
                                              grid,estimator,l,None,target,model)
                            else:
                                importances = gd.best_estimator_.feature_importances_.tolist()#._final_estimator
                                print(Accuracy)
                                features.append(importances)
                                DB_upload(Accuracy,X_train,X_test,y_test, y_pred,importances,
                                                 grid,estimator,l,None,target,model)
                                feature_list = list(X_train.columns)
                                #create a list of tuples
                                feature_importance= sorted(zip(importances, feature_list), reverse=True)
                                                    #create two lists from the previous list of tuples
            #                     dff = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
            #                     dff.to_csv('top5000_feature_importance.csv')
            #                     DB_upload(Accuracy,X_train,X_test,y_test, y_pred,importances,grid,estimator,l,
            #                               cm,target,model)
            #                     encoded_classes = list(self.cleaned_Data_frm)
                            return Accuracy
                        sklearn(X,Y,algos)
                    elif 'ai' == (', '.join(["%s" %  self.ai])):
                        print('H2o')
                        def H2o(x,y):
                            df = h2o.H2OFrame(self.data_sorted)
                            train,  test = df.split_frame(ratios=[.8])
                            train[y] = train[y].asfactor()
                            test[y] = test[y].asfactor()

                            # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
                            aml = H2OAutoML(max_models=10, seed=1)
                            aml.train(x=x, y=y, training_frame=train)

                            # View the AutoML Leaderboard
                            lb = aml.leaderboard
                            print(lb.head(rows=lb.nrows))
                            return lb
                        H2o(x,y)
                    else:
                        def Reg_model():
                            model = Sequential()
                            model.add(Dense(500, input_dim=X_train.shape[1], activation= "relu"))
                            model.add(Dense(100, activation= "relu"))
                            model.add(Dense(50, activation= "relu"))
                            model.add(Dense(1))
                            model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["accuracy"])
                            return model
                        model = KerasClassifier(build_fn=Reg_model, verbose=0)
                        # define the grid search parameters
                        batch_size = [10, 20, 40, 60, 80, 100]
                        epochs = [10, 50, 100]
                        param_grid = dict(batch_size = batch_size, epochs = epochs)
                        grid = GridSearchCV(estimator = model, param_grid=param_grid, n_jobs=-1, cv=3)
                        grid_result = grid.fit(X_train, y_train)
                        grid = grid.best_params_
                        model = 'DNN'
                        print("DNN",features)
                        DB_upload(Accuracy,X_train,X_test,y_test,y_pred,features[0],
                                  grid,grid,l,None,target,model)
                        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #         except:
    #             print("Rgeression model not buidling")
    def classification(self,cleaned_Data_frm1, cleaned_Data_frm,y,cursor ,conn):
#         try:
            Modles_reuslts =[]
            Names = []
            print("Model building")
            float_cols = self.float_col
            result = pd.concat([cleaned_Data_frm1,cleaned_Data_frm,y,float_cols], axis=1)
            self.data_sorted1 = result.loc[:,~result.columns.duplicated()]
            self.data_sorted2 =  self.data_sorted1.sort_values(self.i)
            self.data_sorted = self.data_sorted2.dropna(thresh=self.data_sorted2.shape[0]*0.5,how='all',axis=1)
            self.data_sorted  = self.data_sorted.dropna()
            new_list = [list(set(self.data_sorted.columns).difference(self.x.columns))]
            X = self.data_sorted.drop([self.i],axis=1)
            print(X.shape)
            Y = self.data_sorted[self.i]
            print(Y.unique())
            X= X.fillna(X.mean())
            y = (', '.join(["%s" % self.i]))
            print(y)
            cols = list(self.data_sorted.columns)
            x = cols
            x.remove(y)
#             X_train, X_test = train_test(X_train, X_test)
            # List of pipelines for ease of iteration
            l = 0
            access_key_id = self.access_key_id 
            secret_access_key = self.secret_access_key
            models = ['Random Forest','KNN','XGB','SVC']
#             model = models[algos]
#             library = (', '.join(["%s" % sklearn]))
            if  'sklearn'== (', '.join(["%s" %  self.sklearn])):
                print("good")
                def sklearn(X,Y,algos):
                    model = models[algos]
                    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
                    gd = RandomizedSearchCV(self.Classifier[algos],self.Classifiers_grids[algos],cv = 5, n_jobs=-1,
                                            verbose=True,refit = True)
                    gd.fit(X_train, y_train)
                    grid = gd.best_params_
                    estimator = gd.best_estimator_
                    y_pred=gd.predict(X_test)
                    cm =confusion_matrix(y_test, y_pred)
                    target = self.i
                    Accuracy = metrics.accuracy_score(y_test, y_pred)
                    print(cm)
                    print(grid)
                    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
                    if model=='KNN':
                        perm = PermutationImportance(gd, random_state=1).fit(X_train,y_train)      
                        importances = perm.feature_importances_
        #                     DB_upload(Accuracy,X_train,X_test,y_test,y_pred,
        #                                importances,grid,estimator,l,cm,target,model)
                    elif model == 'SVC':
                        importances = gd.best_estimator_.coef_
                        imp = importances.tolist()
                        importances = imp[0]
        #                     DB_upload(Accuracy,X_train,X_test,y_test,y_pred, 
        #                               importances,grid,estimator,l,cm,target,model)
                    else:
                        importances = gd.best_estimator_.feature_importances_.tolist()
                        #create a feature list from the original dataset (list of columns)
                        # What are this numbers? Let's get back to the columns of the original dataset
                        feature_list = list(X_train.columns)
                        #create a list of tuples
                        feature_importance= sorted(zip(importances, feature_list), reverse=True)
                                            #create two lists from the previous list of tuples
    #                     dff = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
    #                     dff.to_csv('top5000_feature_importance.csv')
    #                     DB_upload(Accuracy,X_train,X_test,y_test, y_pred,importances,grid,estimator,l,
    #                               cm,target,model)
    #                     encoded_classes = list(self.cleaned_Data_frm)
                    return Accuracy
                sklearn(X,Y,algos)
            elif 'ai' == (', '.join(["%s" %  self.ai])):
                print('H2o')
                def H2o(x,y):
                    df = h2o.H2OFrame(self.data_sorted)
                    train,  test = df.split_frame(ratios=[.8])
                    train[y] = train[y].asfactor()
                    test[y] = test[y].asfactor()

                    # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
                    aml = H2OAutoML(max_models=10, seed=1)
                    aml.train(x=x, y=y, training_frame=train)

                    # View the AutoML Leaderboard
                    lb = aml.leaderboard
                    print(lb.head(rows=lb.nrows))
                    return lb
                H2o(x,y)
            else:
                    print('Dnn')
                    if self.types == 'Classification_problem':
                        def DNN():
                            model = Sequential()
                            model.add(Dense(512, input_dim=X_train.shape[1], init='normal', activation='relu'))
                            model.add(BatchNormalization())
                            model.add(Dropout(0.5))
                            model.add(Dense(32, init='normal', activation='relu'))
                            model.add(BatchNormalization())
                            model.add(Dropout(0.5))
                            model.add(Dense(1, init='normal', activation='sigmoid'))
                            model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
                            return model
                        X = self.data_sorted.drop([self.i],axis=1)
                        Y = self.data_sorted[self.i]
                        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)
                    
                        classifier = KerasClassifier(build_fn=DNN, verbose=1)
                        batch_size = [10 ,20, 40, 60, 80, 100]
                        epochs = [10, 50, 100]
                        param_grid = dict(batch_size=batch_size, epochs=epochs)
                        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
                        grid_result = grid.fit(X_train, y_train)
                        estimator = grid.best_estimator_
                        Accuracy= grid_result.best_score_
                        print("%s" % (estimator))
                        perm = PermutationImportance(grid, scoring='accuracy', random_state=1).fit(X_train,y_train)      
                        print(perm.feature_importances_)
                        DB_upload(Accuracy,X_train,X_test,y_test, y_pred,importances,grid,estimator,l,
                                      cm,target,model)
                        # summarize results
                        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                        
                    else:
                        a = np.unique(self.y)
                        a.sort()
                        b=a[-1]
                        b +=1
                        def DNN(dropout_rate=0.0, weight_constraint=0):
                            # create model
                            model = Sequential()
                            model.add(Dense(42, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
                            model.add(Dropout(dropout_rate))
                            model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
                            model.add(Dense(b,activation='softmax'))
                            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                            return model
                        classifier = KerasClassifier(build_fn=DNN, epochs=50, batch_size=10, verbose=1)
                        weight_constraint = [1, 2, 3, 4, 5]
                        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
                        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, cv=3)
                        grid_result = grid.fit(X_train, y_train)
                        estimator = grid.best_estimator_
                        Accuracy= grid_result.best_score_
                        print(Accuracy)
                        DB_upload(Accuracy,X_train,X_test,y_test,y_pred,importances,grid,estimator,l,
                                  cm,target,model)
                        print("%s" % (estimator))
    #         except:
    #             print('Regression model building failed')

def train_test(X_train,X_test):
    try:
        vs_constant = VarianceThreshold(threshold=0)
        # select the numerical columns only.
        numerical_x_train = X_train[X_train.select_dtypes([np.number]).columns]
        # fit the object to our data.
        vs_constant.fit(numerical_x_train)
        # get the constant colum names.
        constant_columns = [column for column in numerical_x_train.columns
                            if column not in numerical_x_train.columns[vs_constant.get_support()]]
        # detect constant categorical variables.
        constant_cat_columns = [column for column in X_train.columns 
                                if (X_train[column].dtype == "O" and len(X_train[column].unique())  == 1 )]
        all_constant_columns = constant_cat_columns + constant_columns
        X_train.drop(labels=all_constant_columns, axis=1, inplace=True)
        X_test.drop(labels=all_constant_columns, axis=1, inplace=True)
        print(X_train.shape)
        # threshold value for quasi constant.
        ####### Quasi-Constant Features
        threshold = 0.98
        # create empty list
        quasi_constant_feature = []
        # loop over all the columns
        for feature in X_train.columns:
            # calculate the ratio.
            predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
            # append the column name if it is bigger than the threshold
            if predominant >= threshold:
                quasi_constant_feature.append(feature) 
        X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        X_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        print(X_train.shape)
        #######Duplicated Features
        # transpose the feature matrice
        train_features_T = X_train.T
      ########  Correlation Filter Methods
        # select the duplicated features columns names
        duplicated_columns = train_features_T[train_features_T.duplicated()].index.values
        # drop those columns
        X_train.drop(labels=duplicated_columns, axis=1, inplace=True)
        X_test.drop(labels=duplicated_columns, axis=1, inplace=True)
        print(X_train.shape)
        #correlated_features = set()
        #correlation_matrix = X_train.corr()
        #for i in range(len(correlation_matrix .columns)):
         #   for j in range(i):
          #      if abs(correlation_matrix.iloc[i, j]) > 0.8:
           #         colname = correlation_matrix.columns[i]
            #        correlated_features.add(colname)
        #X_train.drop(labels=correlated_features, axis=1, inplace=True)
        #X_test.drop(labels=correlated_features, axis=1, inplace=True)
        #print(X_train.shape)
        return X_train,X_test
    except:
        print('sucsessfully completed QC')
def DB_upload(Accuracy,X_train,X_test,y_test,y_pred,importances,grid,estimator,l,
              cm,target,model):
    x_train_means = X_train.mean(axis=0)
    print(x_train_means)
    request_payload_data = {}
    data = X_train[:].values.T
    data1 = pd.DataFrame(data)
    print(data1.shape)
    print(X_train.shape)
    print(X_train.values.shape)
    print(data.shape)
    print("importance",importances)
    columnsNameslist = list(X_train.columns)
    for col_index in range(0,len(list(X_train.columns))):
        data = data1.values[col_index]
        data2 = (data - data.min()) / (data.max() - data.min())
        hist = plt.hist(data2)
        request_payload_data[columnsNameslist[col_index]] = {
            'mean': x_train_means[col_index],
            'importance': importances[col_index],
            'normalised_data_distribution': list(hist[0])
        }
    performance_matrix = {
            'Samples in training set': len(X_train),
            'Samples in test set': len(X_test),
        }
    gt = y_test
    key = str(target)+'.pkl'
    performance_matrix['Mean Absolute Error (MAE)'] = metrics.mean_absolute_error(gt, y_pred)
    performance_matrix['Mean Squared Error (MSE)'] = metrics.mean_squared_error(gt, y_pred)
    performance_matrix['Root Mean Squared Error (RMSE)'] = np.sqrt(metrics.mean_squared_error(gt, y_pred))
    print(Accuracy)
    cr = {"performance":{"Algorithm": model,
                          "performance_matrix": performance_matrix,
                          },
          "Accuracy":Accuracy,
          "confusion matrix": cm,
          "grid": grid,
          "Estimator": estimator,
          "model_file_name": key,
          "request_payload_data": request_payload_data,
          "dataset_name":dname
          }
    mySql_insert_query = """INSERT INTO Phenotype_model_info(model_file_path,
                                        confusion_matrix, hyperparameter_grid, best_estimator,
                                        dataset_name, request_payload,Phenotypes_id_id, performence)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) """
    request_payload_data= request_payload_data
    esti = {'estimator':str(estimator)}
    cm = {'confusion_metrics':cm.tolist()}
    
    cr = {'Total_perfomence':str(cr)}
    grid = {'Best_grid':str(grid)}

    model_file_name  =model+'.pkl'
#     joblib.dump(estimator, model_file_name)
   # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=access_key_id,
                             aws_secret_access_key=secret_access_key)
#     s3_client.upload_file(model_file_name,bucket , model_file_name)
    print('sucsess')
    recordTuple = (
                    model, json.dumps(cm), 
                    json.dumps(grid),
                    json.dumps(esti), 
                    dname, json.dumps(request_payload_data),
                    l,json.dumps(cr)
    )
    cursor.execute(mySql_insert_query, recordTuple)
    conn.commit()
if __name__ == '__main__':
#     try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--target", help="enter target feature", type = str)
        parser.add_argument("--dataset", help="enter dataset name", type = str)
        parser.add_argument("--algorithm", help="algorithm name", type = int)
        parser.add_argument("--sklearn", help="sklearn", type = str)
        parser.add_argument("--ai", help="h2o", type = str)
        args = parser.parse_args()
        i=args.target
        global sklearn
        sklearn=args.sklearn
        global ai
        ai = args.ai
        global dname
        dname = args.dataset
        global algos
        algos = args.algorithm
        config_object = ConfigParser()
        ini = config_object.read(r'config.ini')
        config_object.read(ini)
        userinfo = config_object["MYSQL"]
        global access_key_id
        global secret_access_key
        access_key_id = userinfo["access_key_id"]
        secret_access_key  = userinfo["secret_access_key"]
        bucket  = userinfo["bucket"]
        object_key = (', '.join(["%s" % dname]))
        tar = (', '.join(["%s" % i]))
        bucket_name = bucket
        file_name,extension = splitext(object_key)
        path = 's3://{}:{}@{}/{}'.format(access_key_id, secret_access_key, bucket_name, object_key)
        if extension == '.csv':
            df = pd.read_csv(smart_open(path))
            print("file type is csv")
            df.isnull().sum()
            df1 = df.fillna(df.mean())
            df.isnull().sum()
            df1 = df1.sort_values(i)
            print(df1)
            model_instance = Model(df1,i,ini = ini,sklearn = sklearn,ai= ai)
            model_instance.dtypes_handliing()
            print(model_instance.handiling_categorical())
            model_instance.handiling_int_col()
            cleaned_Data_frm= model_instance.concat_cat()
            cleaned_Data_frm1= model_instance.concat_int()
            print(cleaned_Data_frm)
            y = model_instance.encoder()
            cursor ,conn = model_instance.localdb()
            print(y)
            if y.dtypes == 'int64' or y.dtypes =='int32':

                model_instance.classification(cleaned_Data_frm, cleaned_Data_frm1,y,cursor ,conn )
            else:
                model_instance.QC(cleaned_Data_frm, cleaned_Data_frm1,y,cursor ,conn ) 
        elif extension == '.vcf':
            callset = allel.read_vcf(smart_open(path))
            snps = callset['variants/ID']
            da = callset['calldata/GT']
            data = da.transpose([2,0,1]).reshape(-1,da.shape[1])
            df = pd.DataFrame(data)
            a = len(df)
            h = int(a/2)
            sm = callset['samples']
            def split(df) :
                hd = df.head(h)
                tl = df.tail(len(df)-h)
                return hd,tl
            # Split dataframe into top 3 rows (first) and the rest (second)
            heads,tails  = split(df)
            df = pd.DataFrame(data)
            heads,tails  = split(df)
            df = pd.DataFrame(heads)
            df1 = pd.DataFrame(tails)
            df.columns = sm
            df1.columns = sm
            df['snps'] = snps
            df1['snps'] = snps
            sum_df = df.set_index('snps').add(df1.set_index('snps'), fill_value=0).reset_index()
            sum_df1 = df.iloc[:h]
            sum_df2 = df.iloc[h:]
            name1 = str(h)+'heads_snps.csv'
            name2 = str(h)+'tails_snps.csv'
            sum_df1.to_csv(name1)
            sum_df2.to_csv(name2)
            print("vcffile")
        else:
            print("No input file name")
#     except:
#         print("Please enter valid target feature")
