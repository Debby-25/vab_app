import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold,StratifiedKFold ,GroupKFold
from sklearn.metrics import mean_squared_error


import warnings
warnings.filterwarnings('ignore')

def get_processed_data(train_path,test_path) :
    
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    data = pd.concat([train, test]).reset_index(drop=True)
    
    col = ['country', 'year', 'urban_or_rural']
    
    ## Count of unique features
    for i in col:
        data['count_'+i] = data[i].map(data[i].value_counts())
        
    ## Combination features
    data['all_ghsl'] = data['ghsl_built_1975_to_1990']+data['ghsl_built_pre_1975']+data['ghsl_built_1990_to_2000']+data['ghsl_built_2000_to_2014']
    data['all_landcover_fraction'] = data['landcover_crops_fraction']+data['landcover_urban_fraction']
    data['all_waters'] = data['landcover_water_permanent_10km_fraction'] + data['landcover_water_seasonal_10km_fraction']
    
    # get train , test
    train = data[data['ID'].isin(train['ID'].values)]
    test = data[~data['ID'].isin(train['ID'].values)]
    features = [x for x in train.columns if x not in 
                ['ID','country','urban_or_rural','Target','year']]
    return train , test , features
     

train_path = 'Train.csv' ; test_path = 'Test.csv'
train , test , features = get_processed_data(train_path,test_path)
     

def get_model(Name='lgbm') :
    if Name=='lgbm' :
      return LGBMRegressor({'objective' :'regression','boosting_type' : 'gbdt','metric': 'rmse' ,
                              'learning_rate' : 0.05,'num_iterations': 1500,'max_depth' :4 ,'num_leaves' : 150,
                              'max_bins': 85,'min_data_in_leaf':30,'reg_lambda' :75})
     

Model_Name = "lgbm"

X = train[features]
y = train['Target']
test_ = test[features]

folds = KFold(n_splits=10, shuffle=True, random_state=2021)
oofs  = np.zeros((len(X)))
test_predictions = np.zeros((len(test)))


for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    
    X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    

    clf = get_model(Name=Model_Name)
    clf.fit(X_trn, y_trn, eval_set = [(X_val, y_val)],
            verbose = 0, early_stopping_rounds = 50)
    
    vp = clf.predict(X_val)
    oofs[val_idx] = vp
    val_score = mean_squared_error((vp), (y_val),squared=False)
    print(4*'-- -- -- --')
    print(f'Fold {fold_+1} Val score: {val_score}')
    print(4*'-- -- -- --')
    
    tp = clf.predict(test_)
    test_predictions += tp / folds.n_splits

print()
print(3*'###',10*"^",3*'###')
print(mean_squared_error(y, oofs,squared=False))

submission = pd.DataFrame()
submission['ID'] = test['ID']
submission['Target'] = np.clip(test_predictions, 0.141000, 0.808657)
     

dir_path = 'LearningSolutionV3'
submission.to_csv(f'{dir_path}.csv',index=False)