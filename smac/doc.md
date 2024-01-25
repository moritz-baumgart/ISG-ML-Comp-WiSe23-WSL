## This file documents the different HP configuration I obtained using SMAC for the different Algorithms:

## !WARNING!: For all the configuration where I used SMOTE and cross validation: I originally just put SMOTE all the way in the beginning and let it generate new samples on all data and then cross validated. It turns that this causes a huge data leak resulting in very high validation scores, but I didn't notice/question the scores at first. I have rerun cross validation for all configurations where SMOTE was used (this time correctly by using SMOTE in each fold seperately) and noted that as a seperate score.


### HistGradientBoostingClassifier:
#### Config 1:
- HPOFacade with default CS using 10% TTS in training
- Val f1: 0.75 (using 10% TTS)
- Kaggle: 0.74178
```Python
Configuration(values={
  'l2_regularization': 0.010781990505819624,
  'learning_rate': 0.21667124489360576,
  'max_bins': 8,
  'max_depth': 19,
  'max_iter': 132,
  'max_leaf_nodes': 33,
  'min_samples_leaf': 3,
})
```


#### Config 2:
- MFFacede with default CS, min_budget=1, max_budget=200 (max_iter) using 10% TTS in training
- Val f1: 0.7797859932913799 5CV
- Kaggle: 0.75405
```Python
Configuration(values={
  'l2_regularization': 0.2983202368943038,
  'learning_rate': 0.12101433715577768,
  'max_bins': 143,
  'max_depth': 28,
  'max_leaf_nodes': 7,
  'min_samples_leaf': 34,
})
```


#### Config 3:
- MFFacede with default CS, min_budget=1, max_budget=200 (max_iter) using 5 fold CV in training
- Val f1 (5CV): 0.7972254828205283; With smote: 0.8629313232155944
- Kaggle: 0.75450; With smote: 0.78165

- Val Smote Fix (5CV): 0.801358634364895
```Python 
Configuration(values={
  'l2_regularization': 0.04426091526172612,
  'learning_rate': 0.27215534692918875,
  'max_bins': 95,
  'max_depth': 18,
  'max_leaf_nodes': 45,
  'min_samples_leaf': 3,
})
```

### GradientBoostingClassifier
- HPOFacade with default CS using 5-fold-CV
- Val f1 (5CV): 0.8302890584291795
- Kaggle: 0.72443 (worse than no HPO?, No HPO: 0.74407)

- Val Smote Fix (5CV): 0.7457166976714453
```Python
Configuration(values={
  'ccp_alpha': 0.0008966736741063057,
  'learning_rate': 0.4312350953167421,
  'loss': 'exponential',
  'max_depth': 33,
  'max_features': None,
  'max_leaf_nodes': 33,
  'min_impurity_decrease': 0.06360397078117137,
  'min_samples_leaf': 0.11120704068946179,
  'min_samples_split': 0.21042060422634004,
  'min_weight_fraction_leaf': 0.19354830546561044,
  'n_estimators': 188,
  'subsample': 0.9227895485953006,
})
```

### XGBClassifier

#### Config 1:
- HPOFacade with default CS using 5-fold-CV
- Val f1: 0.7843797286983205 (5CV)
- Kaggle: 0.75152

- Val Smote Fix (5CV): 0.768467286670429
```Python
Configuration(values={
  'colsample_bylevel': 0.6610844187123263,
  'colsample_bynode': 0.9368118870203981,
  'colsample_bytree': 0.9538224962398499,
  'gamma': 0,
  'learning_rate': 0.24629157183550024,
  'max_delta_step': 5,
  'max_depth': 8,
  'min_child_weight': 0,
  'reg_alpha': 5.6473477794995155,
  'reg_lambda': 19.276845193262904,
  'scale_pos_weight': 1.88126,
  'subsample': 0.5851337532795001,
})
```


#### Config 2:
- HPOFacade with default CS, expect not setting scale_pos_weight and instead oversampling using SMOTE
- using 5-fold-CV in training
- Val f1 (5CV): 0.8317572894299143
- Kaggle: 0.76295

- Val Smote Fix (5CV): 0.7732125656038009
```Python
Configuration(values={
  'colsample_bylevel': 0.5176886775686282,
  'colsample_bynode': 0.7282438085663235,
  'colsample_bytree': 0.5431252531168697,
  'gamma': 0,
  'learning_rate': 0.9530582731900737,
  'max_delta_step': 1,
  'max_depth': 6,
  'min_child_weight': 2,
  'reg_alpha': 3.4244255197855935,
  'reg_lambda': 84.15175046900218,
  'subsample': 0.7185255201070447,
})
```


#### Config 3:
- HPOFacade with default CS, expect not setting scale_pos_weight and instead oversampling using SMOTE
- using 5-fold-CV in training
- Val f1 (5CV): 0.8700117470440899
- Kaggle: 0.77848

- Val Smote Fix (5CV): 0.8011912482858309
```Python
Configuration(values={
  'colsample_bylevel': 0.6543301384218317,
  'colsample_bynode': 0.9439972406076952,
  'colsample_bytree': 0.7773758702787967,
  'gamma': 0,
  'learning_rate': 0.1827843463017609,
  'max_delta_step': 4,
  'max_depth': 8,
  'min_child_weight': 0,
  'reg_alpha': 0.1895331843304295,
  'reg_lambda': 0.6188481066931769,
  'subsample': 0.9979218768593112,
})
```


#### Config 4:
- see C3
- Val f1 (5CV): 0.87585449315288
- Kaggle: 0.76863

- Val Smote Fix (5CV): 0.7997504678009676
```Python
Configuration(values={
  'colsample_bylevel': 0.9597115594169324,
  'colsample_bynode': 0.7358832260110567,
  'colsample_bytree': 0.5676124568264106,
  'gamma': 0,
  'learning_rate': 0.19792336150129256,
  'max_delta_step': 10,
  'max_depth': 7,
  'min_child_weight': 0,
  'reg_alpha': 0.0075600180074234325,
  'reg_lambda': 0.1345830108991683,
  'subsample': 0.9996182263793291,
})
```



### XGBRFClassifier
#### Config 1:
- HPOFacade with default CS using 5-fold-CV
- Val f1: 0.6877405518734403 (5CV)
- Kaggle: 0.66308

- Val Smote Fix (5CV): 0.5146328549260736
```Python
Configuration(values={
  'colsample_bylevel': 0.8210993183179356,
  'colsample_bynode': 0.8856575595180459,
  'colsample_bytree': 0.5816487543331568,
  'gamma': 2,
  'learning_rate': 0.49163590844348753,
  'max_delta_step': 2,
  'max_depth': 9,
  'min_child_weight': 0,
  'reg_alpha': 86.50751030451893,
  'reg_lambda': 81.4229982909532,
  'scale_pos_weight': 1.88126,
  'subsample': 0.8442617982595704,
})
```

#### Config 2:
- HPOFacade with default CS
- using 5-fold-CV
- Val f1: 0.7106394304955221 (5CV); with smote: 0.8038994667982259
- Kaggle: 0.67205; with smote: 0.72534

- Val Smote Fix (5CV): 0.7422799810845181
```Python
Configuration(values={
  'colsample_bylevel': 0.36308877196546113,
  'colsample_bynode': 0.78451794494519,
  'colsample_bytree': 0.7464462511201432,
  'gamma': 3,
  'learning_rate': 0.4405209483222396,
  'max_delta_step': 2,
  'max_depth': 11,
  'min_child_weight': 0,
  'reg_alpha': 15.189965200387235,
  'reg_lambda': 2.288238795615241,
  'subsample': 0.9282325161585102,
})
```

#### Config 3:
- HPOFacade with default CS
- using 5-fold-CV
- Val f1: 0.7803239201319048 (5CV); with smote: 0.8584743040322422
- Kaggle: 0.75156; with smote: 0.8149

- Val Smote Fix (5CV): 0.79032021532011
```Python
Configuration(values={
  'colsample_bylevel': 0.9775917731221307,
  'colsample_bynode': 0.8013717446667272,
  'colsample_bytree': 0.5388125463997465,
  'gamma': 2,
  'learning_rate': 0.5081032972297059,
  'max_delta_step': 9,
  'max_depth': 12,
  'min_child_weight': 0,
  'reg_alpha': 0.2781818137506308,
  'reg_lambda': 0.0096988404529888,
  'subsample': 0.9276082022708639,
})
```



### CatBoostClassifier
#### Config 1:
- HPOFacade with default CS, SMOTE, outliers min max, drop ft2
- smac using 90/10 TTS
- Val f1 (5CV): 0.8633702552510524
- Kaggle: 0.75805

- Val Smote Fix (5CV): 0.7884727429183427
```Python
values = {
    "bagging_temperature": 9.933171093235632,
    "depth": 4,
    "l2_leaf_reg": 5.420115711716861,
    "random_strength": 0.3008985550781157,
}
```


### MLPClassifier
#### Config 1:
- HPOFacade with default CS, SMOTE, StdScale, outliers min max, drop ft2
- 3 CV in smac training
- Val f1 (5CV): 0.8288192201539892
- Kaggle: 0.73306

- Val Smote Fix (5CV): 0.7467376623140715
```Python
values={
  'activation': 'logistic',
  'alpha': 0.0006796700786147267,
  'hidden_layer_sizes': 108,
  'learning_rate': 'adaptive',
  'learning_rate_init': 0.02899177860096097,
  'momentum': 0.1650630459189415,
  'solver': 'adam',
}
```



## Default CS:
When it says "default CS" somewhere above, the following configurations were used:


### For HistGradientBoostingClassifier:
```Python
ConfigurationSpace(
  {
    "learning_rate": (0.01, 0.3),
    # "max_iter": (50, 200), # set by smac instead
    "max_depth": (3, 50),
    "min_samples_leaf": (1, 50),
    "max_leaf_nodes": (2, 50),
    "l2_regularization": (0.0, 0.3),
    "max_bins": (5, 255),
  }
)
```


### For XGBClassifier and XGBRFClassifier:
```Python
ConfigurationSpace(
  {
    "colsample_bynode": (0.001, 1.0),
    "learning_rate": (0.0, 1.0),
    "reg_lambda": (0.0, 100.0),  # TODO: Look more into this one
    "subsample": (0.001, 1.0),
    "colsample_bylevel": (0.001, 1.0),
    "colsample_bytree": (0.001, 1.0),
    "gamma": (0, 20),
    "max_delta_step": (0, 10),
    "max_depth": (1, 12),
    "min_child_weight": (0, 10),
    "reg_alpha": (0.0, 100.0),  # TODO: Look more into this one
    "scale_pos_weight": 1.88126,  # sum(negative instances) / sum(positive instances) = 1014 / 539
  }
)
```
