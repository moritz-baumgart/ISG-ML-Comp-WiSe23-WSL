## Models:

### Classification
#### ag-20231219_023641:
- My first try with autogluon
- 90/10 train/test split
- eval_metric='f1_macro'
- Nothing else

#### ag-20231219_024849
- Same as ag-20231219_023641, except eval_metric was not set at all

#### ag-20231219_025650
- presets='best_quality'
- df = drop_ft2(df)
- df = remove_outliers(df)


### Regression
#### ag-20240116_230702
- Basic autogluon with no specfic settings
- 90/10 TTS: 
- Remove everything prior 01-01-2014
- AutoGluon infered "multiclass" might try to set to regression myself

#### ag-20240116_233657
- Same as ag-20240116_230702, except:
  - eval_metric="root_mean_squared_error"
  - problem_type="regression"


#### ag-20240117_002332
- Same as ag-20240116_233657, but:
  - presets="best_quality"
  - time_limit=7.5 * 60 * 60 (7.5h)


  