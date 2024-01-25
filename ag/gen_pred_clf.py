from autogluon.tabular import TabularPredictor
from mlcomp.data.load import load_classification_test
from mlcomp.data.preprocess import drop_ft2, remove_outliers
import pandas as pd

predictor = TabularPredictor.load("AutogluonModels/ag-20231219_025650")

print(predictor.model_best)

df = load_classification_test()
df = drop_ft2(df)

p = predictor.predict(df)

res = pd.DataFrame(p)

res.to_csv('pred.csv', index_label='Id')