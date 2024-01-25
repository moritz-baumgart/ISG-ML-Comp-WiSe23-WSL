from mlcomp.data.load import load_classification_train
from sklearn.model_selection import train_test_split
from mlcomp.data.preprocess import drop_ft2, remove_outliers
from autogluon.tabular import TabularPredictor

df = load_classification_train()
df = drop_ft2(df)
df = remove_outliers(df)

train, test = train_test_split(df, test_size=0.1)

predictor = TabularPredictor('label').fit(train, presets='best_quality')

print(predictor.evaluate(test))
