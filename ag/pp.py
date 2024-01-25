from autogluon.tabular import TabularPredictor
from mlcomp.data.load import load_classification_test
from mlcomp.data.preprocess import drop_ft2, remove_outliers
import pandas as pd
from pprint import pprint

predictor = TabularPredictor.load("AutogluonModels/ag-20240116_230702")


print(predictor.leaderboard())