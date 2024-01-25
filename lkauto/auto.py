from lkauto.lkauto import get_best_prediction_model
from lkauto.utils.get_default_configuration_space import get_default_configuration_space
from mlcomp.data.load import load_regression_train
import pandas as pd
import joblib


df = load_regression_train()
get_best_prediction_model(df, log_level='DEBUG')
