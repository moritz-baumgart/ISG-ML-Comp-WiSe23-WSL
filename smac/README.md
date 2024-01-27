### SMAC

This folder contains everything I did with SMAC and additionally some things I based of of configurations obtained by SMAC. My currently best model for the classification task can be found in here as well (inside `stacking.ipynb`).

For results and more info please look into the individual notebooks.

**Especially look into `doc.md` for SMAC results and at the bottom of `stacking.ipynb` for stacking results**

A quick overview:
- `smac.ipynb`: In this file I ran SMAC to optimize hyperparamters for various algorithms.
- `default_cs.py`: Used by `smac.ipynb`, encodes configuration spaces used by SMAC.
- `doc.md`: Documents all configurations obtained by SMAC and how they performed.
- `stacking.ipynb`: Stacking of different models. Here I used configs obtained with `smac.ipynb`
---
- `bagging.ipynb`: An experimental bagging approach.
- `cat.ipynb`: Here I played around with CatBoost, when I first knew/tested it.
- `gen_pred.ipynb`: Used to retrain a model on all the data and do a prediction for kaggle.
- `shap.ipynb`: Here I tried out shap values to see if I can improve my score using them.