### AutoGluon

Unfortunately I cannot commit the large AutoGluon Model files to git, but a documentation on how each model was configured and how it performs is available in `doc.md`.

An overview of all the files in this folder:

- `doc.md`: Hold results for the different models.
---
- `auto-clf.py`: Script for starting AutoGluon with the classification task data.
- `auto-reg.ipynb`: Script for starting AutoGluon with the regression task data and some preprocessing.
- `gen_pred_clf.py`: Generates a prediction for kaggle on the classification test data using a saved model.
- `gen_pred_reg.ipynb`: Generates a prediction for kaggle on the regression test data using a saved model.
- `pp.py`: Prints the leaderboard of a saved model.
