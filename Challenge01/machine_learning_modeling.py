# pip install xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# dataset load and split
# dataloader
columns = pd.read_csv("./2022_train.csv").columns
dataset1 = pd.read_csv("./2022_train.csv").values
# dataset2 = pd.read_csv("./2023_train.csv").values[:-24]
# total_dataset = np.concatenate((dataset1, dataset2))

for i in range(12, 63):
    # model assign
    xgb_reg = xgb.XGBRegressor()

    # hyperparameter setting
    param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    # random search
    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_dist,
        n_iter=200,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=1,
    )

    # dataset
    X = dataset1[:, 2:12]
    y = dataset1[:, i]

    # data split
    train_input = X[:-300]
    train_target = y[:-300]

    test_input = X[-300:]
    test_target = y[-300:]

    # random forest model train
    random_search.fit(train_input, train_target)
    best_model = random_search.best_estimator_

    # scoring
    y_pred = best_model.predict(test_input)
    print(f"[{columns[i]}]")
    print(f"Score: {mean_squared_error(test_target, y_pred)}")
    # print(f"Target: {np.array(test_target[:20])})")
    # print(f"Predict: {y_pred[:20]}")

    plt.figure(figsize=(10, 5))
    plt.title(f"MSE: {mean_squared_error(test_target, y_pred)}")
    plt.plot(np.array(test_target), label="Target", color="red")
    plt.plot(y_pred, label="Predict", color="green")
    plt.legend()
    plt.savefig(f"./model_result/machinelearning/xgboost_model_{columns[i]}.pdf")
    plt.close()

    # model save
    with open(f"./model_result/machinelearning/xgboost_model_{columns[i]}.pickle", "wb") as f:
        pickle.dump(best_model, f)
