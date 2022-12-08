#ML regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet, BayesianRidge
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import numpy as np


#словарь с моделями
def get_models_dict():
  models = dict()
  models['AdaBoostRegressor'] = AdaBoostRegressor()
  models['DecisionTreeRegressor'] = DecisionTreeRegressor()
  models['RandomForestRegressor'] = RandomForestRegressor()
  models['KNeighborsRegressor'] = KNeighborsRegressor()
  models['SupportVectors'] = SVR()
  models['GradientBoostingRegressor'] = GradientBoostingRegressor()
  models['LinearRegression'] = LinearRegression()
  models['KernelRidge'] = KernelRidge()
  models['SGDRegressor'] = SGDRegressor()
  models['ElasticNet'] = ElasticNet()
  models['BayesianRidge'] = BayesianRidge()
  return models


#функция обучения и получение метрик
def get_evaluation(model, X, y):
  cv = RepeatedKFold(n_splits = 5,
                     n_repeats = 5,
                     random_state = 42)

  return cross_val_score(model, X, y,
                           scoring = 'neg_mean_absolute_percentage_error',
                           cv = cv)

def split_data_and_check_algos(column, data, scaler):
    print("Evaluating error for: " + column)
    y = np.array(data[column])
    X = np.array(data.drop(['Depth', 'Width'], axis=1))
    X_scaled = scaler.fit_transform(X)

    models = get_models_dict()
    result = {}

    for name, model in models.items():
        scores = get_evaluation(model, X_scaled, y)  # смотрим ошибки
        result[name] = abs(np.mean(scores))
        print(f'model : {name} имеет ошибку MAPE = {abs(np.mean(scores))}')

    sorted_res = sorted(result.items(), key=lambda x: x[1])
    for v in sorted_res:
        print(str(round(float(v[1]), 6)) + " " + v[0])