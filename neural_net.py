# from scikeras.wrappers import KerasRegressor
import collections

from keras.losses import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, RepeatedKFold
import matplotlib.pyplot as plt
import tensorflow as tf

# #нейронки
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.optimizers import Adam

# оборачиваем архитектуру в функцию
def get_model(output_layers,
              learning_rate,
              activation_1_layer,
              dropout_1_layer,
              neurons_2_layer,
              activation_2_layer,
              losses,
              optimizer_ = Adam):
    model = Sequential()
    model.add(Dense(4, input_shape=(4,), activation=activation_1_layer))
    model.add(Dropout(dropout_1_layer))
    model.add(Dense(neurons_2_layer, activation=activation_2_layer))
    model.add(Dense(output_layers, activation='linear'))
    # early stop

    # компилируем модель
    model.compile(loss=losses,
                  optimizer=optimizer_(learning_rate),
                  metrics=losses)

    return model

def get_evaluation(model, X, y):
  cv = RepeatedKFold(n_splits = 5,
                     n_repeats = 5,
                     random_state = 42)

  return cross_val_score(model, X, y,
                           scoring = 'neg_mean_absolute_percentage_error',
                           cv = cv)

def split_data_and_check_neural_network(column, data, scaler, epochs, batch_size):
    print(f"Evaluating error for: {column}, epochs: {epochs}, batch_size: {batch_size}" )
    out_layers = 1
    if (not isinstance(column, str)) and hasattr(column, "__len__"):
        out_layers = len(column)

    y = np.array(data[column]).tolist()
    X = np.array(data.drop(['Depth', 'Width'], axis=1))
    scaler.fit(X)  # pkl
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,
                                                        test_size= 0.2,
                                                        shuffle = True,
                                                        random_state=42)

    regressor = KerasRegressor(model=get_model,
                              output_layers = out_layers,
                              verbose=0,
                              activation_1_layer='relu',
                              dropout_1_layer=0.15,
                              neurons_2_layer=10,
                              activation_2_layer='relu',
                              learning_rate=0.01,
                              losses=mean_absolute_percentage_error)
    regressor.fit(X_train, y_train)

    # задаём параметры сетки
    param_grid = {'epochs': [epochs],
                  'batch_size': [batch_size],
                  'activation_1_layer': ['relu'],
                  'dropout_1_layer': [0.15],
                  'neurons_2_layer': [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
                  'activation_2_layer': ['relu'],
                  'learning_rate': [0.001],
                  'losses': [mean_absolute_percentage_error]}

    # создаем сетку
    nn_grid = GridSearchCV(estimator=regressor,
                           param_grid=param_grid,
                           scoring='neg_mean_absolute_percentage_error',
                           cv=3)

    # обучаем сетку
    grid_result = nn_grid.fit(X_train, y_train)
    # визуализация обучения нейронки
    plt.plot(grid_result.cv_results_['param_neurons_2_layer'], grid_result.cv_results_['mean_test_score'])
    plt.xlabel('Neurons on second layer')
    plt.ylabel('mean_test_score')
    y_pred = nn_grid.predict(X_test)
    print("Test expectations: " + str(y_test))
    print("Test results: " + str(y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    title = f'Epochs: {epochs}, B size: {batch_size}, Best train err: {round(grid_result.best_score_, 4)}, Test err: {str(round(float(tf.reduce_mean(mape).numpy()), 4))}, Neurons: {grid_result.best_estimator_.neurons_2_layer}'
    plt.title(title)
    print(title)
    # print(str(abs(grid_result.cv_results_['mean_test_score'])))
    # print(str(grid_result.cv_results_['param_neurons_2_layer']))
    result = {}
    result['epochs'] = epochs
    result['best_train_mape'] = round(grid_result.best_score_, 4)
    result['test_mape'] = mape
    result['best_train_neurons'] = grid_result.best_estimator_.neurons_2_layer
    plt.show()

    return result