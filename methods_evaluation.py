#общие
import pandas as pd
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler

from algorithms import split_data_and_check_algos
from neural_net import split_data_and_check_neural_network
from train_and_dump_best_models import train_and_dump

sns.set_style('darkgrid')

path = 'data/ebw_data.csv'

data = pd.read_csv(path)
#grouped_data_mean = data.groupby(['IW','IF','VW','FP']).mean().reset_index()
scaler = StandardScaler()
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
train_and_dump(scaler, data)

# Пробежимся по разному кол-ву эпох с разными кол-вами нейронов 2го слоя,
# чтобы определить оптимальную конфигурацию
results = []
for epochs in range(10, 110, 10):
    results.append(split_data_and_check_neural_network(['Width','Depth'], data, scaler, epochs, 5))
print('Width + Depth prediction results: ')
print(str(results))

results = []
for epochs in range(10, 110, 10):
    results.append(split_data_and_check_neural_network('Width', data, scaler, epochs, 5))
print('Width prediction results: ')
print(str(results))

results = []
for epochs in range(10, 110, 10):
    results.append(split_data_and_check_neural_network('Depth', data, scaler, epochs, 5))
print('Depth prediction results: ')
print(str(results))

split_data_and_check_algos('Width', data, scaler)
split_data_and_check_algos('Depth', data, scaler)