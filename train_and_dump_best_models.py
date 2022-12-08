#ML regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pickle


def train_and_dump(scaler, data):
    y_depth = np.array(data['Depth'])
    y_width = np.array(data['Width'])
    X = np.array(data.drop(['Depth', 'Width'], axis=1))
    X_scaled = scaler.fit_transform(X)

    gbr_regressor = GradientBoostingRegressor()
    dec_tree_regressor = DecisionTreeRegressor()

    gbr_regressor = gbr_regressor.fit(X_scaled, y_depth)
    dec_tree_regressor = dec_tree_regressor.fit(X_scaled, y_width)

    pickle.dump(gbr_regressor, open('models/gbr_regressor.pkl', 'wb'))
    pickle.dump(dec_tree_regressor, open('models/dec_tree_regressor.pkl', 'wb'))