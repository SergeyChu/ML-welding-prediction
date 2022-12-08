import flask


#ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask import Flask, render_template

#model save
import pickle


app = Flask(__name__)
gbr_depth_regressor = pickle.load(open('models/gbr_regressor.pkl', 'rb'))
dec_tree_width_regressor = pickle.load(open('models/dec_tree_regressor.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

print('Loaded')


@app.route('/')
def home():
    if flask.request.method == 'GET':
        return render_template('main.html')


@app.route('/predict', methods = ['POST'])
def predict():
    IW = float(flask.request.form['IW'])
    IF = float(flask.request.form['IF'])
    VW = float(flask.request.form['VW'])
    FP = float(flask.request.form['FP'])
    data = [IW,IF,VW,FP]
    data_scaled = scaler.fit_transform([data])
    depth_pred = gbr_depth_regressor.predict(data_scaled)
    width_pred = dec_tree_width_regressor.predict(data_scaled)
    return render_template('main.html', depth=depth_pred, width=width_pred)


if __name__ == '__main__':
    app.run()