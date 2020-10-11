import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='Template')
reg_model = pickle.load(open('Linear_Regression_Model.pkl', 'rb'))
cls_model = pickle.load(open('NaiveBayes_Classifier_Model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    r = reg_model.predict(final_features)
    c = cls_model.predict(final_features)

    output = round(r[0] * c[0], 2)

    # output = round(c[0], 2)

    """
    if output < 1:
        output=0
    """
    if output > 0:
        return render_template('index.html', prediction_text='The person is likely to donated $ {}'.format(output))

    else:
        return render_template('index.html', prediction_text='The person is a non-donor')


if __name__ == "__main__":
    app.run(debug=True)
