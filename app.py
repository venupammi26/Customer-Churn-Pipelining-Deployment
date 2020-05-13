

import pandas as pd
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import base_est,csv

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      #f.save(secure_filename(f.filename))
      data =pd.read_csv(f)
      data['Churn_Prediction']=model.predict(data)
      output =data[['Customer ID','Churn_Prediction']]
      return render_template('index.html',  tables=[output.to_html(classes='data')], titles=output.columns.values)
      
     # return render_template('index.html', prediction_text='data shape $ {}'.format(output.loc[0]))
#if __name__ == "__main__":
 #   app.run(host='localhost', debug=True, port=5000)
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) 