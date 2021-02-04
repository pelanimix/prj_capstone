print(__doc__)

# Author: Pelani Malange <pmalange@za.ibm.com>

import os
import re
import csv
import uuid
from datetime import date, datetime
from datetime import timedelta
import time
from collections import Counter,defaultdict
import pandas as pd
import numpy as np
import joblib
import json
import socket
import pmdarima as pm
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preprocessing as ppc
from pmdarima import arima
from flask import Flask, jsonify, request
from flask_json import FlaskJSON, JsonError,json_response,as_json

MODEL_VERSION = "0.1"



## specify the directory you saved the data and images in
DATA_DIR = os.path.join(".","data")
IMAGE_DIR = os.path.join(".","images")
MODEL_DIR  = os.path.join(".","models")
PRED_DIR = os.path.join(".","preds")


app = Flask(__name__)
json = FlaskJSON(app)

#@app.route("/")
def statusping():
    html = "<h3>Flask predict says Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

def update_target(target_file,df, overwrite=False):
    """
    update line by line in case data are large
    """

    if overwrite or not os.path.exists(target_file):
        df.to_csv(target_file, index=True)   
    else:
        df.to_csv(target_file, mode='a', header=False, index=True)

    
def _update_predict_log(y_pred, y_proba, query, runtime):
    """
    update predict log file
    """
    
    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    logfile = "sales-arima-{}-{}.log".format(today.year, today.month)

    ## write the data to a csv file    
    header = ['unique_id', 'timestamp', 'y_pred', 'y_proba', 'period', 'model_version', 'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), y_pred, y_proba, query, MODEL_VERSION, runtime])
        writer.writerow(to_write)
        
        
@app.route('/home',methods=['POST','GET'])
def showId():
    data = request.get_json(force=True)
    try :
        value = int(data['value'])
    except (KeyError,TypeError,ValueError):
        raise JsonError(description='Invalid value')
    return json_response(value=value + 1)

@app.route('/get_value')
@as_json
def get_value():
    return dict(value=12)
    
    

def model_update():
    # We can also call `update` directly on the pipeline object, which will update
    # the intermittent transformers, where necessary:
    data=fetch_data()
    newly_observed, still_test = test[:15], test[15:]
    model.update(newly_observed, maxiter=10)

    # Calling predict will now predict from newly observed values
    new_preds = model.predict(still_test.shape[0])
    print(new_preds)
  
@app.route('/predict',methods=['POST','GET'])
def model_predict():
    
    num_periods = 30
    ## input checking

    data = request.get_json(force=True)
    #get the number of months to forecast
     #select the country model
        
    try:#value = int(data['value'])
        country  = data['country']
        idx_start_date = data['date']
    except (KeyError,TypeError,ValueError):
        raise JsonError(description='Invalid value')
    idx_start_date = datetime.strptime(idx_start_date, '%d/%m/%Y')
    end_period = idx_start_date + timedelta(num_periods)
    
    #select model based on country 
    str_country  = country.lower()
    
    
    saved_model = str_country+"-"+"sales-arima-0_1.joblib"
    #saved_model = "sales-arima-0_1.joblib"
    model = joblib.load(os.path.join(MODEL_DIR, saved_model))
    # We can compute predictions the same way we would on a normal ARIMA object:
    ## input checking
                          
    print("... predicting")
    
     ## input checking
    #if not request.json:
     #   print("ERROR: API (predict): did not receive request data")
      #  return jsonify([])
    
    ## start timer for runtime
    time_start = time.time()
    
    ## ensure the model is loaded
    #model = joblib.load(os.path.join(MODEL_DIR, saved_model))
    
    
    #preds, conf_int = pipe.predict(n_periods=periods, return_conf_int=True)
    preds, conf_int = model.predict(start=idx_start_date,end=end_period, return_conf_int=True)
    #index_of_fc = pd.date_range(ts.index[-1], periods = n_periods, freq='MS')
    index_of_fc = pd.date_range(idx_start_date, periods = len(preds), freq='D')
    # make series for plotting purpose
    fitted_series = pd.Series(preds, index=index_of_fc)
    df_series = fitted_series.to_frame()
    df_series.columns = ["proj_sales"]
    avgrevpred = df_series.proj_sales.mean().round(3)
    
    #print predicted values 
    
     ## make prediction and gather data for log entry
    y_proba = None
    if 'predict_proba' in dir(model) and model.probability == True:
        y_proba = model.predict_proba(n_periods=1)
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)
    ## update the log file
    _update_predict_log(preds, y_proba,country, runtime)
    
    #print results to outfile
    update_target(os.path.join(PRED_DIR,str_country+'-preds.csv'),df_series,overwrite=True)
    #_update_target(df_series)
    
    

    #return jsonify(preds,conf_int)
    #return jsonify(preds.tolist())

    #return jsonify(avgrevpred)
    return json_response(Predrevenue=avgrevpred)


if __name__ == "__main__":
    
    app.run(debug=True, host='0.0.0.0', port=8081)
    #app.run(host='0.0.0.0', port=8080,debug=True)
    
    # select the model based on country 
    
