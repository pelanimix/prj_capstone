
print(__doc__)
#Author : Pelani Malange <pmalange@za.ibm.com>

# this is a standalone model app , for the sake of the project , the module is part of pred app
import numpy as np 
import sys
import os
import re
import joblib
import csv
import uuid
from datetime import date, datetime
from datetime import timedelta
import time
import socket
from logger import  update_predict_log, update_train_log
from pylab import rcParams
import statsmodels.api as sm
import pandas as pd  
from statsmodels.tsa.seasonal import seasonal_decompose 
from flask import Flask
from matplotlib import pyplot as plt
from pmdarima import auto_arima
import pmdarima as pm
from pmdarima import model_selection
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from math import sqrt
from sklearn.metrics import mean_squared_error
from flask import Flask, jsonify, request, Response
from flask_json import FlaskJSON, JsonError,json_response,as_json

MODEL_VERSION = "0.1"
MODEL_VERSION_NOTE = "auto_arima"

## specify the directory you saved the data and images in
DATA_DIR = os.path.join(".","data")
IMAGE_DIR = os.path.join(".","images")
MODEL_DIR  = os.path.join(".","models")

#app = Flask(__name__)
#json = FlaskJSON(app)


#@app.route("/")
def statusping():
    html = "<h3>Model says hello  {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

def fetch_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "top10countries-data.csv"),index_col="invoice_date", parse_dates=True)
    #print("df: {} x {}".format(df.shape[0], df.shape[1]))

    ## check the first few rows
    #print("\n  Check first 4 rows\n")
    #print(df.head(n=4) 
    return df
def filter_cntry_data(df,country):
    #if !df.country.index:
    #    df = df.set_index('country')
    
    df = df[df.index==country]
    ts = df.groupby("invoice_date")["revenue"].sum().rename("sales")
    
    y_ts = ts.resample('MS').mean()
    
    return y_ts

def model_load(test=False):
    """
    example funtion to load model
    """
    if test : 
        print( "... loading test version of model" )
        model = joblib.load(os.path.join(MODEL_DIR,"sales-arima-0_1.joblib"))
        return(model)

    if not os.path.exists(SAVED_MODEL):
        exc = "Model '{}' cannot be found did you train the full model?".format(SAVED_MODEL)
        raise Exception(exc)
    
    model = joblib.load(SAVED_MODEL)
    return(model)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#@app.route('/train')
def model_train (test=False):
    
## subset the data to enable faster unittests
      
    #create dataframe for temporary capture of traiing results 
    df_res = pd.DataFrame(columns=["country","rmse","mape"])
    
    df = fetch_data()
    ts = df.groupby("invoice_date")["revenue"].sum().rename("sales")
    y= ts.resample('MS').mean()
    
     ## start timer for runtime
    time_start = time.time()
    model = pm.auto_arima(y, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
    if test:
        saved_model ="sales-arima-{}.joblib".format(re.sub("\.", "_", str(MODEL_VERSION)))
        train, test = model_selection.train_test_split(y, test_size=0.1)
        result = model.fit(train)
        joblib.dump(model, os.path.join(MODEL_DIR, saved_model))
        df_res.loc[0] = ["all","0.2", "0.3"] 
        m, s = divmod(time.time()-time_start, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d"%(h, m, s)
        update_train_log(y.shape[0],{'country':all,'rmse':"0.2",'mape':"0.3"},runtime,MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
        
    else:
        country = data.index.unique().tolist()
        c_listlen= len(country)
        for i in range(c_listlen):
            cntry = country[i]
            #format the country variable
            if cntry.isspace():
                str_country= cntry
                str_country = re.sub(r"\s+",'-',str_country)
                str_country =str_country.lower()
            else:
                str_country = cntry.lower()

            #set country model
            saved_model = str_country+"-"+"sales-arima-{}.joblib".format(re.sub("\.", "_", str(MODEL_VERSION)))
            #filter data to train based on country 
            y =filter_cntry_data(df,cntry)

            # Split data into train / test sets 
            train, test = model_selection.train_test_split(y, test_size=0.1)

            #smodel.summary()
            result = model.fit(train)



            #print Autom arima diagnostics
            #results.plot_diagnostics(figsize=(16, 8))
            #plt.show()


            joblib.dump(model, os.path.join(MODEL_DIR, saved_model))
            #result.plot_diagnostics(figsize=(15,12))


            #print( result.summary().tables[1])



            #print("\n Proceed with Auto Arima due to better AIC value\n")
            #forecast 


            forecast  =  model.predict(n_periods=len(test))
            forecast  = pd.DataFrame(forecast,index=test.index,columns=['predictions'])

            #hide the plots as this will be caled via scripts
            #plot
            '''
            plt.plot(train,label='Train')
            plt.plot(test, label='Valid')
            plt.plot(forecast, label ='Prediction')
            plt.legend()
            plt.show()


            '''

            rms = round(sqrt(mean_squared_error(test,forecast)),2)
            #print("Arima rms \n:{}",model_train ())

            mape_result = round(mean_absolute_percentage_error(test,forecast))

            df_res.loc[i] = [cntry,rms, mape_result] 
            m, s = divmod(time.time()-time_start, 60)
            h, m = divmod(m, 60)
            
            update_train_log(y.shape[0],{'country':cntry,'rmse':rms,'mape':mape_result},runtime,MODEL_VERSION, MODEL_VERSION_NOTE, test=False)
            runtime = "%03d:%02d:%02d"%(h, m, s)

    return dict(df_res.to_dict())

#if __name__ == "__main__":
    
 #   app.run(debug=True, host='0.0.0.0', port=8080)
    #app.run(host='0.0.0.0', port=8080,debug=True)
    
    # ETS Decomposition 
    #df = fetch_data()
    #convert to time series data 
    #df["invoice_date"] = pd.to_datetime(df['invoice_date'], format='%d.%m.%Y')## create time series
    #ts = df.groupby("invoice_date")["revenue"].sum().rename("sales")
    
    #y_ts = ts.resample('MS').mean()
    #print("y[2019] is \n{}",y_ts["2019"])
    
    
    
    
    
    
    # Converting the index as date
    #print(ts.describe())
    #ts.tail()
    #print("Show y averages per month\n{}")
    #y.plot(figsize=(15, 6))
    #plt.show()
    
    #print("Show TS averaged monthly normal{}")
    #y_ts.plot()
    #plt.show()
    #result = seasonal_decompose(y_ts, period=1) 
    
    #print("Show seasonal decomposition across 1 month average  due to TS monthly average")
    #rcParams['figure.figsize'] = 18, 8
    #decomposition = seasonal_decompose(y_ts, model='additive',period=1)
    #fig = decomposition.plot()
    #plt.show()
    
    #data range
    #print ( "\n Data range  start : {} and end: {}",ts['invoice_date'].min(), ts['invoice_date'].max())
  
    # ETS plot  
    #result.plot() 
   # pre_train(train)
    
    #model file 
    #saved_model = "sales-arima-{}.joblib".format(re.sub("\.", "_", str(MODEL_VERSION)))
    #results = model_train(train,test,saved_model)
    #validate(results,y_ts)
   
    
    
    #rmse
