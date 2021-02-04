print(__doc__)

# Author: Pelani Malange <pmalange@za.ibm.com>
# App will service model train and prediction

import os
import os.path
from pathlib import Path 
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
from math import sqrt
from sklearn.metrics import mean_squared_error
import json
import socket
from werkzeug.utils import secure_filename
import pmdarima as pm
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preprocessing as ppc
from pmdarima import arima
from flask import render_template, send_from_directory
from flask import Flask, jsonify, request,redirect
from flask_json import FlaskJSON, JsonError,json_response,as_json
from logger import update_train_log, update_predict_log

MODEL_VERSION = "0.1"
MODEL_VERSION_NOTE = "auto_arima"
CNTRY_MODELS =[]



if not os.path.exists(os.path.join(".","data")):
    os.mkdir("data")

## specify the directory you saved the data and images in
DATA_DIR = os.path.join(".","data")
IMAGE_DIR = os.path.join(".","images")
MODEL_DIR  = os.path.join(".","models")
PRED_DIR = os.path.join(".","preds")


app = Flask(__name__)
json_flask = FlaskJSON(app)

#@app.route("/")
def statusping():
    html = "<h3>Flask predict says Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    """
    API endpoint to get logs
    """

    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify([])

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])

    file_path = os.path.join(log_dir, filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify([])
    
    return send_from_directory(log_dir, filename, as_attachment=True)


def update_target(target_file,df, overwrite=False):
    """
    update line by line in case data are large
    """

    if overwrite or not os.path.exists(target_file):
        df.to_csv(target_file, index=True)   
    else:
        df.to_csv(target_file, mode='a', header=False, index=True)

        
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
    

#train model section


'''

def fetch_data(country,df ):
    #df = pd.read_csv(os.path.join(DATA_DIR, "top10countries-data.csv"),index_col="invoice_date", parse_dates=True)
    #print("df: {} x {}".format(df.shape[0], df.shape[1]))

    ## check the first few rows
    #print("\n  Check first 4 rows\n")
    #print(df.head(n=4))
   
    df = df[df.index==country]
    ts = df.groupby("invoice_date")["revenue"].sum().rename("sales")
    
    y_ts = ts.resample('MS').mean()
    
    
    return y_ts

'''



def fetch_data(query):
    df = pd.read_csv(os.path.join(DATA_DIR,query),index_col=0, parse_dates=['invoice_date'])
    #f = open (os.path.join(DATA_DIR,query),"r")
    #dt = json.loads(f.read())
    #df = pd.read_json(dt,orient='split',convert_dates=["invoice_date"])
    #df = pd.read_json(os.path.join(DATA_DIR,"data.txt"))
    print("df size : {} x {}".format(df.shape[0], df.shape[1]))
    df = df.round(2)

    
    return df 
def filter_cntry_data(df,country):
    #if !df.country.index:
    #    df = df.set_index('country')
    
    df = df[df.index==country]
    ts = df.groupby("invoice_date")["revenue"].sum().rename("sales")
    
    y_ts = ts.resample('MS').mean()
    
    return y_ts

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@app.route('/uploadfile',methods=['POST'])
def uploadfile():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename !='':
            f.save(os.path.join(DATA_DIR,f.filename))
        filenm=f.filename
    return filenm

@app.route('/train',methods=['POST'])
def model_train ():
    
    filenm=""
    if request.method == 'POST':
        f = request.files['file']
        if f.filename !='':
            f.save(os.path.join(DATA_DIR,f.filename))
        filenm=f.filename

    print("\nFile name is :\n{}",filenm)
    data = fetch_data(filenm)
    
    #create dataframe for temporary capture of traiing results 
    df_res = pd.DataFrame(columns=["country","rmse","mape"])
    ts = data.groupby("invoice_date")["revenue"].sum().rename("sales")
    y_all = ts.resample('MS').mean()

    if filenm=="test.txt":
        time_start = time.time()
        model = pm.auto_arima(y_all, start_p=1, start_q=1,
                                 max_p=3, max_q=3, m=12,
                                 start_P=0, seasonal=True,
                                 d=None, D=1, trace=True,
                                 error_action='ignore',  
                                 suppress_warnings=True, 
                                 stepwise=True)

        saved_model ="sales-arima-{}.joblib".format(re.sub("\.", "_", str(MODEL_VERSION)))
        #train, test = model_selection.train_test_split(y_all, test_size=0.1)
        result = model.fit(y_all)
        joblib.dump(model, os.path.join(MODEL_DIR, saved_model))
        df_res.loc[0] = ["all","0.2", "0.3"] 
        m, s = divmod(time.time()-time_start, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d"%(h, m, s)
        train_shape = str(y_all.shape[0])+" x  1"
        update_train_log(train_shape,{'country':all,'rmse':"0.2",'mape':"0.3"},runtime,MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
        
    else:
        
        ## input checking

        #get the number of months to forecast
         #select the country model

        try:#value = int(data['value'])
            country = data.index.unique().tolist()
           
        except (KeyError,TypeError,ValueError):
            raise JsonError(description='Invalid value')

        #enforce datetime astype on cloumn invoice_date
        #data["invoice_date"] = pd.to_datetime(data["invoice_date"])

         ## start timer for runtime
        time_start = time.time()
        c_listlen= len(country)



        # Seasonal - fit stepwise auto-ARIMA
            #with ARIMA, due to size of the data , weshall not use train split
        #Having checked the ARIMA fit model, all countries repor the same hyperparameters
        #Training will be done however per country
        model = pm.auto_arima(y_all, start_p=1, start_q=1,
                                 max_p=3, max_q=3, m=12,
                                 start_P=0, seasonal=True,
                                 d=None, D=1, trace=True,
                                 error_action='ignore',  
                                 suppress_warnings=True, 
                                 stepwise=True)

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
            y =filter_cntry_data(data,cntry)

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
            runtime = "%03d:%02d:%02d"%(h, m, s)
            train_shape = str(y_all.shape[0])+" x 1"
            update_train_log(train_shape,{'country':cntry,'rmse':rms,'mape':mape_result},runtime,MODEL_VERSION, MODEL_VERSION_NOTE, test=False)
    #return json_response(rmse = rms, mape=mape_result)
    return dict(df_res.to_dict())
   



# end of train model section

@app.route('/update',methods=['POST'])
def model_update():
    # We can also call `update` directly on the pipeline object, which will update
    # the intermittent transformers, where necessary:
   
    
    
    filenm=""
    if request.method == 'POST':
        f = request.files['file']
        if f.filename !='':
            f.save(os.path.join(DATA_DIR,f.filename))
        filenm=f.filename

    print("\nFile name is :\n{}",filenm)
    data = fetch_data(filenm)
    
    #create dataframe for temporary capture of traiing results 
    df_res = pd.DataFrame(columns=["country","rmse","mape"])


    ## input checking

    #get the number of months to forecast
     #select the country model
        
    try:#value = int(data['value'])
        country = data.index.unique().tolist()
    except (KeyError,TypeError,ValueError):
        raise JsonError(description='Invalid value')
            
     ## start timer for runtime
    time_start = time.time()
    c_listlen= len(country)
    ts = data.groupby("invoice_date")["revenue"].sum().rename("sales")
    y_all = ts.resample('MS').mean()
    idx_start_date = y_all.index.min()
    print ("start date for model updates: " + str(idx_start_date))
        
    end_period = y_all.index.max()
        
    print("\n end period" +str( end_period))
   

    
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
        
        file_path = os.path.join(MODEL_DIR,saved_model)
        if not os.path.exists(file_path):
            continue #skip to next country 
    
            
        #filter data to train based on country 
        y =filter_cntry_data(data,cntry)
        
         # Split data into train / test sets 
        #train, test = model_selection.train_test_split(y, test_size=0.1)
   
        #smodel.summary()
        
        model = joblib.load(os.path.join(MODEL_DIR, saved_model))
        result = model.update(y)
       

        # Now plot the results and the forecast for the test set
        #preds, conf_int = model.predict(n_periods=len(y),return_conf_int=True)
        
    
        forecast, in_sample_confint = model.predict_in_sample(X=None, return_conf_int=True)
        #forecast,conf_int  =  model.predict(start=idx_start_date,end=end_period,return_conf_int=True)
        forecast = forecast[-len(y)]
        
        forecast  = pd.DataFrame(forecast,index=y.index,columns=['predictions'])

        #hide the plots as this will be caled via scripts
        #plot
        '''
        plt.plot(train,label='Train')
        plt.plot(test, label='Valid')
        plt.plot(forecast, label ='Prediction')
        plt.legend()
        plt.show()


        '''

        rms = round(sqrt(mean_squared_error(y,forecast)),2)
        #print("Arima rms \n:{}",model_train ())

        mape_result = round(mean_absolute_percentage_error(y,forecast))

        df_res.loc[i] = [cntry,rms, mape_result] 
        m, s = divmod(time.time()-time_start, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d"%(h, m, s)
        train_shape = str(y_all.shape[0])+" x 1"
        update_train_log(train_shape,{'country':cntry,'rmse':rms,'mape':mape_result},runtime,MODEL_VERSION, MODEL_VERSION_NOTE, test=False)


        joblib.dump(model, os.path.join(MODEL_DIR, saved_model))
     
          ## train logger
       
        
    #return json_response(rmse = rms, mape=mape_result)
    return dict(df_res.to_dict())

  



@app.route('/predict',methods=['POST','GET'])
def model_predict():
    
   ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])
    
    if request.json['type'] == 'dict':
        pass
    else:
        print("ERROR API (predict): only dict data types have been implemented")
        return jsonify([])
    
    ## extract the query
    data = request.json['query']
        
    if request.json['type'] == 'dict':
        pass
    else:
        print("ERROR API (predict): only dict data types have been implemented")
        return jsonify([])

    
    num_periods = 30
    ## input checking

    #0302211000 old way of getting query data 
    #data = request.get_json(force=True)
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
    model = joblib.load(os.path.join(MODEL_DIR, saved_model))
    # We can compute predictions the same way we would on a normal ARIMA object:
    ## input checking
                          
    print("... predicting")
    
    
    ## start timer for runtime
    time_start = time.time()

    
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
    #_update_predict_log(preds, y_proba,country, runtime)
    
    #update_predict_log("[0]", "[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]","00:00:01", MODEL_VERSION, test=True)
    update_predict_log(preds, y_proba,data, runtime, MODEL_VERSION, test=False)
    
    #print results to outfile
    update_target(os.path.join(PRED_DIR,str_country+'-preds.csv'),df_series,overwrite=True)
    #_update_target(df_series)
    
    

    #return jsonify(preds,conf_int)
    #return jsonify(preds.tolist())

    #return jsonify(avgrevpred)
    return json_response(Predrevenue=avgrevpred)


if __name__ == "__main__":
    
    app.run(debug=True, host='0.0.0.0', port=8082)
    #app.run(host='0.0.0.0', port=8080,debug=True)
    
    # select the model based on country 
    
