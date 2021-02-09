# prj_capstone
## AIWorkflow Production
 
 Please note, I chose ARIMA forecasting model for this project , the convention of train, splits is not the same nor the forecasting. But I will take you through it.
 
 
 1. Please review  CS_PKJM.READMEFIRST RISE generated pdf 
 2. Should you have more time, you can spin up the python notebook and give a go on the scripts
 3. Runtime folder has  the docker version of it, you just need to Dockerfile , and access the app on http://0.0.0.0:8082
 4. Sample Quick commands for app.py script
     4.1  model train : http://0.0.0.0:8082/train
     4.2  predict: http://0.0.0.0:8082/predict 
     4.3  model update : http://0.0.0.0:8082/update
 
 This is my first forecasting project , I enjoyed  it, then I read alot of literature, accessed alot medium articles in addition to course content provided
 
 Hope you Enjoy!


## Evaluation Criteria
1. Are there unit tests for the API?
   runtime/unittests/ApiTests.py

2. Are there unit tests for the model?
   runtime/unittests/ModelTests.py
   
3. Are there unit tests for the logging?
   runtime/unittests/LoggerTests.py
   
4. Can all of the unit tests be run with a single script and do all of the unit tests pass?
   runtime/run-tests.py
   
5. Is there a mechanism to monitor performance?
   Please check /logs/predict, /logs/train, /logs/train_test  logs  and logger.py scripts 
   
6. Was there an attempt to isolate the read/write unit tests From production models and logs?
   log files have column denoted test if it was a test. Please check /runtime app.py script , module model_train
   
7. Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined?
   /runtime/app.py
   
8. Does the data ingestion exists as a function or script to facilitate automation?

   /runtime/cs_data_ingestor.py

9. Where multiple models compared?
   /runtime/CS_PKJM.ipyb section for model 1 and 2 comparison

10. Did the EDA investigation use visualizations?
   CS_PKJM.ipyb 
   
11. Is everything containerized within a working Docker image?
   /runtime/Dockerfile

12 .Did they use a visualization to compare their model to the baseline model?
Please check  CS_PKJM.ipyb  under model drift where tree based data structures were used .


