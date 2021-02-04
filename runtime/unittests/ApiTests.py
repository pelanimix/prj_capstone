

#Create API test script

import sys
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np

port = 8082

try:
    requests.post('http://0.0.0.0:{}/predict'.format(port))
    server_available = True
except:
    server_available = False
    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """

    @unittest.skipUnless(server_available, "local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """
        
        fileup= open("./data/test.txt","rb")
        r = requests.post("http://0.0.0.0:{}/train".format(port),files ={"file":fileup})
        fileup.close()
        #train_complete = re.sub("\W+", "", r.text)
        train_complete = re.sub("\W+", "", r.text)
        self.assertEqual(train_complete, 'country0allmape003rmse002')
        '''
         response  = r.text 
        eval_test = dict({"country": {"0": "all"}, "mape": {"0": "0.3"}, "rmse": {"0": "0.2"}})
        self.assertEqual(response, eval_test)
        '''
    
    @unittest.skipUnless(server_available, "local server is not running")
    def test_02_predict_empty(self):
        """
        ensure appropriate failure types
        """
    
        ## provide no data at all 
        r = requests.post('http://0.0.0.0:{}/predict'.format(port))
        self.assertEqual(literal_eval(r.text), [])
        
        ## provide improperly formatted data
        #r = requests.post('http://0.0.0.0:{}/predict'.format(port), json={"key":"value"})     
        #self.assertEqual(literal_eval(r.text),[])

    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_03_predict(self):
        """
        test the predict functionality
        """

        query_data = {"country":"United Kingdom", "date" :"01/08/2019"}

        query_type = 'dict'
        request_json = {'query':query_data, 'type':query_type}

        r = requests.post('http://0.0.0.0:{}/predict'.format(port), json=request_json)
        response = literal_eval(r.text)

        self.assertEqual(response,{'Predrevenue': 23607.119, 'status': 200})
        
       

    @unittest.skipUnless(server_available, "local server is not running")
    def test_04_logs(self):
        """
        test the log functionality
        """

        file_name = 'train-test.log'
        request_json = {'file':'train-test.log'}
        r = requests.get('http://0.0.0.0:{}/logs/{}'.format(port, file_name))

        with open(file_name, 'wb') as f:
            f.write(r.content)
        
        self.assertTrue(os.path.exists(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)

        
### Run the tests
if __name__ == '__main__':
    unittest.main()
