# The line above create a file called "cs_data_ingestor.py" in the  runtime  working
# directory and write the the reste of the cell in this file.

import json
import os
import sys
import getopt
import re
import shutil
import time
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
## For plotting
import matplotlib.pyplot as plt## For outliers detection

data_dir = ""
DATA_DIR1 = os.path.join(".","data","cs-train")
DATA_DIR2 = os.path.join(".","data")
DATA_DIRP = os.path.join(".","data","cs-production")
IMAGE_DIR = os.path.join(".","images")


plt.style.use('seaborn')

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

#code adopted from solution guidance 
def fetch_data(data_dir):
    """
    load all json formatted files into a dataframe
    """

    ## input testing
    if not os.path.isdir(data_dir):
        raise Exception("specified data dir does not exist")
    if not len(os.listdir(data_dir)) > 0:
        raise Exception("specified data dir does not contain any files")

    file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search("\.json",f)]
    correct_columns = ['country', 'customer_id', 'day', 'invoice', 'month',
                       'price', 'stream_id', 'times_viewed', 'year']

    ## read data into a temp structure
    all_months = {}
    for file_name in file_list:
        df = pd.read_json(file_name)
        all_months[os.path.split(file_name)[-1]] = df

    ## ensure the data are formatted with correct columns
    for f,df in all_months.items():
        cols = set(df.columns.tolist())
        if 'StreamID' in cols:
             df.rename(columns={'StreamID':'stream_id'},inplace=True)
        if 'TimesViewed' in cols:
            df.rename(columns={'TimesViewed':'times_viewed'},inplace=True)
        if 'total_price' in cols:
            df.rename(columns={'total_price':'price'},inplace=True)

        cols = df.columns.tolist()
        if sorted(cols) != correct_columns:
            raise Exception("columns name could not be matched to correct cols")

    ## concat all of the data
    df = pd.concat(list(all_months.values()),sort=True)
    years,months,days = df['year'].values,df['month'].values,df['day'].values 
    dates = ["{}-{}-{}".format(years[i],str(months[i]).zfill(2),str(days[i]).zfill(2)) for i in range(df.shape[0])]
    df['invoice_date'] = np.array(dates,dtype='datetime64[D]')
    df['invoice'] = [re.sub("\D+","",i) for i in df['invoice'].values]
    df["revenue"] = df["times_viewed"]*df["price"]
    
    ## sort by date and reset the index
    df.sort_values(by='invoice_date',inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    #export file 
   
    
    return(df)


def filter_data(df):
    ## find the top ten countries (wrt revenue)
    
    print("\n Imported data  with the following attrbutes \n")
    df.info()
    columns_to_show = ["revenue"]
    df_agg= df.groupby(['country'])[columns_to_show].sum().round(3).sort_values(['revenue'],ascending=False)
    #df.sort_values([''])
    #top10 = df_agg.sort_values(['price'],ascending=False).groupby('country').head(10)
    print("\nTop 10 countries to use \n{}".format("-"*15))
    
    top10 = df_agg.head(n=10)
    print(top10.index.unique())
    print("Filtering data based on top 10 countries to train model")
    df_top10 = df[df.country.isin(top10.index.unique())]
    print(df_top10.head(n=10))
    return df_top10



def update_target(target_file,df_clean, overwrite=False):
    """
    update line by line in case data are large
    """

    if overwrite or not os.path.exists(target_file):
        df_clean.to_csv(target_file, index=False)   
    else:
        df_clean.to_csv(target_file, mode='a', header=False, index=False)


def create_plot(df):
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    table1 = pd.pivot_table(df, index='country', columns='year', values='price',aggfunc='mean').round(3)
    table1.plot(kind='bar', ax=ax1)
    ax1.set_ylabel("Average price");

    table2 = pd.pivot_table(df, index='country', columns='year', values="revenue",aggfunc='sum').round(3)
    table2.plot(kind='bar', ax=ax2)
    ax2.set_ylabel("Total revenue viewership");

    ## adjust the axis to accomadate the legend
    ax1.set_ylim((0,9.3))
    ax2.set_ylim((0,1.3))
    image_path = os.path.join(IMAGE_DIR,"revenue.png")
    plt.savefig(image_path,bbox_inches='tight',pad_inches = 0,dpi=200)
    print("{} created.".format(image_path))

        


if __name__ == "__main__":
    
    

    ## collect args
    arg_string = "%s -c update "%sys.argv[0]
    try:
        optlist, args = getopt.getopt(sys.argv[1:],'c:')
    except getopt.GetoptError:
        print(getopt.GetoptError)
        raise Exception(arg_string)

    ## handle args
    #streams_file = None
    
    #db_file = None
    mode_exec = None
    for o, a in optlist:
        if o == '-c':
            mode_exec = a
            
    if mode_exec == "train":
        data_dir  = DATA_DIR1
    else:
        data_dir = DATA_DIRP
    
    
    df_raw =fetch_data(data_dir)
    print("\n Data information after import\n{}".format("-"*15))
    print(df_raw.info())
    print(df_raw.describe())
    
    print("\n Number of of days \n {}",df_raw["invoice_date"].max() - df_raw["invoice_date"].min())
    print("\ndf_raw before cleaning \n{}".format("-"*15))
    print(df_raw.isnull().sum(axis = 0))
    
    print("\n Data after cleaning")
    columns = ['country', 'day','month', 'price',  'times_viewed','year','invoice_date','revenue']
    df_analysis= df_raw[columns]
    print("\n df_analysis \n")
    create_plot(df_analysis)
    print ("\nFinal data for analysis\n{}".format("-"*15))
    print (df_analysis.info())
    
    print("Data with 10 countries for analysis \n{}",df_analysis.head(n=10))
    
    #update_target(os.path.join(data_dir2,'customer-data.csv'),df_analysis,overwrite=True)
    #print('\nCreated  file customer-data.csv')
    
         
    
    #filter training data based on top 10 countries
    
    df10 = filter_data(df_analysis)
    #save the data for  later use in  model training and testing 
    if mode_exec =="train":
        update_target(os.path.join(DATA_DIR2,'top10countries-data.csv'),df10,overwrite=True)
        print('\n Created  top10country-data.csv  for training in ./data folder \n' )
    else:
        update_target(os.path.join(DATA_DIR2,'updtop10countries-data.csv'),df10,overwrite=True)
        print('\n  Created  uptop10countries-data.csv to simulate production data in ./data folder \n')

    
    #convert invoice_date to string first before sending as json 
    
    #data.txt and text.txt are used for  testing apis
    #df10["invoice_date"] = df10["invoice_date"].dt.strftime("%d/%m/%Y")
    df10["invoice_date"] = df10["invoice_date"].astype(str)
    df10.to_json (os.path.join(DATA_DIR2,r'data.txt'), orient='split')
    df_test = df10.head(n=100)
    update_target(os.path.join(DATA_DIR2,'test.txt'),df_test,overwrite=True)
  

    
#pd.pivot_table(df, index= ['country','year'], values=columns_to_show,aggfunc='mean').round(3)
