# Importing all required modules 
import bz2 as bz2
import datetime as dt
import joblib
import webbrowser as wb
import streamlit as st

import numpy as np
import pandas as pd
import streamlit as sl
import streamlit.components.v1 as com

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://www.uel.edu.vn" target="_blank">
  <img class="image-25"src="https://www.uel.edu.vn/Resources/Images/SubDomain/HomePage/Style/logo_uel.png" width="350"
  </a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

# Function to unpickle 
# def decompress_pickle(file):
#   data = bz2.BZ2File(file, 'rb')
#   data = pickle.load(data)
#   return data

# Function use to open link which will download the demo excel file
def open_link(str):
  wb.open(str)

# Unpacking Scaler pkl file
S_file = open('model.pkl','rb')
scaler = joblib.load(S_file)

# creating 4 Dicts which is used to convert to str to int befor giveing input to ML module
# S_DICT = {'Male':1,'Female':2}
# M_DICT = {'Married':1,'Single':2,'Others':3}
# E_DICT ={'Graduate school':1,'University':2,'High school':3,'Others':4}
# PAY_DICT= {'Zero Bill':0,'Paid duly':-1,'1 Month Delay':1,'2 Months Delay':2,'3 Months Delay':3,'4 Months Delay':4,'5 Months Delay':5,'6 Months Delay':6,'7 Months Delay':7,'8 Months Delay':8,'9 Months & Above Delay':9}

# while loop for Dynamic Months
n=0
Dynamic_months={}
current_month=dt.datetime.now()
while n < 6:
    month=current_month.strftime('%B')
    Dynamic_months['m{0}'.format(n)]=month
    current_month=current_month-dt.timedelta(days=27)
    n=n+1

# Function which handels multi transactions
def multi_cust(file):
  if file:
    # Reading as Pandas dataframe
    df = pd.read_excel(file)
    # Preprocessing
    cust_ID_DF=pd.DataFrame()
    df.drop(0,axis=0,inplace=True)
    # cust_ID_DF['customer_id/Name'] = df['Customer_ID/Name']
    # df.drop('Customer_ID/Name',axis=1,inplace=True)
    # df['AVG_BILL_AMT'] = ((df['1st Month.1']+df['2nd Month.1']+df['3rd Month.1']+df['4th Month.1']+df['5th Month.1']+df['6th Month.1'])/6)
#     df['SEX'] = df['SEX'].apply(lambda x: S_DICT[x])
#     df['EDUCATION'] = df['EDUCATION'].apply(lambda x: E_DICT[x])
#     df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: M_DICT[x])
#     df['1st Month'] = df['1st Month'].apply(lambda x: PAY_DICT[x])
#     df['2nd Month'] = df['2nd Month'].apply(lambda x: PAY_DICT[x])
#     df['3rd Month'] = df['3rd Month'].apply(lambda x: PAY_DICT[x])
#     df['4th Month'] = df['4th Month'].apply(lambda x: PAY_DICT[x])
#     df['5th Month'] = df['5th Month'].apply(lambda x: PAY_DICT[x])
#     df['6th Month'] = df['6th Month'].apply(lambda x: PAY_DICT[x])
    # Scaling the model
    # df_scaled = scaler.transform(df)
    # Pridecting
    # multi_pred = model.predict(df_scaled)
    # cust_ID_DF['default']=multi_pred
    # cust_ID_DF['Status for next month']=cust_ID_DF['default'].apply(lambda x : 'Repay' if x == 0 else 'Default')
    # cust_ID_DF.drop('default',axis=1,inplace=True)
    # Saving excel with only Customer name/ID with prediction
    data_frame= cust_ID_DF.to_csv()
    # Showing on the platform
    sl.table(cust_ID_DF)
    # Download button for the file
    sl.download_button(label='Download tabel',data=data_frame,mime='text/csv',file_name='Bill Payment Status for Next month')

 # Function to print out put which also converts numeric output from ML module to understandable STR 
def pred_out(num):
    if num == 1:
      sl.warning('This transaction IS LIKELY TO BE FRAUD..!')
    else:
      sl.success('This transaction IS NOT LIKELY TO BE A FRAUD..!')

# Titel 
sl.title('Fraud Transaction Prediction')

# Selection 
rad_b=sl.radio('Please select that you want give Single or Multiple transaction data',options=['Single','Multiple'])

features = ['value','num_date_review','dob','unknown_var_1','unknown_var_2','unknown_var_3','unknown_var_7','unknown_var_8','unknown_var_9','unknown_var_10','unknown_var_12','unknown_var_14','unknown_var_15','social_friend_count', 'unknown_var_17']

# Body of the page using FORM
def main():
  if rad_b == 'Single':
    form = sl.form('Customer Details')
    dob = form.number_input('dob', value=19810807)

    col1,col2 = form.columns(2)
    social_friend_count = col1.number_input('social_friend_count',value=653)
    unknown_var_17 = col2.number_input('unknown_var_17', value=1436)
    

    # form.subheader('Bill Amount in the respective months in $')
    col14,col15,col16=form.columns(3)
    unknown_var_15=col14.number_input('unknown_var_15', value=0.27)
    unknown_var_14=col15.number_input('unknown_var_14', value=0.195)
    unknown_var_12=col16.number_input('unknown_var_12', value=0.12)
    col17,col18,col19=form.columns(3)
    unknown_var_10=col17.number_input('unknown_var_10', value=13)
    unknown_var_9=col18.number_input('unknown_var_9', value=30)
    unknown_var_8=col19.number_input('unknown_var_8', value=8.414)
    
    # form.subheader('Amount paid for previous bill in $')
    col7,col8,col9=form.columns(3)
    unknown_var_7=col7.number_input('unknown_var_7', value=2)
    unknown_var_1=col8.number_input('unknown_var_1', value=4)
    unknown_var_2=col9.number_input('unknown_var_2', value=5)
    col11,col12,col13=form.columns(3)
    unknown_var_3=col11.number_input('unknown_var_3', value=11)
    num_date_review=col12.number_input('num_date_review', value=9)
    value=col13.number_input('value', value=3367400)
    
    # Creating new feature Average Bill Amount 
    features=[value,num_date_review,dob,unknown_var_1,unknown_var_2,unknown_var_3,unknown_var_7,unknown_var_8,unknown_var_9,unknown_var_10,unknown_var_12,unknown_var_14,unknown_var_15,social_friend_count]
    pred = scaler.predict(np.array(features,ndmin=2))

    P_satus=form.form_submit_button("Predict")
    # If predict button clicked it will predict
    if P_satus:
      pred_out(pred)

  else:
    # Multi transaction 
    sl.subheader('Please Download the Demo excel file')
    sl.text('Note:- enter the details of custome, save & Upload, Dont change to format.!')
    # HTML code for downloading demo file 
    com.html(f"""<button onclick="window.location.href='https://drive.google.com/uc?export=download&id=10aYBUF50jjAWvi-ukZZE2Q6_8pbLoUon';">
                      Download Demo File</button>""",height=30)
    multi_cust(sl.file_uploader('Please Upload Excel file'))


if __name__ == '__main__':
  main()
  
    
