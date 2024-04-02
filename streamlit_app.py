import datetime as dt
import webbrowser as wb
import streamlit as st
import numpy as np
import pandas as pd
import streamlit as sl
import streamlit.components.v1 as com

# Import các module cần thiết
import joblib

# Titel 
st.title('Stock Return Prediction')

# Selection 
rad_b = st.radio('Please select that you want give Single or Multiple transaction data', options=['Single', 'Multiple'])

# Function use to open link which will download the demo excel file
def open_link(str):
    wb.open(str)

# Unpacking Scaler pkl file
with open('model.pkl', 'rb') as S_file:
    scaler = joblib.load(S_file)

# Body of the page using FORM
def main():
    if rad_b == 'Single':
        form = st.form('Customer Details')
        dob = form.number_input('Return on stockholder’s equity', value=0.25)

        col1, col2 = form.columns(2)
        social_friend_count = col1.number_input('Margin before interest and tax', value=0.3)
        unknown_var_17 = col2.number_input('Return on total assets growth ratio', value=0.1)

        col14, col15, col16 = form.columns(3)
        unknown_var_15 = col14.number_input('Net assets growth ratio after tax', value=0.27)
        unknown_var_14 = col15.number_input('Cash flow ratio', value=0.173)
        unknown_var_12 = col16.number_input('Accounts receivable turnover ratio', value=5)

        col17, col18, col19 = form.columns(3)
        unknown_var_10 = col17.number_input('Quick ratio', value=1.41)
        unknown_var_9 = col18.number_input('Net assets per stock', value=15000)
        unknown_var_8 = col19.number_input('Fixed asset turnover ratio', value=0.76)

        unknown_var_7 = 2
        unknown_var_1 = 4
        unknown_var_2 = 5
        unknown_var_3 = 11
        num_date_review = 9

        value = form.number_input('Return on assets', value=0.0304)

        features = [value, num_date_review, dob, unknown_var_1, unknown_var_2, unknown_var_3,
                    unknown_var_7, unknown_var_8, unknown_var_9, unknown_var_10, unknown_var_12,
                    unknown_var_14, unknown_var_15, social_friend_count]

        pred = scaler.predict(np.array(features).reshape(1, -1))

        P_status = form.form_submit_button("Predict")
        if P_status:
            pred_out(pred)

    else:
        # Multi transaction 
        st.subheader('Please Download the Demo excel file')
        st.text('Note:- enter the details of custome, save & Upload, Dont change to format.!')
        # HTML code for downloading demo file 
        com.html(f"""<button onclick="window.location.href='https://drive.google.com/uc?export=download&id=10aYBUF50jjAWvi-ukZZE2Q6_8pbLoUon';">
                          Download Demo File</button>""", height=30)
        multi_cust(sl.file_uploader('Please Upload Excel file'))

def multi_cust(file):
    if file:
        # Reading as Pandas dataframe
        df = pd.read_excel(file)
        # Preprocessing
        cust_ID_DF = pd.DataFrame()
        df.drop(0, axis=0, inplace=True)
        # cust_ID_DF['customer_id/Name'] = df['Customer_ID/Name']
        # df.drop('Customer_ID/Name', axis=1, inplace=True)
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
        data_frame = cust_ID_DF.to_csv()
        # Showing on the platform
        sl.table(cust_ID_DF)
        # Download button for the file
        sl.download_button(label='Download tabel', data=data_frame, mime='text/csv', file_name='Bill Payment Status for Next month')

def pred_out(num):
    if num == 1:
        sl.warning('You should BUY at the start of the year and sell at the end of the year for a profit!')
    else:
        sl.success('You should NOT BUY at the start of the year and sell at the end of the year!')

if __name__ == '__main__':
    main()
