import datetime as dt
import webbrowser as wb
import streamlit as st
import numpy as np
import pandas as pd
import streamlit as sl
import streamlit.components.v1 as com

# Import các module cần thiết
import joblib

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

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to use a machine learning (ML) model to predict if this company is FRAUD or NOT FRAUD based on M-score.')
  
  st.markdown('**What is M-score?**')
  st.info(''' M-Score is a mathematical model that uses eight financial ratios weighted by coefficients to identify whether a company has manipulated its profits.
  The M-score model's eight variables are:
1. DSRI: Days' sales in a receivable index
2. GMI: Gross margin index
3. AQI: Asset quality index
4. SGI: Sales growth index
5. DEPI: Depreciation index
6. SGAI: Sales and general and administrative expenses index
7. LVGI: Leverage index
8. TATA: Total accruals to total assets

  The eight variables are then weighted together according to the following formula:
Beneish M-Score = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI – 0.172*SGAI + 4.679*TATA – 0.327*LVGI
  ''')


# Selection 
rad_b = st.radio('Please select that you want to give Single or Multiple data', options=['Single', 'Multiple'])

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
        dob = form.number_input('Accounts Receivables')

        col1, col2, col3 = form.columns(3)
        social_friend_count = col1.number_input('Revenue')
        unknown_var_17 = col2.number_input('Number of Days')
        unknown_var_1 = col3.number_input('Gross Profit Margin')
        
        col14, col15, col4 = form.columns(3)
        unknown_var_15 = col14.number_input('PPE')
        unknown_var_14 = col15.number_input('Total Long-termInvestments')
        unknown_var_2 = col4.number_input('Current Assets')

        
        col16,col17, col18 = form.columns(3)
        unknown_var_12 = col16.number_input('Total Assets')
        unknown_var_10 = col17.number_input('Depreciation')
        unknown_var_9 = col18.number_input('Sales & General Administration Expenses')

        col20,col21, col22 = form.columns(3)
        unknown_var_7 = col20.number_input('Income from Continuing Operations')
        unknown_var_3 = col21.number_input('Cash Flow from Operations')
        num_date_review = col22.number_input('Current Liabilities')
        
        col19,col23 = form.columns(2)
        unknown_var_8 = col19.number_input('Total Revenue')
        value = col23.number_input('Total Long-term Debt')

        features = [value, num_date_review, dob, unknown_var_1, unknown_var_2, unknown_var_3,
                    unknown_var_7, unknown_var_8, unknown_var_9, unknown_var_10, unknown_var_12,
                    unknown_var_14, unknown_var_15, social_friend_count]

        pred = scaler.predict(np.array(features).reshape(1, -1))

        P_status = form.form_submit_button("Predict")
        if P_status:
            pred_out(pred)

    else:
        # Multi transaction 
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
        sl.warning('The model predicts that this company is FRAUD')
    else:
        sl.success('The model predicts that this company is NOT FRAUD')

if __name__ == '__main__':
    main()
