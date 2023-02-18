import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import base64
import io
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='เปรียบเทียบ Machine learning Algorithm',
                   page_icon='😎',
                   layout='centered',
                   initial_sidebar_state='expanded')

# ---DEFINE---
data = pd.DataFrame()
model_choices = ['LinearRegression', 'PolynomialRegression', 'DecisionTreeRegressor', 'RandomForestRegressor']

# ---SESSION STATE---
if 'SELECTED_COLOUMNS' not in st.session_state:
    st.session_state.SELECTED_COLUMNS = []
st.session_state.SELECTED_COLUMNS = st.session_state.SELECTED_COLUMNS
if 'DATA' not in st.session_state:
    st.session_state.DATA = pd.DataFrame()
st.session_state.DATA = st.session_state.DATA


# อัพโหลดไฟล์
def upload_file(s):
    if s.type == 'text/csv':
        s = pd.read_csv(s)
    else:
        s = pd.read_excel(s)
    return s


def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>ดาวน์โหลด {filename} </a>'
    return href


# ---DISPLAY1
def display2():
    st.write('---')
    st.subheader('2. แสดงชุดข้อมูลทั้งหมด')
    st.write('**ตารางแสดงข้อมูล**')
    st.dataframe(source1, use_container_width=True)
    st.write('')
    st.write('**ข้อมูลทางสถิติ**')
    st.write(f'จำนวนแถว : {source1.shape[0]:,d} แถว')
    st.write(f'จำนวนคอลัมน์ : {source1.shape[1]:,d} คอลัมน์')
    st.write(source1.describe().T)
    st.write('**แสดงความสัมพันธ์ของข้อมูล**')
    st.write(source1.corr())
    st.write('')


def display3():
    st.write('**ชุดข้อมูลที่นำมาวิเคราะห์**')
    st.write(f'จำนวนแถว : {data.shape[0]:,d} แถว')
    st.write(f'จำนวนคอลัมน์ : {data.shape[1]:,d} คอลัมน์')
    st.dataframe(data)
    st.write('**แสดงกราฟ Pair plot**')
    pair_plot = sns.pairplot(data)
    st.pyplot(pair_plot)


def separate_feature(d, l):
    features = d.drop(columns=l)
    label = d[l]
    return features, label


def display4():

    st.write(f'**Features :**')
    for i, c in enumerate(X.columns):
        i+1, c


# def linReg():
#     st.subheader('Linear Regression')
#     lm = LinearRegression()
#     lm.fit(X_train, y_train)
#     lm_result = lm.predict(X_test)
#     lm_coef = lm.coef_
#     lm_var = lm.feature_names_in_
#     lm_intercept = lm.intercept_
#     lm_score = lm.score(X_test, y_test)
#     lm_rmse = np.sqrt(mean_squared_error(y_test, lm_result))
#     st.write(f'ค่า R² : {lm_score:,.4f}')
#     st.write(f'ค่า RMSE : {lm_rmse:,.2f}')
#     st.write('**ค่าสัมประสิทธิ์**')
#     for i in zip(lm_var, lm_coef):
#         str(i[0]) + " : " + str(i[1])
#     st.write('**ค่าคงที่**')
#     st.write(f'intercept : {lm_intercept}')
#     st.write('#')
#
#
#
# def polyReg():
#     st.subheader('Polynomial Regression')
#     degree = st.number_input('อันดับ(degree)', value=2, min_value=1, max_value=10, step=1)
#     poly = PolynomialFeatures(degree)
#     X_train_poly = poly.fit_transform(X_train)
#     X_test_poly = poly.fit_transform(X_test)
#     pm = LinearRegression()
#     pm.fit(X_train_poly, y_train)
#     y_predicted = pm.predict(X_test_poly)
#     score = pm.score(X_test_poly, y_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
#     st.write(f'ค่า R² : {score:,.4f}')
#     st.write(f'ค่า RMSE : {rmse:,.2f}')
#     st.write('#')


# ---HEADER---
st.title('Machine Learning Algorithm \n ## _for Predictive Analytics_ 🗿')
st.write('*by  Sorawut Nitsunkit*')
st.write('เป็นเครื่องมือเพื่ออำนวยความสะดวกในการสำรวจและวิเคราะห์ข้อมูลผ่านโมเดลในแบบต่าง ๆ')

st.write('')

# ---PAGE---
st.subheader('1. อัพโหลดไฟล์')
try:
    source1 = st.file_uploader('อัพโหลดไฟล์ต้องการสร้างโมเดล',
                               type=('xlsx', 'csv'), accept_multiple_files=False)
    source1 = upload_file(s=source1)
except:
    st.error('ยังไม่ได้อัพโหลดไฟล์')
else:
    display2()
    st.write('---')
    st.subheader('3. เลือกข้อมูลที่ใช้วิเคราะห์')
    with st.form('เลือกข้อมูล'):
        st.write('**เลือกข้อมูล**')
        selected_columns = st.multiselect('เลือกข้อมูลที่ใช้',
                                      options=source1.columns,
                                      label_visibility='collapsed',
                                      )
        st.write('**การจัดการข้อมูลเบื้องต้น**')
        drop_na = st.radio('ลบแถวที่มีค่าว่าง', options=('ไม่', 'ใช่'),
                           horizontal=True)

        submited_columns = st.form_submit_button('ยืนยันข้อมูล', type='primary')

    if submited_columns:
        if selected_columns == []:
            st.error('ยังไม่ได้เลือกข้อมูล')
        else:
            data = source1[selected_columns]
            if drop_na == 'ใช่':
                data.dropna(inplace=True)
            st.session_state.DATA = data
            display3()
    data = st.session_state.DATA
    st.write('---')
    st.subheader('4. เลือก Label & Features')
    label_column = st.selectbox('**เลือก Label**', options=data.columns,
                         key='LABEL_COLUMNS')
    try:
        X, y = separate_feature(data, label_column)
        display4()
    except:
        ""
    else:
        st.write('---')
        st.subheader('5. เลือกโมเดล')
        models = st.multiselect('เลือกโมเดล', options=model_choices)
        test_size = st.slider('กำหนดสัดส่วนชุดทดสอบของชุดข้อมูล',
                              value=0.20, min_value=0.00, max_value=1.00,
                              step=0.01)
        random_state = st.slider('Random state', value=50,
                                 min_value=1, max_value=100, step=1)
        X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        st.write(f'**จำนวนข้อมูลชุดทดสอบ {X_test.shape[0]} แถว จาก {data.shape[0]} แถว**')
        X_axis = st.selectbox('**เลือกตัวแปรสำหรับแกน X เพื่อพลอตกราฟ**', options=X.columns)
        st.write('')

    #

        ml = pd.DataFrame(X)
        ml.sort_values(by=X_axis, inplace=True)
        ml2 = ml.copy()
        # Linear Regression
        if 'LinearRegression' in models:
            st.subheader('Linear Regression')
            lm = LinearRegression()
            lm.fit(X_train, y_train)
            lm_result = lm.predict(X_test)
            lm_predicted = lm.predict(ml)
            lm_coef = lm.coef_
            lm_var = lm.feature_names_in_
            lm_intercept = lm.intercept_
            lm_score = lm.score(X_test, y_test)
            lm_rmse = np.sqrt(mean_squared_error(y_test, lm_result))
            st.write(f'ค่า R² : {lm_score:,.4f}')
            st.write(f'ค่า RMSE : {lm_rmse:,.2f}')
            st.write('**ค่าสัมประสิทธิ์**')
            for i in zip(lm_var, lm_coef):
                str(i[0]) + " : " + str(i[1])
            st.write('**ค่าคงที่**')
            st.write(f'intercept : {lm_intercept}')
            st.write('#')
            ml['LinearRegression'] = lm_predicted

        if 'PolynomialRegression' in models:
            st.subheader('Polynomial Regression')
            degree = st.number_input('อันดับ(degree)', value=2, min_value=1, max_value=10, step=1)
            poly = PolynomialFeatures(degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.fit_transform(X_test)
            X_features = poly.fit_transform(ml2)
            pm = LinearRegression()
            pm.fit(X_train_poly, y_train)
            y_result = pm.predict(X_test_poly)
            pm_predicted = pm.predict(X_features)
            pm_score = pm.score(X_test_poly, y_test)
            pm_rmse = np.sqrt(mean_squared_error(y_test, y_result))
            st.write(f'ค่า R² : {pm_score:,.4f}')
            st.write(f'ค่า RMSE : {pm_rmse:,.2f}')
            ml['PolynomialRegression'] = pm_predicted
            st.write('#')

        fig = px.scatter(x=X[X_axis], y=y, title=f'แสดงกราฟระหว่าง {X_axis} และ {label_column}')
        fig.update_xaxes(title=X_axis)
        fig.update_yaxes(title=label_column)


        if 'LinearRegression' in models:
            fig.add_trace(go.Line(x=ml[X_axis], y=ml['LinearRegression'], name='Linear Regression'))

        if 'PolynomialRegression' in models:
            fig.add_trace(go.Scatter(x=ml[X_axis], y=ml['PolynomialRegression'], name='Polynomial Regression',mode="lines", marker_color="red"))

        st.plotly_chart(fig, use_container_width=True)
        st.write('---')

        st.subheader('6. อัพโหลด Features')
        source2 = st.file_uploader('อัพโหลด Features ที่ต้องการคาดการณ์', type=('xlsx', 'csv'),
                                   accept_multiple_files=False)
    try:
        source2 = upload_file(source2).sort_values(by=X_axis)
        source2_origin = source2.copy()
        source2_origin
    except ValueError:
        st.error('ข้อมูลไม่ตรงกัน')
    except:
        st.error('ยังไม่ได้อัพโหลดไฟล์ที่สอดคล้องกับข้อมูล')
    else:
        if source2.columns.equals(X.columns):
            st.info('ข้อมูลถูกต้อง')
            st.write('')
            st.write('**Preview Features**')
            st.dataframe(source2)

        else:
            st.error('คอลัมน์ไม่สอดคล้องกัน')
        try:
            if 'LinearRegression' in models:
                source2_1 = source2.copy()
                lm_forecast = lm.predict(source2_1)
                lm_forecast = pd.DataFrame(lm_forecast)
            if 'PolynomialRegression' in models:
                source2_2 = source2.copy()
                pm_forecast = pm.predict(poly.fit_transform(source2_2))
                pm_forecast = pd.DataFrame(pm_forecast)
        except:
            "error"
        try:
            source2[label_column + '_linReg_forecast'] = lm_forecast
        except:
            ""
        try:
            source2[label_column + '_polyReg_forecast'] = pm_forecast
        except:
            ""
        st.subheader('7. แสดงการคาดการณ์ด้วยโมเดล')
        fig2 = px.scatter(x=X[X_axis], y=y, title=f'แสดงกราฟระหว่าง {X_axis} และ {label_column}')
        fig2.update_xaxes(title=X_axis)
        fig2.update_yaxes(title=label_column)

        try:
            fig2.add_trace(go.Line(x=source2[X_axis], y=source2[label_column + '_linReg_forecast'], name=f'{label_column} _linReg_forecast'))
        except:
            ""
        try:
            fig2.add_trace(go.Line(x=source2[X_axis], y = source2[label_column + '_polyReg_forecast'], name=f'{label_column} _polyReg_forecast' ))
        except:
            ""
        st.plotly_chart(fig2)
        st.write('')
        st.write('**ตารางแสดงผลการคาดการณ์**')
        st.dataframe(source2)
        st.markdown(filedownload(source2, 'forecast_result.csv'), unsafe_allow_html=True)
