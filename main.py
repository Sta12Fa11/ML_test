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

st.set_page_config(page_title='เปรียบเทียบ Machine learning Algorithm',
                   page_icon='😎',
                   layout='centered',
                   initial_sidebar_state='expanded')
if 'SELECTED_COLUMNS' not in st.session_state:
    st.session_state.SELECTED_COLUMNS = []
st.session_state.SELECTED_COLUMNS = st.session_state.SELECTED_COLUMNS
if 'TEST_SIZE' not in st.session_state:
    st.session_state.TEST_SIZE = 20
st.session_state.TEST_SIZE = st.session_state.TEST_SIZE
if 'RANDOM_STATE' not in st.session_state:
    st.session_state.RANDOM_STATE = 50
st.session_state.RANDOM_STATE = st.session_state.RANDOM_STATE
if 'MODELS' not in st.session_state:
    st.session_state.MODELS = []
st.session_state.MODELS = st.session_state.MODELS
# if 'POWER' not in st.session_state:
#     st.session_state.POWER = 1
# st.session_state.POWER = st.session_state.POWER
source = pd.DataFrame()
model_choices = ['LinearRegression', 'PolynomialRegression', 'DecisionTreeRegressor']


# label = st.session_state.COLUMNS


st.title('Machine Learning Algorithm Comparison')
st.write('ใช้เป็นเครื่องมือเพื่ออำนวยความสะดวกในการสำรวจและวิเคราะห์ข้อมูลผ่านโมเดลในแบบต่าง ๆ')


def uploadfile(s):
    if s.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        s = pd.read_excel(s)
    elif s.type == 'text/csv':
        s = pd.read_csv(s)
    return s


def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}> คลิกเพื่อดาวน์โหลด {filename} </a>'
    return href


def showLinearReg():
    st.write('**Linear Regression**')
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)

    st.write(f'ค่า R² : {linReg.score(X_test, y_test):,.4f}')
    # for param in zip(linReg.feature_names_in_, linReg.coef_):
    #     st.write(param)
    # st.write(f'(intercept, {linReg.intercept_})')
    # st.write('**กำหนด Features**')
    # y_predicted = linReg.predict(X_train)


def display():
    st.subheader('1. แสดงชุดข้อมูล')
    st.dataframe(data, use_container_width=True)
    st.write(f'จำนวนแถว : {data.shape[0]:,d} แถว')
    st.write(f'จำนวนคอลัมน์ : {data.shape[1]:,d} คอลัมน์')
    st.write('')
    st.subheader('2. ตัวแปร')
    st.write(f'**Label** :  {label}')

    st.write(f'**Features** : ')
    for index, feature in enumerate(features):
        index+1, feature
    st.write('')
    st.subheader('3. แสดงความสัมพันธ์ของข้อมูลตัวเลข')
    st.dataframe(data.corr(), use_container_width=False)

    fig = sns.pairplot(data)
    st.write('')
    st.subheader('4. Pairplot')
    st.pyplot(fig)

    st.write('')
    st.subheader('5. เปรียบเทียบข้อมูลชุดเรียนรู้และข้อมูลชุดทดสอบ')
    st.write('**ข้อมูลชุุดเรียนรู้**')
    n_train = X_train.shape[0]
    st.write(f'จำนวน : {n_train} แถว')
    st.dataframe(predictions_train, use_container_width=True)
    st.write('**ข้อมูลชุุดทดสอบ**')
    n_test = X_test.shape[0]
    st.write(f'จำนวน : {n_test} แถว')
    st.dataframe(predictions_test, use_container_width=True)

    st.markdown('**R-squared**')
    #     # Tall
    # predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"]]
    # plt.figure(figsize=(3, 9))
    # sns.set_theme(style="whitegrid")
    # ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
    # ax1.set(xlim=(0, 1))
    # st.pyplot(plt)
    # st.markdown(imagedownload(plt, 'plot-r2-tall.pdf'), unsafe_allow_html=True)
    # Wide
    predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"]]
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)

    st.markdown(imagedownload(plt, 'plot-r2.pdf'), unsafe_allow_html=True)

    st.markdown('**RMSE (capped at average)**')
        # Tall
    #     predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"]]
    #     plt.figure(figsize=(3, 9))
    #     sns.set_theme(style="whitegrid")
    #     ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
    # st.markdown(imagedownload(plt, 'plot-rmse-tall.pdf'), unsafe_allow_html=True)
    # Wide
    predictions_test["RMSE"] = [predictions_test['RMSE'].mean() if i > predictions_test['RMSE'].mean() else i for i in predictions_test["RMSE"]]
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, 'plot-rmse.pdf'), unsafe_allow_html=True)

    # model display
    st.subheader('6. แสดงผลโมเดล')
    if st.session_state.MODELS is None:
        st.error('ยังไม่ได้ป้อนโมเดล')
    if 'LinearRegression' in st.session_state.MODELS:
        showLinearReg()
    else:
        st.write("")



# ---SIDEBAR---

with st.sidebar:
    st.header(':orange[Sidebar]')
    st.subheader('1. เพิ่มชุดข้อมูล')
    source1 = st.file_uploader('อัพโหลดไฟล์ excel หรือ csv', type=('xlsx', 'csv'),
                              accept_multiple_files=False,
                              label_visibility="collapsed")


try:
    source1 = uploadfile(source1)


    st.sidebar.subheader('2. เลือกข้อมูลที่ใช้')
    selected_columns = st.sidebar.multiselect('เลือกข้อมูลที่่ใช้',
                                              options=source1.columns,
                                              label_visibility='collapsed',
                                              key='SELECTED_COLUMNS'
                                              )

    st.sidebar.subheader('3. ระบุตัวแปรที่ต้องการคาดการณ์')
    label = st.sidebar.selectbox('ระบุตัวแปรที่ต้องการคาดการณ์ (Label)',
                                 options=selected_columns,
                                 key='LABEL', label_visibility='collapsed')
    data = source1[st.session_state.SELECTED_COLUMNS]

    st.sidebar.subheader('4. ลบแถวที่มีค่าว่าง')
    drop_na = st.sidebar.radio('ลบแถวว่าง', options=['ไม่ใช่', 'ใช่'], label_visibility='collapsed')
    if drop_na == 'ใช่':
        data.dropna(inplace=True)

    features = data.drop(columns=label)
    st.sidebar.subheader('5. ตัวแปรทดสอบชุดข้อมูล')
    st.sidebar.slider('ร้อยละของชุดข้อมูลทดสอบ', min_value=1, max_value=99, step=1, key='TEST_SIZE')
    st.sidebar.slider('กำหนดตัวเลขสุ่มข้อมูล', min_value=0, max_value=100, step=1, key='RANDOM_STATE')
    X = features[features.columns[:]]
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state.TEST_SIZE/100, random_state=st.session_state.RANDOM_STATE)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    model_train, predictions_train = reg.fit(X_train, X_train, y_train, y_train)
    model_test, predictions_test = reg.fit(X_train, X_test, y_train, y_test)
except:
    st.error('กรุณาเพิ่มชุดข้อมูลและระบุข้อมูลที่จำเป็นใน sidebar')
if source1 is not None:
    display()
    st.write('#')
    try:
        st.sidebar.subheader('6. อัพโหลด Feature ที่ต้องการคาดการณ์')
        source2 = st.sidebar.file_uploader('อัพโหลดข้อมูล Features ที่ต้องการคาดการณ์', type=('xlsx', 'csv'),
                                       accept_multiple_files=False,
                                       label_visibility="collapsed")
        source2 = uploadfile(source2)


        if source2 is not None:
            st.sidebar.subheader('7. เลือกโมเดล')
            st.sidebar.multiselect('', options=model_choices,
                               key='MODELS', label_visibility="collapsed")
        else:
            ""
    except:
        st.error('ยังไม่ได้อัพโหลด Features ที่ต้องการคาดการณ์')

st.sidebar.write('#')








# ---DISPLAY---




