import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from matplotlib import pyplot as plt
from scipy.stats import f_oneway
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
import datetime
import streamlit as st
import joblib
from fpdf import FPDF
import base64
from tempfile import NamedTemporaryFile

st.header('Laundry Customer Data')

#dataset
df = pd.read_csv('dataset.csv')
weather = pd.read_csv('weather.csv')
weather.drop(weather.iloc[:, 0 : 3], axis=1, inplace = True)
frames = [df, weather]

merged_df = pd.concat(frames, axis = 1)

#data preprocessing
merged_df[['Washer_No', 'Dryer_No']] = merged_df[['Washer_No', 'Dryer_No']].astype(str)

df_cat = merged_df.select_dtypes(['object'])
df_cat = df_cat.apply(lambda x: x.str.strip())

mode = df_cat.mode()
df_cat.fillna(value = mode.iloc[0], inplace = True)

df_cat['Date'] = pd.to_datetime(df_cat['Date'])
df_cat['Day_of_Week'] = df_cat['Date'].dt.day_name()

df_cat['Time'] = df_cat['Time'].str[:2]
df_cat['Time'] = df_cat['Time'].str.replace(':', '')
df_cat['Time'] = df_cat['Time'].astype(int)
df_cat['Time'] = (df_cat['Time'] % 24) // 4
df_cat['Time'].replace({0: 'Late Night', 1: 'Early Morning', 2: 'Morning', 3: 'Afternoon', 4: 'Evening', 5: 'Night'}, inplace=True)

df_num = merged_df.select_dtypes(exclude=['object'])

mean = df_num.mean().round().to_frame().T
df_num.fillna(value = mean.iloc[0], inplace = True)

frames = [df_cat, df_num]

cleaned_df = pd.concat(frames, axis = 1)

#label Encoding
df_encoded = cleaned_df.copy()

cat_vars = df_encoded.select_dtypes(include=['object'])

label_encoder = preprocessing.LabelEncoder()

df_encoded = cat_vars.apply(label_encoder.fit_transform)


#map and value
date = st.date_input('Date', datetime.date(2015,10,19))
filtered_data = cleaned_df[cleaned_df['Date'].dt.date == date]
if(filtered_data.empty):
    spent = 0
    condition  = 'nan'
else:
    spent = filtered_data['TotalSpent_RM'].mean()
    condition = filtered_data['condition'].mode()[0]
    
st.subheader('Customers at %s' % date)
st.subheader('Weather condition : {}'.format(condition))
st.subheader('Mean Total Spent : RM {:.2f}' .format(spent))
st.map(filtered_data)

figs = []

#Questions
st.header('Question 1: What is the basket size the customers will bring?')
st.subheader('Exploratory Data Analysis')
fig = plt.figure(figsize = (10,6))
plt.title('Count plot of Race separated by Basket Size')
ax = sns.countplot(data = df, x = 'Race', hue = 'Basket_Size')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)

fig = plt.figure(figsize = (10,6))
plt.title('Count plot of Gender separated by Basket Size')
ax = sns.countplot(data = merged_df, x = 'Gender', hue = 'Basket_Size')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)

fig = plt.figure(figsize = (10,6))
plt.title('Count Plot of Body Size separated by Basket Size')
ax = sns.countplot(data = merged_df, x = 'Body_Size', hue = 'Basket_Size')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)

fig = plt.figure(figsize = (10,6))
plt.title('Count plot of customer with kids separated by Basket Size')
ax = sns.countplot(data = merged_df, x = 'With_Kids', hue = 'Basket_Size')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)

fig = plt.figure(figsize = (10,6))
plt.title("Count plot of Customer's Attire separated by Basket Size")
ax = sns.countplot(data = merged_df, x = 'Attire', hue = 'Basket_Size')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)


st.header('Question 2: Did weather information impact the sales?')
st.subheader('Exploratory Data Analysis')
fig = plt.figure(figsize = (10,6))
plt.title('Bar plot of mean total spent in RM separated by weather condition')
ax = sns.barplot(x ='condition', y ='TotalSpent_RM', data = merged_df, palette ='plasma', estimator = np.mean)

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)

fig = plt.figure(figsize = (10,6))
plt.title('Scatter plot of total spent in RM and humidity')
ax = sns.scatterplot(x ='humidity', y ='TotalSpent_RM', data = merged_df, palette ='plasma')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)

fig = plt.figure(figsize = (10,6))
plt.title('Scatter plot of total spent in RM and min temperature')
ax = sns.scatterplot(x ='tempmin', y ='TotalSpent_RM', data = merged_df, palette ='plasma')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)

fig = plt.figure(figsize = (10,6))
plt.title('Scatter plot of total spent in RM and max temperature')
ax = sns.scatterplot(x ='tempmax', y ='TotalSpent_RM', data = merged_df, palette ='plasma')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)

st.header('Question 3: Will time spent affect total spent?')
st.subheader('Exploratory Data Analysis')
fig = plt.figure(figsize = (10,6))
plt.title('Scatter plot of total spent in RM and max temperature')
ax = sns.scatterplot(x ='TimeSpent_minutes', y ='TotalSpent_RM', data = merged_df, palette ='plasma')

for container in ax.containers:
    ax.bar_label(container)
    
st.pyplot(fig)
figs.append(fig)


st.header('Question 4: Which Washer and Dryer will likely to been choosen by customer?')
st.subheader('Exploratory Data Analysis')
fig = plt.figure(figsize = (10,7))
plt.title("Count plot of Race separated by Washer Number")
ax = sns.countplot(data = merged_df, x = 'Race', hue = 'Washer_No')

for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig)
figs.append(fig)


fig = plt.figure(figsize = (10,7))
plt.title("Count plot of Race separated by Dryer Number")
ax = sns.countplot(data = merged_df, x = 'Race', hue = 'Dryer_No')

for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig)
figs.append(fig)


fig = plt.figure(figsize = (10,7))
plt.title("Count plot of Gender separated by Washer Number")
ax = sns.countplot(data = merged_df, x = 'Gender', hue = 'Washer_No')

for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig)
figs.append(fig)


fig = plt.figure(figsize = (10,7))
plt.title("Count plot of Gender separated by Dryer Number")
ax = sns.countplot(data = merged_df, x = 'Gender', hue = 'Dryer_No')

for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig)
figs.append(fig)


fig = plt.figure(figsize = (10,7))
plt.title("Count plot of Customers With Kids separated by Washer Number")
ax = sns.countplot(data = merged_df, x = 'With_Kids', hue = 'Washer_No')

for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig)
figs.append(fig)


fig = plt.figure(figsize = (10,7))
plt.title("Count plot of Customers With Kids separated by Dryer Number")
ax = sns.countplot(data = merged_df, x = 'With_Kids', hue = 'Dryer_No')

for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig)
figs.append(fig)


export_as_pdf = st.button("Export EDA Report")

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

if export_as_pdf:
    pdf = FPDF()
    for fig in figs:
        pdf.add_page()
        with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig.savefig(tmpfile.name)
                pdf.image(tmpfile.name, 10, 10, 200, 100)
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "EDA Report")
    st.markdown(html, unsafe_allow_html=True)
    
st.header('Prediction of data')

rf = joblib.load('rf_os_fs_model.sav')
xgb = joblib.load('xgbr_model.sav')

left_column, right_column = st.columns(2)
with left_column:
    st.subheader('Predict Basket Size')
    Time = []
    Race = []
    Kids_Category = []
    Basket_Colour = []
    Shirt_Colour = []
    Pants_Colour = []
    Washer_No = []
    Dryer_No = []
    Day_of_Week = []
    
    ti = st.selectbox('Time',('Night', 'Late Night', 'Early Morning', 'Morning', 'Afternoon', 'Evening'))
    Time.append(ti)
    
    r = st.selectbox('Race', ('malay', 'indian', 'chinese', 'foreigner '))
    Race.append(r)
    
    kc = st.selectbox('Kids Category',('young', 'no_kids', 'toddler ', 'baby'))
    Kids_Category.append(kc)
    
    bc = st.selectbox('Basket Colour', ('red', 'white', 'blue', 'black', 'pink', 'purple', 'yellow',
       'brown', 'orange', 'green', 'grey'))
    Basket_Colour.append(bc)
    
    sc = st.selectbox('Shirt Colour',('blue', 'white', 'red', 'black', 'brown', 'yellow', 'grey',
       'green', 'purple', 'orange', 'pink'))
    Shirt_Colour.append(sc)
    
    pc = st.selectbox('Pants_Colour', ('black', 'blue_jeans', 'yellow', 'white', 'brown', 'grey',
       'orange', 'blue', 'green', 'red', 'purple', 'pink'))
    Pants_Colour.append(pc)
    
    wn  = st.selectbox('Washer No', ('3', '6', '4', '5'))
    Washer_No.append(wn)
    
    dn  = st.selectbox('Dryer No', ('10', '9', '8', '7'))
    Dryer_No.append(dn)
    
    dow = st.selectbox('Day of Week', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
    Day_of_Week.append(dow)
    
    predict = pd.DataFrame()
    predict['Time'] = Time
    predict['Race'] = Race
    predict['Kids_Category'] = Kids_Category
    predict['Basket_Colour'] = Basket_Colour
    predict['Shirt_Colour'] = Shirt_Colour
    predict['Pants_Colour'] = Pants_Colour
    predict['Washer_No'] = Washer_No
    predict['Dryer_No'] = Dryer_No
    predict['Day_of_Week'] = Day_of_Week
    label_encoder = preprocessing.LabelEncoder()

    pred_encoded = predict.apply(label_encoder.fit_transform)
    if st.button('Predict Basket Size'):
        makeprediction = rf.predict(pred_encoded)
        if(makeprediction == 0):
            st.success('The predicted basket size is small')
        elif(makeprediction == 1):
            st.success('The predicted basket size is big')
        
with right_column:
    st.subheader('Predict Total Spent')
    tempmin = []
    tempmax = []
    temp = []
    humidity = []
    windspeed = []
    cloudcover = []
    solarenergy = []
    tmin = st.slider('Min Temperature of the day ', 73.4, 82.4, 60.5)
    tempmin.append(tmin)
    
    tmax = st.slider('Max Temperature of the day' , 82.4, 96.8, 90.5)
    tempmax.append(tmax)
    
    t = st.slider('Temperature of the day' , 77.7, 87.4, 80.5)
    temp.append(t)
    
    h = st.slider('Humidity of the day ', 59.2, 91.7, 60.5)
    humidity.append(h)
    
    ws = st.slider('Windspeed of the day ', 3.4, 18.2, 5.6)
    windspeed.append(ws)
    
    cc = st.slider('Cloudcover of the day ', 85.8, 98.0, 96.2)
    cloudcover.append(cc)
    
    se = st.slider('Solarenergy of the day ', 7.6, 27.6, 15.7)
    solarenergy.append(se)
    
    weather = pd.DataFrame()
    weather['tempmax'] = tempmax
    weather['tempmin'] = tempmin
    weather['temp'] = temp
    weather['humidity'] = humidity
    weather['windspeed'] = windspeed
    weather['cloudcover'] = cloudcover
    weather['solarenergy'] = solarenergy
    if st.button('Predict Total Spent'):
        makeprediction = xgb.predict(weather)
        output = round(makeprediction[0],2)
        st.success('The predicted total spent is RM{}'.format(output))
