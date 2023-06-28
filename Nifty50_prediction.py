import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.style as style
import plotly.graph_objects as go
import mpld3
import ta

# Set the theme configuration
st.markdown(
    """
    <style>
    body {
        background-color: #3366ff;
    }
    .ticker1 {
        color: lightgreen;
    }
    .index {
        color: lightblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Nifty50 stock comparative analysis')

# Display image
image = "https://img.etimg.com/thumb/msid-79611089,width-1015,height-550,imgsize-299518,resizemode-8/prime/money-and-markets/is-nifty-overvalued-a-short-term-correction-may-settle-the-debate-.jpg"  # Replace with the path to your image file
st.image(image, use_column_width=True)


# Render the date inputs with modified label text
font_size = 28
st.markdown(f"<label style='font-size: {font_size}px;'>Start Date</label>", unsafe_allow_html=True)
start_date = st.date_input('', value=pd.to_datetime('2020-01-01'), key='start_date')
st.markdown(f"<label style='font-size: {font_size}px;'>End Date</label>", unsafe_allow_html=True)
end_date = st.date_input('', value=pd.to_datetime('2022-12-31'), key='end_date')


start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

# Define the index tickers
index_tickers = {
    "^NSEI": "Nifty50",
    "^BSESN": "Sensex",
    "^NSEBANK": "BankNifty",
    "NIFTY_FIN_SERVICE.NS": "FinNifty"
}

# Function to get live index rate
def get_live_index_rate(index_ticker):
    try:
        data = yf.download(index_ticker, period='1d')
        if len(data) > 0:
            live_rate = data['Close'][-1]
            return live_rate
        else:
            st.warning(f"No data available for {index_ticker}")
    except Exception as e:
        st.error(f"Error retrieving data for {index_ticker}: {str(e)}")

# Display live index rates
st.title('CURRENT INDEX PRICE(NSE)')
columns = st.columns(len(index_tickers))
for i, index_ticker in enumerate(index_tickers.keys()):
    index_name = index_tickers[index_ticker]
    live_rate = get_live_index_rate(index_ticker)
    if live_rate is not None:
        column = columns[i]
        column.write(f"<h2 class='index'>{index_name}</h2>", unsafe_allow_html=True)
        column.write(f"<p class='ticker1'>Index Rate: {live_rate}</p>", unsafe_allow_html=True)

        
# Function to calculate RSI
def calculate_rsi(data):
    rsi = ta.momentum.RSIIndicator(data['Close']).rsi()
    return rsi

# Add a chart displaying historical index prices
st.header("Historical Index Prices")
data = yf.download(list(index_tickers.keys()), start=start, end=end)["Close"]
st.line_chart(data)

# Add a table displaying index information
st.header("Index Information")
df = pd.DataFrame.from_dict(index_tickers, orient="index", columns=["Index Name"])
df["Live Rate"] = [get_live_index_rate(ticker) for ticker in index_tickers.keys()]
st.dataframe(df)

      
st.title('Stock Trend Prediction')
# Define a list of stock tickers
stock_tickers = {
    "^NSEI": "NIFTY50 Index",
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BHARTIARTL.NS": "Bharti Airtel",
    "BRITANNIA.NS": "Britannia Industries",
    "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India",
    "DIVISLAB.NS": "Divi's Laboratories",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries",
    "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank",
    "HDFC.NS": "HDFC",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS": "Hindalco Industries",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "IOC.NS": "Indian Oil Corporation",
    "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS": "Infosys",
    "ITC.NS": "ITC",
    "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "M&M.NS": "Mahindra & Mahindra",
    "MARUTI.NS": "Maruti Suzuki",
    "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC",
    "ONGC.NS": "Oil and Natural Gas Corporation",
    "POWERGRID.NS": "Power Grid Corporation of India",
    "RELIANCE.NS": "Reliance Industries",
    "SBIN.NS": "State Bank of India",
    "SHREECEM.NS": "Shree Cement",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "TCS.NS": "Tata Consultancy Services",
    "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "UPL.NS": "UPL",
    "WIPRO.NS": "Wipro",
    "ZEEL.NS": "Zee Entertainment Enterprises"
}


# Get user input for stock ticker
selected_ticker = st.selectbox('Select Stock Ticker', stock_tickers, index=0)

# Download data for the selected stock ticker
df = yf.download(selected_ticker, start=start, end=end).reset_index()

#Describing Data
st.subheader('Data Description')
st.write(df.describe())


s='2010-01-01'
e='2022-12-31'
daf = yf.download(selected_ticker, start=s, end=e).reset_index()

#splitting data into training and testing
data_training=pd.DataFrame(daf['Close'][0:int(len(daf)*0.70)])
data_testing=pd.DataFrame(daf['Close'][int(len(daf)*0.70):int(len(daf))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1)) 

data_training_array=scaler.fit_transform(data_training)

#Slitting data  into x_train anf y_train
x_train=[]
y_train=[]
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

#ML model
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=1)

#Testing part
past_100_days=data_training.tail(100)
final_df=pd.concat([past_100_days,data_testing],ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# Calculate Bollinger Bands
st.subheader('Bollinger Bands Chart')
indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
bb_middle = indicator_bb.bollinger_mavg()
bb_upper = indicator_bb.bollinger_hband()
bb_lower = indicator_bb.bollinger_lband()

# Create Bollinger Bands Chart
bb_fig = go.Figure()

bb_fig.add_trace(go.Scatter(
    x=df['Date'],
    y=bb_middle,
    name='Middle Band',
    line=dict(color='blue')
))

bb_fig.add_trace(go.Scatter(
    x=df['Date'],
    y=bb_upper,
    name='Upper Band',
    line=dict(color='orange')
))

bb_fig.add_trace(go.Scatter(
    x=df['Date'],
    y=bb_lower,
    name='Lower Band',
    line=dict(color='orange')
))

bb_fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_white'
)

# Show the Bollinger Bands chart using Streamlit's st.plotly_chart function
st.plotly_chart(bb_fig, use_container_width=True)

# Calculate Technical Indicators
st.subheader('HLC Candlestick Chart with Technical Indicators')
rsi = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
macd = ta.trend.MACD(close=df['Close']).macd()
ema_short = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
ema_long = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()

# Create HLC Bar Chart with Technical Indicators
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='HLC Bar Chart',
    increasing_line_color='green',
    decreasing_line_color='red'
))

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=rsi,
    name='RSI',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=macd,
    name='MACD',
    line=dict(color='orange')
))

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=ema_short,
    name='EMA Short',
    line=dict(color='purple')
))

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=ema_long,
    name='EMA Long',
    line=dict(color='pink')
))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_white'
)

# Show the chart using Streamlit's st.plotly_chart function
st.plotly_chart(fig, use_container_width=True)

#Visualization
st.subheader('Closing price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


# Calculate RSI
st.subheader('RSI Value Indicator')
rsi = ta.momentum.RSIIndicator(df.Close).rsi()
# Add RSI subplot
fig.add_subplot().plot(rsi, 'purple')
st.pyplot(fig)

# User input for plot selection
st.subheader('Comparision Plots for Open, Close, High, Low, Adj Close and Volume')
plot_choice = st.selectbox(' ', ('Opening value vs Closing',
                                     'Highest value vs Closing', 'Lowest value vs Closing',
                                      'Date vs Volume','Closing VS Adj Close'), index=0)
load = st.progress(0)
style.use('ggplot')

if plot_choice == 'Closing VS Adj Close':
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close', c='b')
    plt.plot(df.index, df['Adj Close'], label='Adj Close', c='r')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Closing Value vs Adj Closing')
    plt.legend()
    st.pyplot(fig)

elif plot_choice == 'Opening value vs Closing':
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Open'], label='Open', c='b')
    plt.plot(df.index, df['Close'], label='Close', c='r')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Opening value vs Closing')
    plt.legend()
    st.pyplot(fig)

elif plot_choice == 'Highest value vs Closing':
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['High'], label='High', c='g')
    plt.plot(df.index, df['Close'], label='Close', c='b')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Highest value vs Closing')
    plt.legend()
    st.pyplot(fig)

elif plot_choice == 'Lowest value vs Closing':
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Low'], label='Lowest value', c='c')
    plt.plot(df.index, df['Close'], label='Close', c='b')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Lowest value vs Closing')
    plt.legend()
    st.pyplot(fig)

elif plot_choice == 'Date vs Volume':
    fig=plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Volume'], c='c')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    st.pyplot(fig)
