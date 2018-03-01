


import matplotlib.pyplot as plt, pandas as pd, numpy as np

from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


%cd "Predicting stocks prices via ML and NLP"


df = pd.read_csv('sp500_joined_closes.csv', index_col=0)



df_corr = df.corr()
data1 = df_corr.values
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
fig1.colorbar(heatmap1)
ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
ax1.invert_yaxis()
ax1.xaxis.tick_top()
column_labels = df_corr.columns
row_labels = df_corr.index
ax1.set_xticklabels(column_labels)
ax1.set_yticklabels(row_labels)
plt.xticks(rotation=90)
heatmap1.set_clim(-1, 1)
plt.tight_layout()
plt.show()





def proc(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df









#trading based on the percent variation strategy
def broker(*args):
    cols = [c for c in args]
    threshold = 0.02
    for col in cols:
        if col > threshold:
            return 1
        if col < -threshold:
            return -1
    return 0


#trading based on the double momentum strategy
def broker2(ticker, short_window, long_window, df):
    signals = pd.DataFrame(index=df[ticker].index)
    signals['signal'] = 0.0
    signals['signal'][short_window:]=np.where(
        df[ticker].rolling(window=short_window, min_periods=1, center=False).mean()[short_window:]
        >
        df[ticker].rolling(window=long_window, min_periods=1, center=False).mean()[short_window:]
        , 1.0, 0.0)

    return signals.diff().fillna(0).values.reshape(-1)



from collections import Counter



def extract_featuresets(ticker):
    tickers, df = proc(ticker)

    df['{}_target'.format(ticker)] = list(map(broker,      df['{}_1d'.format(ticker)],
    df['{}_2d'.format(ticker)],
    df['{}_3d'.format(ticker)],
    df['{}_4d'.format(ticker)],
    df['{}_5d'.format(ticker)],
    df['{}_6d'.format(ticker)],
    df['{}_7d'.format(ticker)]))

#    df['{}_target'.format(ticker)] = broker2(ticker, short_window, long_window, df)


    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df_vals = df[[i for i in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0).fillna(0)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df






def inference():
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))
    print()
    print()
    return confidence

clf = VotingClassifier([('lsvc',svm.LinearSVC()),
    ('knn',neighbors.KNeighborsClassifier()),('rfor',RandomForestClassifier())])

ticker='AAPL'
X, y, df = extract_featuresets(ticker)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)


inference()


scores = cross_val_score(estimator=clf, X=X_train, y=y_train,cv=5,scoring='accuracy')






# reading news from NY times
df_stocks = pd.read_pickle('Articles+dj.pkl')


df_stocks['prices'] = df_stocks['adj close'].astype('float')
df_stocks = df_stocks[['prices', 'articles']]
df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))

df_stocks["compound"] =df_stocks["neg"]=df_stocks["neu"]=df_stocks["pos"] = '' # Adding new columns to the data frame corresponding to polarity of sentences



from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
sid = SentimentIntensityAnalyzer()
for date, row in df_stocks.T.iteritems():
    try:
        sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles']).encode('ascii' ,'ignore')
        ss = sid.polarity_scores(sentence)
        df_stocks.loc[date, 'compound':'pos'] = ss.values()
    except TypeError:
        print date





year=2007

# Splitting the training and testing data
train_start_date = str(year) + '-01-01'
train_end_date = str(year) + '-10-31'
test_start_date = str(year) + '-11-01'
test_end_date = str(year) + '-12-31'
train = df_stocks.loc[train_start_date : train_end_date]
test = df_stocks.loc[test_start_date:test_end_date]

# Calculating the sentiment score
sentiment_score_list = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray \
        ([df_stocks.loc[date, 'compound'] ,df_stocks.loc[date, 'neg'] ,df_stocks.loc[date, 'neu'] ,df_stocks.loc[date, 'pos']])
    sentiment_score_list.append(sentiment_score)
numpy_df_train = np.asarray(sentiment_score_list)
sentiment_score_list = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray \
        ([df_stocks.loc[date, 'compound'] ,df_stocks.loc[date, 'neg'] ,df_stocks.loc[date, 'neu'] ,df_stocks.loc[date, 'pos']])
    sentiment_score_list.append(sentiment_score)
numpy_df_test = np.asarray(sentiment_score_list)





from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(numpy_df_train, train['prices'])
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [10,100,1000], 'max_features': [1,2]},
{'bootstrap': [False], 'n_estimators': [10,100,1000], 'max_features': [1,2]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(numpy_df_train,train['prices'])
forest_reg=grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print rf.feature_importances_








tickers, df = proc(ticker)
X, y, df = extract_featuresets(ticker)



#merging stocks values with NY news
df_vals = df[[i for i in tickers]].pct_change()
df_vals = df_vals.replace([np.inf, -np.inf], 0).fillna(0)
df_vals['{}_target'.format(ticker)]=df['{}_target'.format(ticker)] # creating the target y variable
df_vals = df_vals.replace([np.inf, -np.inf], 0).fillna(0)

df_comb=pd.merge(df_vals,df_stocks[['compound','neg','neu','pos']],left_index=True,right_index=True)
X_train = df_comb.loc[train_start_date: train_end_date].drop(['{}_target'.format(ticker)],axis=1).values
y_train = df_comb.loc[train_start_date: train_end_date]['{}_target'.format(ticker)].values
X_test = df_comb.loc[test_start_date: test_end_date].drop(['{}_target'.format(ticker)],axis=1).values
y_test = df_comb.loc[test_start_date: test_end_date]['{}_target'.format(ticker)].values




clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                        ('knn', neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier())])

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)


ticker='AAPL'
ii=np.where(ticker==df.keys())[0]
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
mlp = MLPClassifier(random_state=0,hidden_layer_sizes=(100,100),activation='tanh', solver='lbfgs', alpha=0.005,learning_rate_init=0.005, shuffle=True)
param_grid = {'hidden_layer_sizes': [100 ,(100, 100)] ,'learning_rate_init': [0.005, 0.01 ,0.015], 'alpha' :[0.002 ,0.005]}
_est=[]
grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5)



for X_train_tmp,X_test_tmp in zip([X_train[:,ii],X_train[:,:56],X_train], [X_test[:,ii],X_test[:,:56],X_test]):
    _est.append(clf.fit(X_train_tmp, y_train).score(X_test_tmp, y_test))










short_window = 40
long_window = 100

ticker='AAPL'
df.index=pd.to_datetime(df.index)
signals = pd.DataFrame(index=df[ticker].index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] =df[ticker].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = df[ticker].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]> signals['long_mavg'][short_window:], 1.0, 0.0)



# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111, ylabel='Price in $')

# Plot the closing price
df[ticker].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')

# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')

# Show the plot
plt.show()















##Simulating the sell-buy-hold process in trading

# Set the initial capital
initial_capital = float(100000.0)

# Create a DataFrame `positions`
positions = pd.DataFrame(index=signals.index).fillna(0.0)

# Buy a 100 shares
positions['AAPL'] = 100 * signals['signal']

# Initialize the portfolio with value owned
portfolio = positions.multiply(df[ticker], axis=0)

# Store the difference in shares owned
pos_diff = positions.diff()

# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(df[ticker], axis=0)).sum(axis=1)

# Add `cash` to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(df[ticker], axis=0)).sum(axis=1).cumsum()

# Add `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Print the first lines of `portfolio`
print(portfolio.head())


# Isolate the returns of your strategy
returns = portfolio['returns']

# annualized Sharpe ratio
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

# Print the Sharpe ratio
print(sharpe_ratio)

















### Keras and Tensor flow: LSTM



# coding: utf-8

%cd "Insight Artificial Intelligence"

import sys
sys.path.append("C:\\Users\defaultuser0\PycharmProjects\PyAndrea2\Insight Artificial Intelligence")


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries


#Step 1 Load Data
X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)



#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))  #Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

model.add(Dense(    #Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print ('compilation time : ', time.time() - start)


#Step 3 Train the model
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)



#Step 4 - Plot the predictions!
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)


