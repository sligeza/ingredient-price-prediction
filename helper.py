import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


def plot_ts_diff_hist_acf_pcf(series, lags):
    
    fig = plt.figure(figsize=(16,9))
    plt.title('ACF and PCF')
    
    grid = plt.GridSpec(4, 2, wspace=0.4, hspace=0.7)
    
    test = sm.tsa.adfuller(series)
    ax0 = plt.subplot(grid[0, :])
    ax0.plot(series)
    ax0.set_title('Statinoary (alpha=5%): {}'.format( test[0] <= test[4]['5%'] ), fontsize=18)
    
    ax10 = plt.subplot(grid[1, 0])
    ax10.plot(series.diff(1))
    test = sm.tsa.adfuller(series.diff(1).dropna())
    ax10.set_title('Differnce\nStatinoary (alpha=5%): {}'.format( test[0] <= test[4]['5%'] ))
    
    
    ax11 = plt.subplot(grid[1, 1])
    ax11.hist(series)
    ax11.set_title('Histogram')
    
    ax2 = plt.subplot(grid[2, :])
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=ax2)
    
    ax3 = plt.subplot(grid[3, :])
    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=ax3)
    

def run_model(X, model_order, initial_train_size, forecast_steps=1, no_steps=None, success_metric=mean_absolute_error, plot_title='',plot_chart=True, sarimax_model=None):
    X = np.array(X)
    
    if no_steps is None:
        no_steps = X.shape[0] - initial_train_size - 1
        
    assert initial_train_size <= X.shape[0] - forecast_steps
    assert initial_train_size + no_steps < X.shape[0]

    
    train_test_split_idx = initial_train_size
    
    preds = []
    for idx in range(train_test_split_idx, X.shape[0], forecast_steps):
        train = list(X[:idx])
        
        #model = 
        model = sarimax_model if sarimax_model else ARIMA(train, order=model_order)
        model_fit = model.fit(disp=0)
        preds += list(model_fit.forecast(steps=forecast_steps)[0])
        
    
    
    test = X[initial_train_size:]
    preds = preds[:len(test)] if len(preds) != len(test) else preds
    score = np.round( success_metric(test, preds), 2)
    
    if plot_chart:
        plt.figure(figsize=(15, 5))
        plt.title('{}\norder={}, {}={}'.format(plot_title, model_order, success_metric.__name__, score))
        plt.plot( range(X.shape[0]), X, 'go-' , label='history')
        plt.plot(range(train_test_split_idx, X.shape[0]), preds, 'r*-.', label='forecasted')
        plt.legend();
    
    return score, preds