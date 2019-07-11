import numpy as np
import pandas as pd
from pandas_datareader import data as web
from scipy.stats import norm
import matplotlib.pyplot as plt


def MonteCarlo(data, iterations, strike, maturity):
    """ This function makes use of Monte Carlo simulation
        to calculate the price of a call option on the
        underlying asset.
    """

    log_returns = np.log(1 + data.pct_change())

    rf = 0.025 # risk-free rate

    stdv = log_returns.std() * np.sqrt(250) # Annualized standard deviation

    T = maturity # set the maturity
    t_intervals = 250 
    delta_t = T / t_intervals

    iterations = iterations 

    Z = np.random.standard_normal((t_intervals+1, iterations))
    S = np.zeros_like(Z) # price matrix

    # set the first row of the price matrix to be the last price in the dataframe
    S[0] = data.iloc[-1]

    # Perform simulations
    for t in range(1, t_intervals + 1):
        S[t] = S[t - 1] * np.exp((rf - 0.5 * stdv**2)
                               * delta_t + stdv*np.sqrt(delta_t)*Z[t])

    # Calculate Payoff
    strike = strike
    pyff = np.maximum(S[-1] - strike, 0)

    C = np.exp(-rf * T) * np.sum(pyff) / iterations # calculate call oprice
    print("The price of the call option on %s is:" % data.name, C)

    # Visulize Monte Carlo
    fig = plt.figure(figsize=(9, 6))
    plt.xlim(0, t_intervals)
    plt.grid(True)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Monte Carlo Simulation')
    plt.plot(S[:,:])
    plt.show()

    return C

def main(*args, **kwargs):
    # List of tickers to get pricing data for
    tickers = ['AAPL', 'DLR', 'GOV', 'CHK', 'MYGN', 'ORA']

    data = pd.read_csv('newpfolio.csv', index_col=0,
                       parse_dates=True)[tickers[0]]

    MonteCarlo(data, 1000, 150, 1.0)

    return 0;

if __name__ == '__main__':
    main()
