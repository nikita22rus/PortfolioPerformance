import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

import matplotlib.lines as mlines
from tqdm import tqdm_notebook

import yfinance as yf








# Для отрисовки сетки за графиком
plt.rc('axes', axisbelow=True)

# Функция вычисляющая риск и доходность для конкретного портфеля
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) * num_periods_annually
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(num_periods_annually)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_risk(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]


def min_risk(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_risk, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result



def neg_portfolio_return(weights, mean_returns, cov_matrix):
    return -1*portfolio_performance(weights, mean_returns, cov_matrix)[1]

def max_return(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(neg_portfolio_return, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_risk, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


# Оптимизация по максимальному коэф. Шарпа для части таблицы (данных)
def opt_max_sharpe_ratio(part_price, risk_free_rate):
    returns = part_price.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Оптимизация по заданному риску
def opt_targeted_risk(part_price, target):
    returns = part_price.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_risk(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[0]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_risk(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(neg_portfolio_return, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result



# Оптимизация по заданной доходности
def opt_targeted_return(part_price, target):
    returns = part_price.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_risk, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result




#параметр форматирования данных при выводе в терминале (при желании можно дропнуть)
pd.options.display.max_rows = 500

#выкачиваем данные по всем инструмента входящим в индекс МосБиржи + качаем сам индекс МосБиржи
#Достаем из каждого дата сета цены закрытия
data = yf.download(['SBER.ME','SBERP.ME','GAZP.ME','LKOH.ME','YNDX.ME','GMKN.ME','NVTK.ME','SNGS.ME','SNGSP.ME','PLZL.ME','TATN.ME','TATNP.ME','ROSN.ME','POLY.ME','MGNT.ME','MTSS.ME','FIVE.ME','MOEX.ME','IRAO.ME','NLMK.ME','ALRS.ME','CHMF.ME','VTBR.ME','RTKM.ME','PHOR.ME','TRNFP.ME','RUAL.ME','AFKS.ME','MAGN.ME','DSKY.ME','PIKK.ME','HYDR.ME','FEES.ME','QIWI.ME','AFLT.ME','CBOM.ME','LSRG.ME','RSTI.ME','UPRO.ME'],start="2011-01-01", end="2018-11-01")
imoex = yf.download(['IMOEX.ME'],start="2011-01-01", end="2018-11-01")
imoex = imoex.Close
price = data.Close

# Через сколько дней переоптимизируем портфель
shift = 60

# Rolling window (days)
rolling = 250

# Безрисковая процентная ставка
risk_free_rate = 0.00

# Заданный риск портфеля для оптимизации
TargetRisk = 0.17

# Заданная доходность портфеля для оптимизации
TargetReturn = 0.20

num_periods_annually = 252 # Количество операционных дней в году


index = price.index
p_returns = pd.DataFrame(columns=['p_returns'])
distribution = pd.DataFrame(index=price.columns)
portfolio_alloc = 0
#sharpe_ratio = pd.DataFrame(columns=['sharpe_ratio'])

# Расчитываем доходность портфеля каждый день, но ребалансировку делаем через каждые shift дней
s = 0
d0 = ''
d1 = ''
last = 0
for day in tqdm_notebook(range(rolling-1, price.index.size-1)):
    # каждые shift дней заново оптимизируем портфель
    if s == 0:
        #print('Оптимизируем. day = ', day)
        d0 = index[day-rolling+1]
        d1 = index[day]
        #         print('d0 = ', d0, 'd1 = ', d1)
        p = price.loc[d0:d1]
        opt_res = opt_max_sharpe_ratio(p, risk_free_rate)
        #opt_res = opt_targeted_risk(p, TargetRisk)
        #opt_res = opt_targeted_return(p, TargetReturn)
        if(opt_res['success']==True):
            portfolio_alloc = np.array(opt_res['x'])
            distribution[index[day].strftime('%Y-%m-%d')] = portfolio_alloc
            #distribution[index[day]] = portfolio_alloc
            #sharpe_ratio.loc[index[day]] = -1*opt_res['fun']
        else:
            print('ERROR: ', opt_res['message'])

    # Вычисление доходности портфеля
    date0 = d1
    date1 = index[day+1]
    #     if (s==0):
    #         print('date0 = ', date0, 'date1 = ', date1)
    a_return = (price.loc[date1] - price.loc[date0])/price.loc[date0]
    if (p_returns.size!=0 and s==0):
        last = p_returns.iloc[-1].values
        #last = 0
    p_returns.loc[index[day+1]] = np.sum(a_return*portfolio_alloc) + last

    # Ведем отсчет дней до новой оптимизации портфеля
    s = s + 1
    if s == shift:
        s = 0
        #print(day)



p = price.iloc[rolling-1:]
sharpe_max = opt_max_sharpe_ratio(p, risk_free_rate)
portfolio_alloc = sharpe_max['x']

index = p.index
sharpe_max_returns = pd.DataFrame(columns=['returns'])
for day in range(1, p.index.size):
    # Вычисляем доходность портфеля
    date0 = index[0]
    date1 = index[day]
    a_return = (p.loc[date1] - p.loc[date0])/p.loc[date0]
    sharpe_max_returns.loc[index[day]] = np.sum(a_return*portfolio_alloc)


# Выводит список акций и их доли в портфеле
portfolio_alloc = []
shares = price.columns[np.where(sharpe_max['x']>0.001)]
weight = sharpe_max['x'][sharpe_max['x']>0.001]

for i in range(len(shares)):
    print(shares[i], '\t', round(weight[i]*100,1))
    portfolio_alloc.append(weight[i])



plt.figure(figsize=(10,5))

gray_line = mlines.Line2D([], [], color='gray', linewidth=1, label='Доходность акций')
#plt.plot(price.iloc[rolling-1:].pct_change().cumsum(),'gray', linewidth=1, alpha=0.5)

red_line = mlines.Line2D([], [], color='red', linewidth=3, label='Доходность портфеля')
plt.plot(p_returns, 'red', linewidth=2)

blue_line = mlines.Line2D([], [], color='blue', linewidth=3, label='Индекс Московской Биржи')
plt.plot(imoex[price.iloc[rolling-1:].index[0]:].pct_change().cumsum(), 'blue', linewidth=2)

green_line = mlines.Line2D([], [], color='green', linewidth=3, label='Максимальня доходность по Шарпу')
plt.plot(sharpe_max_returns, 'green', linewidth=2)


plt.grid(True, linestyle='--')
plt.title('Доходность портфеля, доходности акций и Индекс Московской Биржи')
plt.xlabel('Дата')
plt.ylabel('Доходность')
plt.legend(handles=[gray_line, red_line, blue_line, green_line])
plt.tight_layout();


# Через сколько дней переоптимизируем портфель
shift = 60

# Rolling window (days)
rolling = 250

# Безрисковая процентная ставка
risk_free_rate = 0.00

# Заданный риск портфеля для оптимизации
TargetRisk = 0.17

# Заданная доходность портфеля для оптимизации
TargetReturn = 0.20

num_periods_annually = 252 # Количество операционных дней в году


ret_rolling = []
for rolling in tqdm_notebook(range(100, 250, 50), desc='rolling'):
    ret_shift = []
    for shift in tqdm_notebook(range(5, 250, 50), desc='shift', leave=False):
        # Расчитываем доходность портфеля каждый день, но ребалансировку делаем через каждые shift дней
        s = 0
        d0 = ''
        d1 = ''
        last = 0
        index = price.index
        p_returns = pd.DataFrame(columns=['p_returns'])
        distribution = pd.DataFrame(index=price.columns)
        portfolio_alloc = 0
        for day in tqdm_notebook(range(rolling-1, price.index.size-1), desc='day', leave=False):
            # каждые shift дней заново оптимизируем портфель
            if s == 0:
                d0 = index[day-rolling+1]
                d1 = index[day]
                p = price.loc[d0:d1]
                opt_res = opt_max_sharpe_ratio(p, risk_free_rate)
                if(opt_res['success']==True):
                    portfolio_alloc = np.array(opt_res['x'])
                    distribution[index[day].strftime('%Y-%m-%d')] = portfolio_alloc
                else:
                    #print('ERROR: ', opt_res['message'])
                    pass

            # доходность портфеля
            date0 = d1
            date1 = index[day+1]
            a_return = (price.loc[date1] - price.loc[date0])/price.loc[date0]
            if (p_returns.size!=0 and s==0):
                last = p_returns.iloc[-1].values
            p_returns.loc[index[day+1]] = np.sum(a_return*portfolio_alloc) + last

            # Ведем отсчет дней до новой оптимизации портфеля
            s = s + 1
            if s == shift:
                s = 0
        ret_shift.append(p_returns.iloc[-1].values[0])
    ret_rolling.append(ret_shift)


ret_rolling = []
for rolling in tqdm_notebook(range(10, 250, 10), desc='rolling'):
    ret_shift = []
    for shift in range(5, 250, 5):
        d0 = ''
        d1 = ''
        last = 0
        index = price.index
        p_returns = pd.DataFrame(columns=['p_returns'])
        distribution = pd.DataFrame(index=price.columns)
        portfolio_alloc = 0
        for day in range(rolling-1, price.index.size, shift):
            # каждые shift дней заново оптимизируем портфель
            # 99-100+1=0
            d0 = index[day-rolling+1]
            # 99
            d1 = index[day]
            p = price.loc[d0:d1]
            opt_res = opt_max_sharpe_ratio(p, risk_free_rate)
            if(opt_res['success']==True):
                portfolio_alloc = np.array(opt_res['x'])
                distribution[index[day].strftime('%Y-%m-%d')] = portfolio_alloc
            else:
                #print('ERROR: ', opt_res['message'])
                pass

            # Calculating portfolio return
            # 99
            date0 = d1
            # 99+5=104
            if (day+shift > price.index.size):
                date1 = index[-1]
            else:
                date1 = index[day+shift]
            a_return = (price.loc[date1] - price.loc[date0])/price.loc[date0]
            if (p_returns.size!=0):
                last = p_returns.iloc[-1].values
            p_returns.loc[date1] = np.sum(a_return*portfolio_alloc) + last


        ret_shift.append(p_returns.iloc[-1].values[0])
        #print('shift = ', shift, 'day = ', day, ' sum = ', day+shift)
        #sys.stdout.flush()
    ret_rolling.append(ret_shift)


import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode

#для вывода plotly-графиков в ноутбуке
init_notebook_mode(connected=False)


x = [i for i in range(10, 250, 10)]
y = [i for i in range(5, 250, 5)]
z = np.array(ret_rolling)

data = [
    go.Surface(
        x=x,
        y=y,
        z=z
    )
]

layout = go.Layout(
    title='Эффективные границы',
    width=800,
    height=700,
    scene=dict(
        xaxis=dict(
            title='shift',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            title='rolling',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(title='Доходность',
                   gridcolor='rgb(255, 255, 255)',
                   zerolinecolor='rgb(255, 255, 255)',
                   showbackground=True,
                   backgroundcolor='rgb(230, 230,230)'
                   )
    )
)


fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)


