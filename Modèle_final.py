#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:06:31 2023

@author: user
"""
# imports ------
from __future__ import print_function
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

# functions ------

def calculate_portfolio_return(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * len(returns)
    return portfolio_return

def calculate_portfolio_volatility(weights, returns):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns, rowvar=False) * len(returns), weights)))
    return portfolio_volatility

def negative_portfolio_return(weights, returns):
    return -calculate_portfolio_return(weights, returns)

def optimize_portfolio(returns, target_volatility):
    num_assets = returns.shape[1]
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: calculate_portfolio_volatility(x, returns) - target_volatility})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    initial_guess = np.array([1 / num_assets for i in range(num_assets)])
    results = optimize.minimize(negative_portfolio_return, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return results.x

def allocate_funds(returns, target_volatility):
    optimized_weights = optimize_portfolio(returns, target_volatility)
    portfolio_return = calculate_portfolio_return(optimized_weights, returns)
    portfolio_volatility = calculate_portfolio_volatility(optimized_weights, returns)
    return optimized_weights, portfolio_return, portfolio_volatility


def RoundTick_s(x):
    return TickSize_s*int((x/TickSize_s) + 0.5)

def RoundTick_z(x):
    return TickSize_z*int((x/TickSize_z) + 0.5)


DisplayPlots = True

# parameters ------
scale_factor=10

ndays = 20 #in MC sim
np.random.seed(201)
deltat = 1 
StartDate = dt.datetime(2020,9,15,8,0,0)
date_list = [StartDate + dt.timedelta(seconds=x * deltat) for x in range(0, int(8*3600*scale_factor))]
date_text= [x.strftime('%Y-%m-%d %H:%M:%S') for x in date_list]
n = len(date_list)
T = n
A_s = 3 #plus on augmente cette valeur moins il y a d'ordres
A_z = 3
s0 = 1.10 #EUR/USD
z0 = 30000 #BTC/USD
TickSize_s = 1e-4
TickSize_z = 1e-2
sigma_s = s0 * 0.008 / np.sqrt(T) # 0.8% vol journalière
sigma_z = z0 * 0.02 / np.sqrt(T) # 2% vol journalière
k = np.log(2.) / (2.0 * TickSize_s) #liquidité
k_z = np.log(2.) / (2.0 * TickSize_z) 
q0_s = 100000. #influence corr
q0_z = 4. #influence corr
q0_s_intra = 0.
q0_z_intra = 0.

# ordre de 1million impact de 0.01 pt base soit 0.01*0.01% donc 0.000001=1e-6 pour s environ
#et 3 pour z

theta_s = TickSize_s*1e-6 #liquidité
theta_z = TickSize_z*10 

gamma_s = 0.5 * TickSize_s / (sigma_s**2 * T * q0_s)
gamma_z = 0.5 * TickSize_z / (sigma_z**2 * T * q0_z)

nbr_orders_intraday = 100 #influence la croissance après le pic mais également descente après le pic

sec_allocate_intraday= 60 # influence spread + plus on decsend plus le coef directeur pente augmente (Volatilités plus hautes sont atteintes + intraday trades)

dollars_s_intraday = 1e5 #joue sur coef directeur pente après le pic mais trouver le juste milieu pour maintenir niveau spreads
dollars_z_intraday = 1e5

sec_momentum = 300 # réactivité montéé du pic ainsi que réactivité chute du pic
q_momentum = 1 #intensité du pic : si augmente, diminue spreads et augmente le pic et augmente "l'anomalie" , si baisse, diminue pic et augmente spread et diminue "anomalie"

vol_intra_factor=0.001

q_init_s = int(dollars_s_intraday/s0)
q_init_z = int(dollars_z_intraday/z0)

risk_free_rate = (0.01/(256*8*3600))

returns_s=list(np.random.normal(0,0.02*0.02,28800))
returns_z=list(np.random.normal(0,0.03*0.03,28800))

s = np.zeros(T*ndays+1)
z = np.zeros(T*ndays+1)
index= np.zeros(T*ndays+1)

q_s_array = np.zeros(T*ndays+1)
q_z_array = np.zeros(T*ndays+1)

trades_s_noise_array = np.zeros(T*ndays+1)
trades_z_noise_array = np.zeros(T*ndays+1)
trades_s_momentum_array = np.zeros(T*ndays+1)
trades_z_momentum_array = np.zeros(T*ndays+1)
trades_s_intra_array = np.zeros(T*ndays+1)
trades_z_intra_array = np.zeros(T*ndays+1)

delta_b_s_array = np.zeros(T*ndays+1)
delta_a_s_array = np.zeros(T*ndays+1)
delta_b_z_array = np.zeros(T*ndays+1)
delta_a_z_array = np.zeros(T*ndays+1)
delta_s_array = np.zeros(T*ndays+1)
delta_z_array = np.zeros(T*ndays+1)

dN_b_s = 0
dN_a_s =0
N_b_s = 0
N_a_s = 0

dN_b_z = 0
dN_a_z = 0
N_b_z = 0
N_a_z = 0

dN_b_z_intra = 0
dN_a_z_intra = 0
dN_b_s_intra = 0
dN_a_s_intra = 0

counter=0
q_sum_sq_s = 0.
q_sum_s = 0.
q_sum_sq_z = 0.
q_sum_z = 0.

z[0] = RoundTick_z(z0)
s[0] = RoundTick_s(s0)
index[0] = 100

# model ------
def model():
    delta_b_s = 0
    delta_a_s = 0
    delta_b_z = 0
    delta_a_z = 0

    q_s = 0
    q_z = 0
    q_s_intra = 0
    q_z_intra = 0
    counter=0
    for ind in range(ndays):
        print(ind)

        weights_intraday_s = [0,1]
        weights_intraday_z = [0,1]

        for t in range(T):

            trades_s=0
            trades_z=0

            dW_s = np.random.normal(0., 1.)
            u_b_s = np.random.uniform(0., 1.)
            u_a_s = np.random.uniform(0., 1.)

            dW_z = np.random.normal(0., 1.)
            u_b_z = np.random.uniform(0., 1.)
            u_a_z = np.random.uniform(0., 1.)

            counter+=1


            if t%(int(T/nbr_orders_intraday))==0:


                target_volatility_s = vol_intra_factor * t/scale_factor * (T- t) / (5000 * T)
                new_weights_intraday_s = allocate_funds(pd.DataFrame(np.array([np.array(returns_s[-(sec_allocate_intraday+1):-1]),np.full(len(returns_s[-(sec_allocate_intraday+1):-1]), risk_free_rate)]).T, columns=['Risky Asset', 'Risk free rate']), target_volatility_s)[0]

                target_volatility_z = vol_intra_factor * t/scale_factor * (T- t) / (5000 * T)
                new_weights_intraday_z = allocate_funds(pd.DataFrame(np.array([np.array(returns_z[-(sec_allocate_intraday+1):-1]),np.full(len(returns_z[-(sec_allocate_intraday+1):-1]), risk_free_rate)]).T, columns=['Risky Asset', 'Risk free rate']), target_volatility_z)[0]

                #FOR ASSET S

                q0_s_intra = ((dollars_s_intraday/nbr_orders_intraday)*(new_weights_intraday_s - weights_intraday_s)[0])/s[counter-1] # quantity to buy

                weights_intraday_s = new_weights_intraday_s # we actualize the actual weights

                #FOR ASSET Z
                q0_z_intra = ((dollars_z_intraday/nbr_orders_intraday)*(new_weights_intraday_z - weights_intraday_z)[0])/z[counter-1] # quantity to buy

                weights_intraday_z = new_weights_intraday_z # we actualize the actual weights

                q_s = q_s - q0_s_intra
                q_z = q_z - q0_z_intra
                trades_s= q0_s_intra
                trades_z= q0_z_intra

                q_s_intra = q0_s_intra
                q_z_intra = q0_z_intra
                
                trades_s_intra_array[counter] = q0_s_intra
                trades_z_intra_array[counter] = q0_z_intra

            #MOMENTUM TRADERS
            if (index[-1] - np.mean(index[-sec_momentum:])  > 0): 
                
                q0_s_momentum = q_momentum*q0_s 
                q0_z_momentum = q_momentum*q0_z

                q_s= q_s - q0_s_momentum
                q_z= q_z - q0_z_momentum
                trades_s= trades_s + q0_s_momentum
                trades_z= trades_z + q0_z_momentum
                
                trades_s_momentum_array[counter]=q0_s_momentum
                trades_z_momentum_array[counter]=q0_z_momentum

            else:

                q0_s_momentum = -q_momentum*q0_s 
                q0_z_momentum = -q_momentum*q0_z

                q_s = q_s - q0_s_momentum
                q_z = q_z - q0_z_momentum
                trades_s= trades_s + q0_s_momentum
                trades_z= trades_z + q0_z_momentum
                
                trades_s_momentum_array[counter]=q0_s_momentum
                trades_z_momentum_array[counter]=q0_z_momentum


            # Noise Traders
            lambda_b_s = A_s * np.exp(-k * delta_b_s)
            lambda_a_s = A_s * np.exp(-k * delta_a_s)
            lambda_b_z = A_z * np.exp(-k_z * delta_b_z)
            lambda_a_z = A_z * np.exp(-k_z * delta_a_z)

            if u_b_s < lambda_b_s * deltat:
                dN_b_s = 1. 
            else:
                dN_b_s = 0.
            if u_a_s < lambda_a_s * deltat:
                dN_a_s = 1.
            else:
                dN_a_s = 0.
            q_s =  q_s + q0_s * (dN_b_s - dN_a_s)
            trades_s = trades_s - q0_s * (dN_b_s - dN_a_s)

            if u_b_z < lambda_b_z * deltat:
                dN_b_z = 1. 
            else:
                dN_b_z = 0.
            if u_a_z < lambda_a_z * deltat:
                dN_a_z = 1.
            else:
                dN_a_z = 0.
            q_z = q_z + q0_z * (dN_b_z - dN_a_z)
            trades_z=trades_z - q0_z * (dN_b_z - dN_a_z)
            
            trades_s_noise_array[counter]= - q0_z * (dN_b_z - dN_a_z)
            trades_z_noise_array[counter]= - q0_z * (dN_b_z - dN_a_z)

            q_s_array[counter]=q_s
            q_z_array[counter]=q_z
            delta_b_s = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)+gamma_s*sigma_s*sigma_s*T*q_s,0.)) # simpifier 
            delta_a_s = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)-gamma_s*sigma_s*sigma_s*T*q_s,0.))
            delta_b_z = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)+gamma_z*sigma_z*sigma_z*T*q_z,0))
            delta_a_z = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)-gamma_z*sigma_z*sigma_z*T*q_z,0))
            
            delta_b_s_array[counter] = delta_b_s
            delta_a_s_array[counter] = delta_a_s
            delta_b_z_array[counter] = delta_b_z
            delta_a_z_array[counter] = delta_a_z
            delta_s_array[counter] = delta_b_s + delta_a_s
            delta_z_array[counter] = delta_b_z + delta_a_z
            
            
            s[counter] = RoundTick_s(s[counter-1] + theta_s*trades_s + sigma_s * dW_s)
            z[counter] = RoundTick_z(z[counter-1] + theta_z*trades_z + sigma_z * dW_z)
            index[counter]=0.5*(s[counter]/s[0])*100 + 0.5*(z[counter]/z[0])*100

            returns_s.append(np.log(s[counter]/s[counter-1]))
            returns_z.append(np.log(z[counter]/z[counter-1]))
        
    if DisplayPlots:
        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), s.T, '-')
        plt.title('s mid price')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")

        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), z.T, '-')
        plt.title('z mid price')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")

        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), (1./TickSize_s) * np.array([delta_b_s_array,delta_a_s_array,delta_s_array]).T, '-')
        plt.legend(['delta_b','delta_a','delta'])
        plt.title('spreads (in tick units) for asset s')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")

        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), (1./TickSize_s) * np.array([delta_b_z_array,delta_a_z_array,delta_z_array]).T, '-')
        plt.legend(['delta_b','delta_a','delta'])
        plt.title('spreads (in tick units) for asset z')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")

        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), 1e-3 * q_s_array, '-')
        plt.title('market-maker inventory (in K) for s')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")

        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), 1e-3 * q_z_array, '-')
        plt.title('market-maker inventory (in K) for z')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")
        
        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), trades_s_noise_array, '-')
        plt.title('noise trades for s')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")
        
        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), trades_z_noise_array, '-')
        plt.title('noise trades for z')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")
        
        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), trades_s_momentum_array, '-')
        plt.title('momentum trades for s')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")
        
        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), trades_z_momentum_array, '-')
        plt.title('momentum trades for z')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")
        
        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), trades_s_intra_array, '-')
        plt.title('Intraday/Weekly trades for s')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")
        
        plt.figure()
        ax = plt.gca()
        ax.ticklabel_format(axis='y', useOffset=False)
        plt.plot(range(ndays*T+1), trades_z_intra_array, '-')
        plt.title('Intraday/Weekly trades for z')
        plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
        plt.xlabel("Day/Week")

# correlations functions ------

import math
def display_corr(tau_max=5000,s=s,z=z):
    
    def artanh(x):
        return 0.5*math.log((1+x)/(1-x))
    
    def tanh(x):
        return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
       
    
    x=np.array(s)
    y=np.array(z)
    
    BtcEuroverlapping=[]
    
    low=[]
    high=[] 
    tenors=list(range(1,tau_max))
    for tau in tenors:
        print(tau)
        ret_x=np.divide(x[tau:]-x[:-tau],x[-tau])
        ret_y=np.divide(y[tau:]-y[:-tau],y[-tau])
        rho_mat=np.corrcoef(ret_x,ret_y)
        BtcEuroverlapping.append(rho_mat[0,1])
        r=rho_mat[0,1]
        SE=1/np.sqrt(int(len(ret_x)/tau)-3)
        low.append(tanh(artanh(r)-1.96*SE) )
        high.append(tanh(artanh(r)+1.96*SE) )
    
    plt.plot(BtcEuroverlapping)
    plt.title("Overlapping correlation of Eurusd and Btc log returns")
    plt.xlabel("Tau : Increments of 1s")
    plt.ylabel("Pearson Correlation")
    plt.plot(high, label='Low', c='g')
    plt.plot(low, label='High', c='r')
    plt.legend()
    
def display_corr_min(tau_max=20000,s=s[::60],z=z[::60], IC=True):
    
    def artanh(x):
        return 0.5*math.log((1+x)/(1-x))
    
    def tanh(x):
        return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
       
    
    x=np.array(s)
    y=np.array(z)
    
    BtcEuroverlapping=[]
    
    low=[]
    high=[] 
    tenors=list(range(1,tau_max))
    for tau in tenors:
        print(tau)
        ret_x=np.divide(x[tau:]-x[:-tau],x[-tau])
        ret_y=np.divide(y[tau:]-y[:-tau],y[-tau])
        rho_mat=np.corrcoef(ret_x,ret_y)
        BtcEuroverlapping.append(rho_mat[0,1])
        r=rho_mat[0,1]
        SE=1/np.sqrt(int(len(ret_x)/tau)-3)
        low.append(tanh(artanh(r)-1.96*SE) )
        high.append(tanh(artanh(r)+1.96*SE) )
    
    if IC==True:
        
        plt.plot(BtcEuroverlapping)
        plt.title("Overlapping correlation of Eurusd and Btc log returns")
        plt.xlabel("Tau : Increments of 1m")
        plt.ylabel("Pearson Correlation")
        plt.plot(high, label='Low', c='g')
        plt.plot(low, label='High', c='r')
        plt.legend()
    else:
        plt.plot(BtcEuroverlapping)
        plt.title("Overlapping correlation of Eurusd and Btc log returns")
        plt.xlabel("Tau : Increments of 1m")
        plt.ylabel("Pearson Correlation")
        plt.legend()

if __name__ == '__main__':
    model()
    display_corr()
    display_corr_min()
    display_corr_min(20000)
    display_corr_min(20000, IC=False)
