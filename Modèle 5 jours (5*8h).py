from __future__ import print_function
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd


DisplayPlots = False

# parameters ------
ndays = 100 #in MC sim
np.random.seed(201)
deltat = 1 # seconds
StartDate = dt.datetime(2020,9,15,8,0,0)
date_list = [StartDate + dt.timedelta(seconds=x * deltat) for x in range(0, int(3600*120))]
date_text= [x.strftime('%Y-%m-%d %H:%M:%S') for x in date_list]
n = len(date_list)
T = n
A_s = 2 #plus on augmente cette valeur moins il y a d'ordres
A_z = 2
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

dollars_s_intraday = 9e10 #joue sur coef directeur pente après le pic mais trouver le juste milieu pour maintenir niveau spreads
dollars_z_intraday = 9e10

sec_momentum = 300 # réactivité montéé du pic ainsi que réactivité chute du pic
q_momentum = 2.5 #intensité du pic : si augmente, diminue spreads et augmente le pic et augmente "l'anomalie" , si baisse, diminue pic et augmente spread et diminue "anomalie"

q_init_s = int(dollars_s_intraday/s0)
q_init_z = int(dollars_z_intraday/z0)

risk_free_rate = (0.01/(256*8*3600))

nbr_orders_markow = 100
sec_markow = 28800
dollars_markow = 9e9 #8e9
target_volatility_markow= 0.005

returns_index=list(np.random.normal(0,0.025*0.025,28800)) #minute returns
returns_s=list(np.random.normal(0,0.02*0.02,28800))
returns_z=list(np.random.normal(0,0.03*0.03,28800))

s_min=[s0]
z_min=[z0]

q_s = np.zeros(T*ndays+1)
s = np.zeros(T*ndays+1)
q_z = np.zeros(T*ndays+1)
z = np.zeros(T*ndays+1)
index= np.zeros(T*ndays+1)

delta_b_s = np.zeros(T*ndays+1)
delta_a_s = np.zeros(T*ndays+1)
delta_b_z = np.zeros(T*ndays+1)
delta_a_z = np.zeros(T*ndays+1)

dN_b_s = np.zeros(T*ndays+1)
dN_a_s = np.zeros(T*ndays+1)
N_b_s = np.zeros(T*ndays+1)
N_a_s = np.zeros(T*ndays+1)

dN_b_z = np.zeros(T*ndays+1)
dN_a_z = np.zeros(T*ndays+1)
N_b_z = np.zeros(T*ndays+1)
N_a_z = np.zeros(T*ndays+1)

dN_b_z_intra = np.zeros(T*ndays+1)
dN_a_z_intra = np.zeros(T*ndays+1)
dN_b_s_intra = np.zeros(T*ndays+1)
dN_a_s_intra = np.zeros(T*ndays+1)

dN_b_s_markow=np.zeros(T*ndays+1)
dN_a_s_markow=np.zeros(T*ndays+1)
dN_b_z_markow=np.zeros(T*ndays+1)
dN_a_z_markow=np.zeros(T*ndays+1)

q_z_markow = np.zeros(T*ndays+1)
q_s_markow = np.zeros(T*ndays+1)
q_z_intra = np.zeros(T*ndays+1)
q_s_intra = np.zeros(T*ndays+1)
q_z_momentum = np.zeros(T*ndays+1)
q_s_momentum = np.zeros(T*ndays+1)

new_weights_intraday_s_list=[]
new_weights_intraday_z_list=[]

total_traded_noise_s = []
total_traded_intraday_s = []
total_traded_momentum_s = []
total_traded_noise_z = []
total_traded_intraday_z = []
total_traded_momentum_z = []
total_traded_daily_s = []
total_traded_daily_z = []



# ---------------------------


import numpy as np


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

def model(state):
    print("**** SIMULATED MARKET-MAKING -- AVELLANEDA-STOIKOV *********")
    print("Start simulation %s, end %s, deltat=%d sec, n=%d"%(date_text[0],date_text[-1],deltat,n))
    print("Av pnl,std,std pos for s,std pos for z,Sharpe")

    AllPnls = np.zeros(ndays)

    counter=0
    q_sum_sq_s = 0.
    q_sum_s = 0.
    q_sum_sq_z = 0.
    q_sum_z = 0.
    
    z[0] = RoundTick_z(z0)
    s[0] = RoundTick_s(s0)
    index[0] = 100
    
    weights_markow_0= [0.5,0.5]
    
    if state ==0 :  # Simple automated market making with noise traders 
        for ind in range(ndays):
            print(ind)
            
            dW_s = np.array(np.random.normal(0., 1., T))
            u_b_s = np.random.uniform(0., 1., T)
            u_a_s = np.random.uniform(0., 1., T)
                
            dW_z = np.array(np.random.normal(0., 1., T))
            u_b_z = np.random.uniform(0., 1., T)
            u_a_z = np.random.uniform(0., 1., T)
             
            for t in range(T):
                counter+=1            
                # Noise Traders
                lambda_b_s = A_s * np.exp(-k * delta_b_s[counter - 1])
                lambda_a_s = A_s * np.exp(-k * delta_a_s[counter - 1])
                lambda_b_z = A_z * np.exp(-k_z * delta_b_z[counter - 1])
                lambda_a_z = A_z * np.exp(-k_z * delta_a_z[counter - 1])
                
                if u_b_s[t] < lambda_b_s * deltat:
                    dN_b_s[counter] = 1. 
                else:
                    dN_b_s[counter] = 0.
                if u_a_s[t] < lambda_a_s * deltat:
                    dN_a_s[counter] = 1.
                else:
                    dN_a_s[counter] = 0.
                N_b_s[counter] = N_b_s[counter-1] + dN_b_s[counter]
                N_a_s[counter] = N_a_s[counter-1] + dN_a_s[counter]
                q_s[counter] =  q0_s * (N_b_s[counter] - N_a_s[counter])
                
                if u_b_z[t] < lambda_b_z * deltat:
                    dN_b_z[counter] = 1. 
                else:
                    dN_b_z[counter] = 0.
                if u_a_z[t] < lambda_a_z * deltat:
                    dN_a_z[counter] = 1.
                else:
                    dN_a_z[counter] = 0.
                N_b_z[counter] = N_b_z[counter-1] + dN_b_z[counter]
                N_a_z[counter] = N_a_z[counter-1] + dN_a_z[counter]
                q_z[counter] = q0_z * (N_b_z[counter] - N_a_z[counter])
                

                  # Avellaneda-Stoikov model
                delta_b_s[counter] = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)+gamma_s*sigma_s*sigma_s*T*q_s[counter],0.)) # simpifier 
                delta_a_s[counter] = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)-gamma_s*sigma_s*sigma_s*T*q_s[counter],0.))
                delta_b_z[counter] = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)+gamma_z*sigma_z*sigma_z*T*q_z[counter],0))
                delta_a_z[counter] = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)-gamma_z*sigma_z*sigma_z*T*q_z[counter],0))


                s[counter] = RoundTick_s(s[counter-1] + theta_s*q_s[counter] + sigma_s * dW_s[t])
                z[counter] = RoundTick_z(z[counter-1] + theta_z*q_z[counter] + sigma_z * dW_z[t])
                #s[counter] = RoundTick_s(s[counter-1]  + sigma_s * dW_s[t])
                #z[counter] = RoundTick_z(z[counter-1]  + sigma_z * dW_z[t])
                #print("theta",theta_s*q_s[counter])
                #print("sigma",sigma_s * dW_s[t])
                
            


        delta_b_s[0] = delta_b_s[1]
        delta_a_s[0] = delta_a_s[1]
        delta_b_z[0] = delta_b_z[1]
        delta_a_z[0] = delta_a_z[1]
        
        p_b_s = s - delta_b_s
        p_a_s = s + delta_a_s
        delta_s = delta_a_s + delta_b_s
        
        p_b_z = z - delta_b_z
        p_a_z = z + delta_a_z
        delta_z = delta_a_z + delta_b_z

        dx_s = q0_s * (np.multiply(p_a_s,dN_a_s) - np.multiply(p_b_s,dN_b_s))
        x_s = np.cumsum(dx_s)
        dx_z = q0_z * (np.multiply(p_a_z,dN_a_z) - np.multiply(p_b_z,dN_b_z))
        x_z = np.cumsum(dx_z)
        pnl = x_s + np.multiply(q_s,s) + x_z + np.multiply(q_z,z)

        AllPnls[ind] = pnl[-1]
        q_sum_sq_s += np.sum(np.multiply(q_s,q_s))
        q_sum_s += np.sum(q_s)
        q_var_s = (1./(ndays * n)) * q_sum_sq_s - ((1./(ndays * n))  * q_sum_s) * ((1./(ndays * n))  * q_sum_s)
        q_sum_sq_z += np.sum(np.multiply(q_z,q_z))
        q_sum_z += np.sum(q_z)
        q_var_z = (1./(ndays * n)) * q_sum_sq_z - ((1./(ndays * n))  * q_sum_z) * ((1./(ndays * n))  * q_sum_z)

        print("%1.0f, %1.0f, %1.0f,%1.0f"
              %(np.mean(AllPnls),np.std(AllPnls),np.sqrt(q_var_s),np.sqrt(q_var_z)))

        if DisplayPlots:
             print("Displaying plots -----------------------------")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(T*ndays+1), np.array([p_b_s]).T, '-')
             plt.plot(range(ndays*T+1), np.array([s]).T, '-')
             plt.plot(range(ndays*T+1), np.array([p_a_s]).T, '-')
             plt.title('asset price: bid/mid/ask for asset s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), (1./TickSize_s) * np.array([delta_b_s,delta_a_s,delta_s]).T, '-')
             plt.legend(['delta_b','delta_a','delta'])
             plt.title('spreads (in tick units) for asset s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), 1e-3 * q_s, '-')
             plt.title('market-maker inventory (in K) for s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
             
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), np.array([p_b_z]).T, '-')
             plt.plot(range(ndays*T+1), np.array([z]).T, '-')
             plt.plot(range(ndays*T+1), np.array([p_a_z]).T, '-')
             plt.title('asset price: bid/mid/ask for asset z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), (1./TickSize_z) * np.array([delta_b_z,delta_a_z,delta_z]).T, '-')
             plt.legend(['delta_b','delta_a','delta'])
             plt.title('spreads (in tick units) for asset z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), 1e-3 * q_z, '-')
             plt.title('market-maker inventory (in K) for z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), pnl, '-')
             plt.title('market-maker pnl (in USD)')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
            
             plt.show()
    elif state==1:
        for ind in range(ndays):
            print(ind)
            
            dW_s = np.array(np.random.normal(0., 1., T))
            u_b_s = np.random.uniform(0., 1., T)
            u_a_s = np.random.uniform(0., 1., T)
                
            dW_z = np.array(np.random.normal(0., 1., T))
            u_b_z = np.random.uniform(0., 1., T)
            u_a_z = np.random.uniform(0., 1., T)
             
            for t in range(T):
                counter+=1    
                
                #Momentum traders 
                
                if (sum(returns_index[-sec_momentum::]) / sec_momentum) > (returns_index[-1]):
                    
                    q0_s_momentum = q_momentum*q0_s 
                    q0_z_momentum = q_momentum*q0_z
                    
                    q_s_momentum[counter]=(q0_s_momentum)
                    q_z_momentum[counter]=(q0_z_momentum)
        
                    q_s[counter]= q_s[counter] + q0_s_momentum
                    q_z[counter]= q_z[counter] + q0_z_momentum
                    
                elif (sum(returns_index[-sec_momentum::]) / sec_momentum)< (returns_index[-1]):
                    
                    q0_s_momentum = -q_momentum*q0_s 
                    q0_z_momentum = -q_momentum*q0_z
                    
                    q_s_momentum[counter]=(q0_s_momentum)
                    q_z_momentum[counter]=(q0_z_momentum)
        
                    q_s[counter]= q_s[counter] + q0_s_momentum
                    q_z[counter]= q_z[counter] + q0_z_momentum
                    
                # Noise Traders
                lambda_b_s = A_s * np.exp(-k * delta_b_s[counter - 1])
                lambda_a_s = A_s * np.exp(-k * delta_a_s[counter - 1])
                lambda_b_z = A_z * np.exp(-k_z * delta_b_z[counter - 1])
                lambda_a_z = A_z * np.exp(-k_z * delta_a_z[counter - 1])
                
                if u_b_s[t] < lambda_b_s * deltat:
                    dN_b_s[counter] = 1. 
                else:
                    dN_b_s[counter] = 0.
                if u_a_s[t] < lambda_a_s * deltat:
                    dN_a_s[counter] = 1.
                else:
                    dN_a_s[counter] = 0.
                N_b_s[counter] = N_b_s[counter-1] + dN_b_s[counter]
                N_a_s[counter] = N_a_s[counter-1] + dN_a_s[counter]
                q_s[counter] =  q_s[counter] + q0_s * (N_b_s[counter] - N_a_s[counter])
                
                if u_b_z[t] < lambda_b_z * deltat:
                    dN_b_z[counter] = 1. 
                else:
                    dN_b_z[counter] = 0.
                if u_a_z[t] < lambda_a_z * deltat:
                    dN_a_z[counter] = 1.
                else:
                    dN_a_z[counter] = 0.
                N_b_z[counter] = N_b_z[counter-1] + dN_b_z[counter]
                N_a_z[counter] = N_a_z[counter-1] + dN_a_z[counter]
                q_z[counter] = q_z[counter] + q0_z * (N_b_z[counter] - N_a_z[counter])
                

                  # Avellaneda-Stoikov model
                delta_b_s[counter] = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)+gamma_s*sigma_s*sigma_s*T*q_s[counter],0.)) # simpifier 
                delta_a_s[counter] = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)-gamma_s*sigma_s*sigma_s*T*q_s[counter],0.))
                delta_b_z[counter] = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)+gamma_z*sigma_z*sigma_z*T*q_z[counter],0))
                delta_a_z[counter] = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)-gamma_z*sigma_z*sigma_z*T*q_z[counter],0))


                s[counter] = RoundTick_s(s[counter-1] + theta_s*q_s[counter] + sigma_s * dW_s[t])
                z[counter] = RoundTick_z(z[counter-1] + theta_z*q_z[counter] + sigma_z * dW_z[t])
                index[counter]=0.5*(s[counter]/s[0])*100 + 0.5*(z[counter]/z[0])*100
                
                returns_s.append(np.log(s[counter]/s[counter-1]))
                returns_z.append(np.log(z[counter]/z[counter-1]))
                returns_index.append(np.log(index[counter]/index[counter-1]))
                
                if t%60==0:
                    s_min.append(s[counter])
                    z_min.append(z[counter])
                
            


        delta_b_s[0] = delta_b_s[1]
        delta_a_s[0] = delta_a_s[1]
        delta_b_z[0] = delta_b_z[1]
        delta_a_z[0] = delta_a_z[1]
        
        p_b_s = s - delta_b_s
        p_a_s = s + delta_a_s
        delta_s = delta_a_s + delta_b_s
        
        p_b_z = z - delta_b_z
        p_a_z = z + delta_a_z
        delta_z = delta_a_z + delta_b_z

        dx_s = q0_s * (np.multiply(p_a_s,dN_a_s) - np.multiply(p_b_s,dN_b_s))
        x_s = np.cumsum(dx_s)
        dx_z = q0_z * (np.multiply(p_a_z,dN_a_z) - np.multiply(p_b_z,dN_b_z))
        x_z = np.cumsum(dx_z)
        pnl = x_s + np.multiply(q_s,s) + x_z + np.multiply(q_z,z)

        AllPnls[ind] = pnl[-1]
        q_sum_sq_s += np.sum(np.multiply(q_s,q_s))
        q_sum_s += np.sum(q_s)
        q_var_s = (1./(ndays * n)) * q_sum_sq_s - ((1./(ndays * n))  * q_sum_s) * ((1./(ndays * n))  * q_sum_s)
        q_sum_sq_z += np.sum(np.multiply(q_z,q_z))
        q_sum_z += np.sum(q_z)
        q_var_z = (1./(ndays * n)) * q_sum_sq_z - ((1./(ndays * n))  * q_sum_z) * ((1./(ndays * n))  * q_sum_z)

        print("%1.0f, %1.0f, %1.0f,%1.0f"
              %(np.mean(AllPnls),np.std(AllPnls),np.sqrt(q_var_s),np.sqrt(q_var_z)))

        if DisplayPlots:
             print("Displaying plots -----------------------------")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(T*ndays+1), np.array([p_b_s]).T, '-')
             plt.plot(range(ndays*T+1), np.array([s]).T, '-')
             plt.plot(range(ndays*T+1), np.array([p_a_s]).T, '-')
             plt.title('asset price: bid/mid/ask for asset s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), (1./TickSize_s) * np.array([delta_b_s,delta_a_s,delta_s]).T, '-')
             plt.legend(['delta_b','delta_a','delta'])
             plt.title('spreads (in tick units) for asset s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), 1e-3 * q_s, '-')
             plt.title('market-maker inventory (in K) for s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
             
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), np.array([p_b_z]).T, '-')
             plt.plot(range(ndays*T+1), np.array([z]).T, '-')
             plt.plot(range(ndays*T+1), np.array([p_a_z]).T, '-')
             plt.title('asset price: bid/mid/ask for asset z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), (1./TickSize_z) * np.array([delta_b_z,delta_a_z,delta_z]).T, '-')
             plt.legend(['delta_b','delta_a','delta'])
             plt.title('spreads (in tick units) for asset z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), 1e-3 * q_z, '-')
             plt.title('market-maker inventory (in K) for z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
             
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), q_s_momentum, '-')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
             plt.title('intraday traded quantity for s')
             
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), q_z_momentum, '-')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
             plt.title('intraday traded quantity for z')
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), pnl, '-')
             plt.title('market-maker pnl (in USD)')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
            
             plt.show()
                
                
        
    elif state==2:
        for ind in range(ndays):
            print(ind)
            
            dW_s = np.array(np.random.normal(0., 1., T))
            u_b_s = np.random.uniform(0., 1., T)
            u_a_s = np.random.uniform(0., 1., T)
                
            dW_z = np.array(np.random.normal(0., 1., T))
            u_b_z = np.random.uniform(0., 1., T)
            u_a_z = np.random.uniform(0., 1., T)
         
            weights_intraday_s = [0,1]
            weights_intraday_z = [0,1]
             
            for t in range(T):
                counter+=1
                
                
                if t%(int(T/nbr_orders_intraday))==0:
            
                   
                    target_volatility_s = 0.000005 * t * (T- t) / (5000 * T) #0.000008 * t * (T- t) / (5000 * T)
                    new_weights_intraday_s = allocate_funds(pd.DataFrame(np.array([np.array(returns_s[-(sec_allocate_intraday+1):-1]),np.full(len(returns_s[-(sec_allocate_intraday+1):-1]), risk_free_rate)]).T, columns=['Risky Asset', 'Risk free rate']), target_volatility_s)[0]
                    new_weights_intraday_s_list.append(new_weights_intraday_s[0])
                    
                    target_volatility_z = 0.000005 * t * (T- t) / (5000 * T) #0.000008 * t * (T- t) / (5000 * T)
                    new_weights_intraday_z = allocate_funds(pd.DataFrame(np.array([np.array(returns_z[-(sec_allocate_intraday+1):-1]),np.full(len(returns_z[-(sec_allocate_intraday+1):-1]), risk_free_rate)]).T, columns=['Risky Asset', 'Risk free rate']), target_volatility_z)[0]
                    new_weights_intraday_z_list.append(new_weights_intraday_z[0])
                    
                    #FOR ASSET S

                    q0_s_intra = int(((dollars_s_intraday/nbr_orders_intraday)*(new_weights_intraday_s - weights_intraday_s)[0])/s[counter-1]) # quantity to buy

                    weights_intraday_s = new_weights_intraday_s # we actualize the actual weights
            
                    #FOR ASSET Z
                    q0_z_intra = ((dollars_z_intraday/nbr_orders_intraday)*(new_weights_intraday_z - weights_intraday_z)[0])/z[counter-1] # quantity to buy
                        
                    weights_intraday_z = new_weights_intraday_z # we actualize the actual weights

                    q_s[counter] = q0_s_intra
                    q_z[counter] = q0_z_intra
                    q_s_intra[counter] = q0_s_intra
                    q_z_intra[counter] = q0_z_intra
                    total_traded_intraday_s.append(abs(q0_s_intra*s[counter-1]))
                    total_traded_intraday_z.append(abs(q0_z_intra*z[counter-1]))
                
                #MOMENTUM TRADERS
                if (sum(returns_index[-sec_momentum::]) / sec_momentum) > (returns_index[-1]):
                    
                    q0_s_momentum = q_momentum*q0_s 
                    q0_z_momentum = q_momentum*q0_z
                    
                    q_s_momentum[counter]=(q0_s_momentum)
                    q_z_momentum[counter]=(q0_z_momentum)
        
                    q_s[counter]= q_s[counter] + q0_s_momentum
                    q_z[counter]= q_z[counter] + q0_z_momentum
                    
                    total_traded_momentum_s.append(abs(q0_s_momentum*s[counter-1]))
                    total_traded_momentum_z.append(abs(q0_z_momentum*z[counter-1]))
                    
                elif (sum(returns_index[-sec_momentum::]) / sec_momentum)< (returns_index[-1]):
                    
                    q0_s_momentum = -q_momentum*q0_s 
                    q0_z_momentum = -q_momentum*q0_z
                    
                    q_s_momentum[counter]=(q0_s_momentum)
                    q_z_momentum[counter]=(q0_z_momentum)
        
                    q_s[counter]= q_s[counter] + q0_s_momentum
                    q_z[counter]= q_z[counter] + q0_z_momentum
                    
                    total_traded_momentum_s.append(abs(q0_s_momentum*s[counter-1]))
                    total_traded_momentum_z.append(abs(q0_z_momentum*z[counter-1]))
                    
        
                    
                # Noise Traders
                lambda_b_s = A_s * np.exp(-k * delta_b_s[counter - 1])
                lambda_a_s = A_s * np.exp(-k * delta_a_s[counter - 1])
                lambda_b_z = A_z * np.exp(-k_z * delta_b_z[counter - 1])
                lambda_a_z = A_z * np.exp(-k_z * delta_a_z[counter - 1])
                
                if u_b_s[t] < lambda_b_s * deltat:
                    dN_b_s[counter] = 1. 
                else:
                    dN_b_s[counter] = 0.
                if u_a_s[t] < lambda_a_s * deltat:
                    dN_a_s[counter] = 1.
                else:
                    dN_a_s[counter] = 0.
                N_b_s[counter] = N_b_s[counter-1] + dN_b_s[counter]
                N_a_s[counter] = N_a_s[counter-1] + dN_a_s[counter]
                q_s[counter] =  q_s[counter] + q0_s * (N_b_s[counter] - N_a_s[counter])
                
                if u_b_z[t] < lambda_b_z * deltat:
                    dN_b_z[counter] = 1. 
                else:
                    dN_b_z[counter] = 0.
                if u_a_z[t] < lambda_a_z * deltat:
                    dN_a_z[counter] = 1.
                else:
                    dN_a_z[counter] = 0.
                N_b_z[counter] = N_b_z[counter-1] + dN_b_z[counter]
                N_a_z[counter] = N_a_z[counter-1] + dN_a_z[counter]
                q_z[counter] = q_z[counter] + q0_z * (N_b_z[counter] - N_a_z[counter])
                
                total_traded_noise_s.append(abs(q0_s * (N_b_s[counter] - N_a_s[counter]) * s[counter-1]))
                total_traded_noise_z.append(abs(q0_z * (N_b_z[counter] - N_a_z[counter]) * z[counter-1]))
                

                  # Avellaneda-Stoikov model
                delta_b_s[counter] = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)+gamma_s*sigma_s*sigma_s*T*q_s[counter],0.)) # simpifier 
                delta_a_s[counter] = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)-gamma_s*sigma_s*sigma_s*T*q_s[counter],0.))
                delta_b_z[counter] = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)+gamma_z*sigma_z*sigma_z*T*q_z[counter],0))
                delta_a_z[counter] = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)-gamma_z*sigma_z*sigma_z*T*q_z[counter],0))


                s[counter] = RoundTick_s(s[counter-1] + theta_s*q_s[counter] + sigma_s * dW_s[t])
                z[counter] = RoundTick_z(z[counter-1] + theta_z*q_z[counter] + sigma_z * dW_z[t])
                index[counter]=0.5*(s[counter]/s[0])*100 + 0.5*(z[counter]/z[0])*100
                
                returns_s.append(np.log(s[counter]/s[counter-1]))
                returns_z.append(np.log(z[counter]/z[counter-1]))
                returns_index.append(np.log(index[counter]/index[counter-1]))
                
                if t%60==0:
                    s_min.append(s[counter])
                    z_min.append(z[counter])
                
            


        delta_b_s[0] = delta_b_s[1]
        delta_a_s[0] = delta_a_s[1]
        delta_b_z[0] = delta_b_z[1]
        delta_a_z[0] = delta_a_z[1]
        
        p_b_s = s - delta_b_s
        p_a_s = s + delta_a_s
        delta_s = delta_a_s + delta_b_s
        
        p_b_z = z - delta_b_z
        p_a_z = z + delta_a_z
        delta_z = delta_a_z + delta_b_z

        dx_s = q0_s * (np.multiply(p_a_s,dN_a_s) - np.multiply(p_b_s,dN_b_s))
        x_s = np.cumsum(dx_s)
        dx_z = q0_z * (np.multiply(p_a_z,dN_a_z) - np.multiply(p_b_z,dN_b_z))
        x_z = np.cumsum(dx_z)
        pnl = x_s + np.multiply(q_s,s) + x_z + np.multiply(q_z,z)

        AllPnls[ind] = pnl[-1]
        q_sum_sq_s += np.sum(np.multiply(q_s,q_s))
        q_sum_s += np.sum(q_s)
        q_var_s = (1./(ndays * n)) * q_sum_sq_s - ((1./(ndays * n))  * q_sum_s) * ((1./(ndays * n))  * q_sum_s)
        q_sum_sq_z += np.sum(np.multiply(q_z,q_z))
        q_sum_z += np.sum(q_z)
        q_var_z = (1./(ndays * n)) * q_sum_sq_z - ((1./(ndays * n))  * q_sum_z) * ((1./(ndays * n))  * q_sum_z)
        
        
        sum_total_traded_noise_s=np.sum(total_traded_noise_s)
        sum_total_traded_intraday_s=np.sum(total_traded_intraday_s)
        sum_total_traded_momentum_s=np.sum(total_traded_momentum_s)
        sum_total_traded_noise_z=np.sum(total_traded_noise_z)
        sum_total_traded_intraday_z=np.sum(total_traded_intraday_z)
        sum_total_traded_momentum_z=np.sum(total_traded_momentum_z)
        print("\n")
        print("Proportion for s noise trades / (noise trades + intraday trades + momentum trades)", sum_total_traded_noise_s / (sum_total_traded_noise_s+sum_total_traded_intraday_s+sum_total_traded_momentum_s))
        print("Proportion for z noise trades / (noise trades + intraday trades + momentum trades)", sum_total_traded_noise_z / (sum_total_traded_noise_z+sum_total_traded_intraday_z+sum_total_traded_momentum_z))
        print("\n")
        print("Proportion for s momentum trades / (intraday trades + momentum trades)", sum_total_traded_momentum_s / (sum_total_traded_intraday_s+sum_total_traded_momentum_s))
        print("Proportion for z momentum trades / (intraday trades + momentum trades)", sum_total_traded_momentum_z / (sum_total_traded_intraday_z+sum_total_traded_momentum_z))
        print("\n")
        print("Ratio noise traders s/z", sum_total_traded_noise_s / sum_total_traded_noise_z)
        print("Ratio momentum traders s/z", sum_total_traded_momentum_s / sum_total_traded_momentum_z)
        print("Ratio intraday traders s/z", sum_total_traded_intraday_s / sum_total_traded_intraday_z)
        print("\n")
        print("Spread moyen S tick unit", (1./TickSize_s)*np.mean(delta_s))
        print("Spread moyen Z tick unit", (1./TickSize_z)*np.mean(delta_z))
        print("Spread max S tick unit", (1./TickSize_s)*np.max(delta_s))
        print("Spread max Z tick unit", (1./TickSize_z)*np.max(delta_z))
        print("\n")
        print("Spread moyen S", np.mean(delta_s))
        print("Spread moyen Z", np.mean(delta_z))
        print("Spread max S ",  np.max(delta_s))
        print("Spread max Z tick unit",np.max(delta_z))
        print("\n")
        print("Quantité moyenne S tradée", np.mean(np.abs(q_s)))
        print("Quantité moyenne Z tradée", np.mean(np.abs(q_z)))
        print("Quantité max S tradée", np.max(np.abs(q_s)))
        print("Quantité max Z tradée", np.max(np.abs(q_z)))
        print("\n")
        

        if DisplayPlots:
             print("Displaying plots -----------------------------")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(T*ndays+1), np.array([p_b_s]).T, '-')
             plt.plot(range(ndays*T+1), np.array([s]).T, '-')
             plt.plot(range(ndays*T+1), np.array([p_a_s]).T, '-')
             plt.title('asset price: bid/mid/ask for asset s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), (1./TickSize_s) * np.array([delta_b_s,delta_a_s,delta_s]).T, '-')
             plt.legend(['delta_b','delta_a','delta'])
             plt.title('spreads (in tick units) for asset s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), 1e-3 * q_s, '-')
             plt.title('market-maker inventory (in K) for s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
             
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), np.array([p_b_z]).T, '-')
             plt.plot(range(ndays*T+1), np.array([z]).T, '-')
             plt.plot(range(ndays*T+1), np.array([p_a_z]).T, '-')
             plt.title('asset price: bid/mid/ask for asset z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), (1./TickSize_z) * np.array([delta_b_z,delta_a_z,delta_z]).T, '-')
             plt.legend(['delta_b','delta_a','delta'])
             plt.title('spreads (in tick units) for asset z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), 1e-3 * q_z, '-')
             plt.title('market-maker inventory (in K) for z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), pnl, '-')
             plt.title('market-maker pnl (in USD)')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
            
             plt.show()
                
    elif state==3:
        for ind in range(ndays):
            print(ind)
    
            #weights_markow = allocate_funds(pd.DataFrame(np.array([np.array(returns_s[-28801:-1]),returns_z[-28801:-1],np.full(len(returns_s[-28801:-1]), risk_free_rate)]).T, columns=['Asset s','Asset z', 'Risk free rate']), target_volatility)[0]
            #with index
            weights_markow = allocate_funds(pd.DataFrame(np.array([returns_index[-sec_markow:-1],np.full(len(returns_s[-sec_markow:-1]), risk_free_rate)]).T, columns=['Index', 'Risk free rate']), target_volatility_markow)[0]
            
            #using historical returns to simplify
            print(weights_markow)
            dW_s = np.array(np.random.normal(0., 1., T))
            u_b_s = np.random.uniform(0., 1., T)
            u_a_s = np.random.uniform(0., 1., T)

                
            dW_z = np.array(np.random.normal(0., 1., T))
            u_b_z = np.random.uniform(0., 1., T)
            u_a_z = np.random.uniform(0., 1., T)
            
                        
            weights_intraday_s = [0,1]
            weights_intraday_z = [0,1]
             
            for t in range(T):
                counter+=1
                                
                if t%(int(T/nbr_orders_markow))==0:
                    q0_s_markow = int(((weights_markow_0-weights_markow)[0]/2*dollars_markow/s[counter-1])/nbr_orders_markow) # rebalancing orders in quantity
                    q0_z_markow = ((weights_markow_0-weights_markow)[0]/2*dollars_markow/z[counter-1])/nbr_orders_markow
        
                    q_s[counter]= q_s[counter] + q0_s_markow
                    q_z[counter]= q_z[counter] + q0_z_markow
                    q_s_markow[counter]= q0_s_markow
                    q_z_markow[counter]= q0_z_markow
                    total_traded_daily_s.append(abs(q0_s_markow*s[counter-1]))
                    total_traded_daily_z.append(abs(q0_z_markow*z[counter-1]))

                if t%(int(T/nbr_orders_intraday))==0:
            
                   
                    target_volatility_s = 0.000005 * t * (T- t) / (5000 * T)
                    new_weights_intraday_s = allocate_funds(pd.DataFrame(np.array([np.array(returns_s[-(sec_allocate_intraday+1):-1]),np.full(len(returns_s[-(sec_allocate_intraday+1):-1]), risk_free_rate)]).T, columns=['Risky Asset', 'Risk free rate']), target_volatility_s)[0]
                    new_weights_intraday_s_list.append(new_weights_intraday_s[0])
                    
                    target_volatility_z = 0.000005 * t * (T- t) / (5000 * T)
                    new_weights_intraday_z = allocate_funds(pd.DataFrame(np.array([np.array(returns_z[-(sec_allocate_intraday+1):-1]),np.full(len(returns_z[-(sec_allocate_intraday+1):-1]), risk_free_rate)]).T, columns=['Risky Asset', 'Risk free rate']), target_volatility_z)[0]
                    new_weights_intraday_z_list.append(new_weights_intraday_z[0])
                    
                    #FOR ASSET S

                    q0_s_intra = int(((dollars_s_intraday/nbr_orders_intraday)*(new_weights_intraday_s - weights_intraday_s)[0])/s[counter-1]) # quantity to buy

                    weights_intraday_s = new_weights_intraday_s # we actualize the actual weights
            
                    #FOR ASSET Z
                    q0_z_intra = ((dollars_z_intraday/nbr_orders_intraday)*(new_weights_intraday_z - weights_intraday_z)[0])/z[counter-1] # quantity to buy
                        
                    weights_intraday_z = new_weights_intraday_z # we actualize the actual weights

                    q_s[counter] =  q_s[counter] + q0_s_intra
                    q_z[counter] =  q_z[counter] + q0_z_intra
                    q_s_intra[counter] = q0_s_intra
                    q_z_intra[counter] = q0_z_intra
                    total_traded_intraday_s.append(abs(q0_s_intra*s[counter-1]))
                    total_traded_intraday_z.append(abs(q0_z_intra*z[counter-1]))
                
                #MOMENTUM TRADERS
                if (sum(returns_index[-sec_momentum::]) / sec_momentum) > (returns_index[-1]):
                    
                    q0_s_momentum = q_momentum*q0_s 
                    q0_z_momentum = q_momentum*q0_z
                    
                    q_s_momentum[counter]=(q0_s_momentum)
                    q_z_momentum[counter]=(q0_z_momentum)
        
                    q_s[counter]= q_s[counter] + q0_s_momentum
                    q_z[counter]= q_z[counter] + q0_z_momentum
                    
                    total_traded_momentum_s.append(abs(q0_s_momentum*s[counter-1]))
                    total_traded_momentum_z.append(abs(q0_z_momentum*z[counter-1]))
                    
                elif (sum(returns_index[-sec_momentum::]) / sec_momentum)< (returns_index[-1]):
                    
                    q0_s_momentum = -q_momentum*q0_s 
                    q0_z_momentum = -q_momentum*q0_z
                    
                    q_s_momentum[counter]=(q0_s_momentum)
                    q_z_momentum[counter]=(q0_z_momentum)
        
                    q_s[counter]= q_s[counter] + q0_s_momentum
                    q_z[counter]= q_z[counter] + q0_z_momentum
                    
                    total_traded_momentum_s.append(abs(q0_s_momentum*s[counter-1]))
                    total_traded_momentum_z.append(abs(q0_z_momentum*z[counter-1]))
                    
        
                    
                # Noise Traders
                lambda_b_s = A_s * np.exp(-k * delta_b_s[counter - 1])
                lambda_a_s = A_s * np.exp(-k * delta_a_s[counter - 1])
                lambda_b_z = A_z * np.exp(-k_z * delta_b_z[counter - 1])
                lambda_a_z = A_z * np.exp(-k_z * delta_a_z[counter - 1])
                
                if u_b_s[t] < lambda_b_s * deltat:
                    dN_b_s[counter] = 1. 
                else:
                    dN_b_s[counter] = 0.
                if u_a_s[t] < lambda_a_s * deltat:
                    dN_a_s[counter] = 1.
                else:
                    dN_a_s[counter] = 0.
                N_b_s[counter] = N_b_s[counter-1] + dN_b_s[counter]
                N_a_s[counter] = N_a_s[counter-1] + dN_a_s[counter]
                q_s[counter] =  q_s[counter] + q0_s * (N_b_s[counter] - N_a_s[counter])
                
                if u_b_z[t] < lambda_b_z * deltat:
                    dN_b_z[counter] = 1. 
                else:
                    dN_b_z[counter] = 0.
                if u_a_z[t] < lambda_a_z * deltat:
                    dN_a_z[counter] = 1.
                else:
                    dN_a_z[counter] = 0.
                N_b_z[counter] = N_b_z[counter-1] + dN_b_z[counter]
                N_a_z[counter] = N_a_z[counter-1] + dN_a_z[counter]
                q_z[counter] = q_z[counter] + q0_z * (N_b_z[counter] - N_a_z[counter])
                
                total_traded_noise_s.append(abs(q0_s * (N_b_s[counter] - N_a_s[counter]) * s[counter-1]))
                total_traded_noise_z.append(abs(q0_z * (N_b_z[counter] - N_a_z[counter]) * z[counter-1]))
                

                  # Avellaneda-Stoikov model
                delta_b_s[counter] = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)+gamma_s*sigma_s*sigma_s*T*q_s[counter],0.)) # simpifier 
                delta_a_s[counter] = RoundTick_s(max(0.5*gamma_s*sigma_s*sigma_s*T+(1./gamma_s)*np.log(1.+gamma_s/k)-gamma_s*sigma_s*sigma_s*T*q_s[counter],0.))
                delta_b_z[counter] = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)+gamma_z*sigma_z*sigma_z*T*q_z[counter],0))
                delta_a_z[counter] = RoundTick_z(max(0.5*gamma_z*sigma_z*sigma_z*T+(1./gamma_z)*np.log(1.+gamma_z/k_z)-gamma_z*sigma_z*sigma_z*T*q_z[counter],0))


                s[counter] = RoundTick_s(s[counter-1] + theta_s*q_s[counter] + sigma_s * dW_s[t])
                z[counter] = RoundTick_z(z[counter-1] + theta_z*q_z[counter] + sigma_z * dW_z[t])
                index[counter]=0.5*(s[counter]/s[0])*100 + 0.5*(z[counter]/z[0])*100
                
                returns_s.append(np.log(s[counter]/s[counter-1]))
                returns_z.append(np.log(z[counter]/z[counter-1]))
                returns_index.append(np.log(index[counter]/index[counter-1]))
                
                if t%60==0:
                    s_min.append(s[counter])
                    z_min.append(z[counter])
                
            


        delta_b_s[0] = delta_b_s[1]
        delta_a_s[0] = delta_a_s[1]
        delta_b_z[0] = delta_b_z[1]
        delta_a_z[0] = delta_a_z[1]
        
        p_b_s = s - delta_b_s
        p_a_s = s + delta_a_s
        delta_s = delta_a_s + delta_b_s
        
        p_b_z = z - delta_b_z
        p_a_z = z + delta_a_z
        delta_z = delta_a_z + delta_b_z

        dx_s = q0_s * (np.multiply(p_a_s,dN_a_s) - np.multiply(p_b_s,dN_b_s))
        x_s = np.cumsum(dx_s)
        dx_z = q0_z * (np.multiply(p_a_z,dN_a_z) - np.multiply(p_b_z,dN_b_z))
        x_z = np.cumsum(dx_z)
        pnl = x_s + np.multiply(q_s,s) + x_z + np.multiply(q_z,z)

        AllPnls[ind] = pnl[-1]
        q_sum_sq_s += np.sum(np.multiply(q_s,q_s))
        q_sum_s += np.sum(q_s)
        q_var_s = (1./(ndays * n)) * q_sum_sq_s - ((1./(ndays * n))  * q_sum_s) * ((1./(ndays * n))  * q_sum_s)
        q_sum_sq_z += np.sum(np.multiply(q_z,q_z))
        q_sum_z += np.sum(q_z)
        q_var_z = (1./(ndays * n)) * q_sum_sq_z - ((1./(ndays * n))  * q_sum_z) * ((1./(ndays * n))  * q_sum_z)
        
        
        sum_total_traded_noise_s=np.sum(total_traded_noise_s)
        sum_total_traded_intraday_s=np.sum(total_traded_intraday_s)
        sum_total_traded_momentum_s=np.sum(total_traded_momentum_s)
        sum_total_traded_noise_z=np.sum(total_traded_noise_z)
        sum_total_traded_intraday_z=np.sum(total_traded_intraday_z)
        sum_total_traded_momentum_z=np.sum(total_traded_momentum_z)
        sum_total_traded_daily_s = np.sum(total_traded_daily_s)
        sum_total_traded_daily_z = np.sum(total_traded_daily_z)
        print("\n")
        print("Proportion for s noise trades / (noise trades + intraday trades + momentum trades)", sum_total_traded_noise_s / (sum_total_traded_noise_s+sum_total_traded_intraday_s+sum_total_traded_momentum_s))
        print("Proportion for z noise trades / (noise trades + intraday trades + momentum trades)", sum_total_traded_noise_z / (sum_total_traded_noise_z+sum_total_traded_intraday_z+sum_total_traded_momentum_z))
        print("\n")
        print("Proportion for s momentum trades / (intraday trades + momentum trades)", sum_total_traded_momentum_s / (sum_total_traded_intraday_s+sum_total_traded_momentum_s))
        print("Proportion for z momentum trades / (intraday trades + momentum trades)", sum_total_traded_momentum_z / (sum_total_traded_intraday_z+sum_total_traded_momentum_z))
        print("\n")
        print("Proportion for s total intraday trades / (markow trades)", (sum_total_traded_noise_s+sum_total_traded_intraday_s+sum_total_traded_momentum_s) / sum_total_traded_daily_s)
        print("Proportion for z total intraday trades / (markow trades)", (sum_total_traded_noise_z+sum_total_traded_intraday_z+sum_total_traded_momentum_z) / sum_total_traded_daily_z)
        print("\n")
        print("Ratio noise traders s/z", sum_total_traded_noise_s / sum_total_traded_noise_z)
        print("Ratio momentum traders s/z", sum_total_traded_momentum_s / sum_total_traded_momentum_z)
        print("Ratio intraday traders s/z", sum_total_traded_intraday_s / sum_total_traded_intraday_z)
        print("\n")
        print("Spread moyen S tick unit", (1./TickSize_s)*np.mean(delta_s))
        print("Spread moyen Z tick unit", (1./TickSize_z)*np.mean(delta_z))
        print("Spread max S tick unit", (1./TickSize_s)*np.max(delta_s))
        print("Spread max Z tick unit", (1./TickSize_z)*np.max(delta_z))
        print("\n")
        print("Spread moyen S", np.mean(delta_s))
        print("Spread moyen Z", np.mean(delta_z))
        print("Spread max S ",  np.max(delta_s))
        print("Spread max Z tick unit",np.max(delta_z))
        print("\n")
        print("Quantité moyenne S tradée", np.mean(np.abs(q_s)))
        print("Quantité moyenne Z tradée", np.mean(np.abs(q_z)))
        print("Quantité max S tradée", np.max(np.abs(q_s)))
        print("Quantité max Z tradée", np.max(np.abs(q_z)))
        print("\n")
        

        if DisplayPlots:
             print("Displaying plots -----------------------------")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(T*ndays+1), np.array([p_b_s]).T, '-')
             plt.plot(range(ndays*T+1), np.array([s]).T, '-')
             plt.plot(range(ndays*T+1), np.array([p_a_s]).T, '-')
             plt.title('asset price: bid/mid/ask for asset s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), (1./TickSize_s) * np.array([delta_b_s,delta_a_s,delta_s]).T, '-')
             plt.legend(['delta_b','delta_a','delta'])
             plt.title('spreads (in tick units) for asset s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), 1e-3 * q_s, '-')
             plt.title('market-maker inventory (in K) for s')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
             
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), np.array([p_b_z]).T, '-')
             plt.plot(range(ndays*T+1), np.array([z]).T, '-')
             plt.plot(range(ndays*T+1), np.array([p_a_z]).T, '-')
             plt.title('asset price: bid/mid/ask for asset z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), (1./TickSize_z) * np.array([delta_b_z,delta_a_z,delta_z]).T, '-')
             plt.legend(['delta_b','delta_a','delta'])
             plt.title('spreads (in tick units) for asset z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), 1e-3 * q_z, '-')
             plt.title('market-maker inventory (in K) for z')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
        
             plt.figure()
             ax = plt.gca()
             ax.ticklabel_format(axis='y', useOffset=False)
             plt.plot(range(ndays*T+1), pnl, '-')
             plt.title('market-maker pnl (in USD)')
             plt.xticks(range(0,ndays*T+1, int(T*ndays/10)), range(0,ndays+1, int(ndays/10)))
             plt.xlabel("Days")
            
            
             plt.show()
                          
    else:
        return 0
    
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
    plt.title("Overlapping correlation of Eurusd and Btc  log returns")
    plt.xlabel("Tau : Increments of 1s")
    plt.ylabel("Pearson Correlation")
    plt.plot(high, label='Low', c='g')
    plt.plot(low, label='High', c='r')
    plt.legend()
    
def display_corr_min(tau_max=2880,s_min=s_min, z_min=z_min, IC=True):
    
    def artanh(x):
        return 0.5*math.log((1+x)/(1-x))
    
    def tanh(x):
        return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
       
    
    x=np.array(s_min)
    y=np.array(z_min)
    
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
        plt.title("Overlapping correlation of Eurusd and Btc  log returns")
        plt.xlabel("Tau : Increments of 1m")
        plt.ylabel("Pearson Correlation")
        plt.plot(high, label='Low', c='g')
        plt.plot(low, label='High', c='r')
        plt.legend()
    else:
        plt.plot(BtcEuroverlapping)
        plt.title("Overlapping correlation of Eurusd and Btc  log returns")
        plt.xlabel("Tau : Increments of 1m")
        plt.ylabel("Pearson Correlation")
        plt.legend()


if __name__ == '__main__':
    model(2)
    '''
    display_corr()
    display_corr_min()
    display_corr_min(20000)
    display_corr_min(20000, IC=False)
    plt.plot(s_min)
    plt.plot(z_min)
    '''
