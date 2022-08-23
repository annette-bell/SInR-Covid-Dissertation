#!/usr/bin/env python
# coding: utf-8

# # Decreasing Computational Time 
#  
# The code below examines the transmission rate results for the gamma, lognormal, and weibull distributions of generation intervals. Each of these three distributions were modeled from a population mean of $\bar{x}=4.9$ and standard deviation of $\sigma=2.0$. In the outputs, we can visualize the type of results that were initially retuned with estimates of $w_j = \frac{1}{n}$ $\forall$ $j \in 1,...,n$ and $\lambda=n\sigma$ where $\sigma=\frac{1}{15}$. To see a more indepth explanation and analysis of the entire study, please see the thesis paper located in the GitHub repository. 
# 
# ---
# 
# #### Step 1: Import all necessary libraries

# In[1]:


import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import weibull_min
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.stats as stats 
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from matplotlib.offsetbox import AnchoredText

#import warnings
#warnings.filterwarnings("ignore")


# <br><br>
# 
# ---
# 
# #### Step 2: Pull Generation Interval Data

# In[2]:


gi_df = pd.read_excel('data\GI_Values.xlsx')
gi_df.sort_values(by=['Dist_Type','Mean'], inplace=True)
gi_df.reset_index(drop=True, inplace=True)

#=====GAMMA DISTRIBUTION=====
gamma_df = gi_df[gi_df["Dist_Type"]=="gamma"]
gamma_df.loc[:,"shape"]= (gamma_df.loc[:,"Mean"]**2)/(gamma_df.loc[:,"SD"]**2)
gamma_df.loc[:,"scale"] = (gamma_df.loc[:,"SD"]**2)/(gamma_df.loc[:,"Mean"])


#=====WEIBULL DISTRIBUTION=====
weibull_df = gi_df[gi_df["Dist_Type"]=="weibull"]

#=====LOGNORMAL DISTRIBUTION=====
lognorm_df = gi_df[gi_df["Dist_Type"]=="lognormal"]


# <br><br>
# 
# ---
# 
# #### Step 3: Create Functions to Estimate the Tranmission Rates
# 
# To see full explanation of the code, please see Section {} of the thesis. 

# In[3]:


#===================OBJECTIVE FUNCTION====================
def objective(w_lamb,tau):
    '''
    Objective: To maximize the log likelihood of W(u) (ie min(-W(u)))
    
    Inputs:
        w_lamb= weights [w_1, w_2,...,w_n] and lambda in one list
        tau = set of times since infection [tau_1, tau_2,...,tau_m]
    Outputs: 
        objective: value (-W(u))
    '''
    
    w=w_lamb[:-1]    
    lamb=w_lamb[-1]
    
    n=len(w)
    
    objective = 0
    
    for tau_i in tau: #FOR EACH TIME SINCE EXPOSURE
        wlog_val = w[0]
        for j in range(1,n): #CALCULATE TERMS WITHIN LOG  
            wlog_val = wlog_val + (w[j]*(((lamb*tau_i)**j)/math.factorial(j)))
        objective = objective + (math.log(wlog_val) - (lamb*tau_i) + math.log(lamb))
        
    return(-1 *objective)

#===================CONSTRAINT 1====================
def constraint(w_lamb):
    '''
    Constraint 1:  Weights must sum to 1
    Inputs:
        w: list of weights
    
    Outputs:
        constraint1: value of 1 - sum of the weights
    '''
    w=w_lamb[:-1]
    n = len(w)
    
    constraint1 = 1
    
    for j in range(n):
        constraint1 = constraint1 - w[j]
    return constraint1


#=================== CALCULATE WEIGHTS AND BETAS ====================
def solver(tau, R_0, n):
    '''
    The following function returns a list of weights given the 5 inputs. 
    
    Inputs:
        tau: list of generation intervals times (in days)
        R_0: basic reproduction number
        n: number of infectious comparments
        
    Output:
        w_val: list of weights (based on minimization)
        lambda: lambda value   (based on minimization)
        b_val: list of betas   (based on minimization)
    '''
    
    wl_0 = np.zeros(n+1)  
    
    
    for i in range(n):
        wl_0[i] = 1/n
    
    wl_0[-1]= n/15
    
    b = (0, 1)
    bnds=()
    for i in range(n):
        bnds = bnds + (b,)
        
    b_lamb = (0.00000000001, None)#np.mean(tau)/np.var(tau)+.3)
    bnds = bnds + (b_lamb,)

    #specify constraints
    con1 = {'type': 'eq', 'fun': constraint}
    cons = ([con1])
    
    #optimize
    solution = minimize(objective, wl_0, method='SLSQP',                        args=(tau), bounds=bnds,constraints=cons)

    #get weights
    w_val = solution.x[:-1]
    lamb = solution.x[-1]
    b_val = [weight*lamb*R_0 for weight in w_val]
        
    return(w_val, lamb, b_val)



#===========Initial Estimates ===============
def solutions(gi_data,min_n, max_n, R_0):
    
    weight = []
    lambda_n = []
    beta = []
    
    for n_val in list(range(min_n, max_n+1)):   
        w, l, b = solver(gi_data, R_0, n_val)
        
        #Update on status of run
        if n_val == int((max_n+1-min_n)/4+min_n): 
            print("n=",str(n_val)," is done.25% Done!")
            
        if n_val == int((max_n+1-min_n)/2+min_n): 
            print("n=",str(n_val)," is done. Half way there!")
        
        if n_val == int(3*(max_n+1-min_n)/4 + min_n): 
            print("n=",str(n_val)," is done. 75% Done!")
            
        if n_val == max_n: 
            print("Done!")

        weight.append(w)
        lambda_n.append(l)
        beta.append(b)
    
    return weight, lambda_n, beta


#=================== BETAS(u) ====================
def beta_u(tau,beta_vals, lambda_n):
    '''
    Beta(u): Find transmission rate for every time in u_i
    
    Inputs:
        u_i: list of generation intervals times (in days)
        beta_vals: list of betas (based on minimization)
        lambda_n: rate that infected move to the next compartment 

    Outputs:
        y: 
    
    
    '''
    n = len(beta_vals)
    y = []
    
    for tau_i in tau:
        transmission=0
        for j in range(n):
            transmission = transmission + beta_vals[j]*((np.exp(-lambda_n*tau_i)*(lambda_n*tau_i)**j)/math.factorial(j))

        y.append(transmission)

    return(y)

#============Beta(u) Plot==================
def beta_u_plot(lambda_n, beta_vals):
    
    #create x axis
    x = np.linspace(0, 16, 100)
    
    #create df of x, y data
    beta_df = pd.DataFrame(x, columns=["x"])
    
    beta_df[str(len(beta_vals))] = [float(b) for b in beta_u(x,beta_vals,lambda_n)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta_df.x, y=beta_df.iloc[:, 1]))
    
            
    #format graph
    fig.update_layout(legend_title_text='Compartment')
    fig.update_xaxes(title_text='Days', nticks=20)
    fig.update_yaxes(title_text='Transmission Rate')

    return(fig)


# <br><br>
# 
# ---
# 
# #### Step 4: Create Generation Interval Data
# The data was modeled from $\bar{x}=4.9$ and $\sigma=2.0$. We used a random seed of 1234 and took 5,000 observations. 

# In[4]:


#Settings:
np.random.seed(1234)
m=5000
#=========GAMMA============
gi_g = 3 
gamma_shape = gamma_df.loc[gi_g]["shape"]
gamma_scale = gamma_df.loc[gi_g]["scale"]
gi_gamma_obs = np.random.gamma(gamma_shape, gamma_scale, m)

#=========LOGNORMAL============
log_mean = gamma_df.loc[gi_g]["Mean"]
log_sd = gamma_df.loc[gi_g]["SD"]
log_var = log_sd**2
norm_mean = np.log(log_mean)-0.5*np.log((log_sd/log_mean)**2+1) #scale=e^norm_mean
norm_var = np.log((log_sd/log_mean)**2+1) 
norm_sd = np.sqrt(norm_var) # equivalent to the shape
gi_lognorm_obs = lognorm.rvs(s=norm_sd, scale=math.exp(norm_mean), size=m)


#=========WEIBULL============
weibull_mean = gamma_df.loc[gi_g]["Mean"]
weibull_std = gamma_df.loc[gi_g]["SD"]

def G(k): 
    '''Calculates the Gamma portion of the equation'''
    return math.gamma(1+2/k)/(math.gamma(1+1/k)**2)
def f(k,b):
    '''States that the Gamma solution G(k) must be equal to b'''
    return G(k)-b #function solves for k

b = (weibull_std**2)/(weibull_mean**2)+1
init = 1 # The starting estimate for the root of f(x) = 0.
weibull_shape = fsolve(f,init,args=(b))[0]
weibull_scale = weibull_mean/math.gamma(1+1/weibull_shape)
gi_weibull_obs = weibull_min.rvs(weibull_shape,scale=weibull_scale, size=m)


# In[5]:


#=========VISUALIZE==========
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
count=0
titles=['Generation Interval of Gamma Distribution','Generation Interval of Lognormal Distribution',       'Generation Interval of Weibull Distribution']
layout=[axs[0,0], axs[0,1],axs[1,0]]
dist_string = ["Gamma", "Lognormal","Weibull"]
for dist in [gi_gamma_obs,gi_lognorm_obs,gi_weibull_obs]:
    ax1 = sns.histplot(data=dist, kde=True, color=sns.color_palette()[count], 
                       line_kws={"lw":3}, ax=layout[count])
    
    #ax1.set_title(titles[count], fontsize=14)
    ax1.set_xlabel('Time Since Infection (days)', fontsize=15)
    ax1.set_ylabel('Frequency', fontsize=15)
    ax1.set_ylim([0, 350])
    
    textstr1 = '\n'.join((
        "%s" % dist_string[count],
        r'$\bar{x}=%.3f$' % (np.mean(dist), ),
        r'$s=%.3f$' % (np.std(dist), )))
    at1 = AnchoredText(textstr1, prop=dict(size=14), frameon=True, loc='upper right')
    at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.15")
    ax1.add_artist(at1)
    
    count+=1

ax = sns.kdeplot(data=[gi_gamma_obs,gi_lognorm_obs,gi_weibull_obs], 
                 fill= True, linewidth=1, ax=axs[1,1])
ax.set_xlabel('Time Since Infection (days)', fontsize=15)
ax.set_ylabel('Density', fontsize=15)

plt.legend(title='Distribution', loc='upper right', labels=['Weibull', 'Lognormal',"Gamma"], 
           title_fontsize=14, fontsize=14)
    
plt.show()


# <br><br><br>
# 
# ---
# 
# #### Step 5: Inspect Initial Distribution Results
# 
# Steps to help determine initial estimates for each distribution are below.
# 
# 1. Run Model
# 2. Visualize Trasmission Curve
# 3. Show Results of the objective value, holding period, and transmission weights.
# 
# 
# 
# ###### Gamma Distribution:

# In[6]:


print("Simulated from: Gamma(shape=", str(round(gamma_shape,3)),", scale=", str(round(gamma_scale,3)),")     where mean=",       str(round(gamma_df.loc[gi_g]["Mean"],3)), "    and  std=",str(round(gamma_df.loc[gi_g]["SD"],3)),sep="")
print("Actual:         Gamma(shape=", str(round((np.mean(gi_gamma_obs)**2)/np.var(gi_gamma_obs),3)),
      ", scale=", str(round(np.var(gi_gamma_obs)/np.mean(gi_gamma_obs),3)),")     where mean=", \
      str(round(np.mean(gi_gamma_obs),3)),"  and  std=", str(round(np.std(gi_gamma_obs),3)), sep="")

#visualize
plt.figure(figsize = (10,5))
ax= sns.histplot(x=gi_gamma_obs, binwidth=0.2,kde=True, line_kws={'lw': 1})
ax.set(xlabel='Time Since Infection', ylabel='Frequency', 
       title="Generation Interval Observations of Gamma Distribution")
ax.lines[0].set_color('black')


# In[7]:


R0 = 2.3
w_gamma, l_gamma, b_gamma = solutions(gi_gamma_obs, 1, 25, R0)


# In[8]:


gamma_fig = make_subplots(rows=1, cols=1)

for beta,lamb in zip(b_gamma,l_gamma):
    data = beta_u_plot(lamb, beta)
    
    gamma_fig.add_trace(
        go.Scatter(x=data['data'][0]['x'], y=data['data'][0]['y'],
                  name="GI of n="+str(len(beta))),
        row=1, col=1
    )

#-----STYLING--------
gamma_fig.update_layout(
    title="Expected Infectious Curve of Gamma Distribution",
    xaxis_title="Duration of Infection (days)",
    yaxis_title="Transmission Rate",
    legend_title="Legend",
    font=dict(size=14),
    width=900,
    height=500,
)

gamma_fig.show()


# In[9]:


for w, l in zip(w_gamma,l_gamma):
    print("n=",str(len(w)))
    print("Obj: ",str(round(objective(list(w)+[l],gi_gamma_obs),1)),          "        lambda:", str(round(l,2)))
    print("weights", str([round(val,2) for val in w]), "\n")


# <br><br><br>
# 
# ---
# 
# 
# ###### Lognormal Distribution:
# 
# 

# In[10]:


act_norm_mean = np.log(np.mean(gi_lognorm_obs))-0.5*np.log((np.std(gi_lognorm_obs)/np.mean(gi_lognorm_obs))**2+1)
act_norm_std = np.sqrt(np.log((np.std(gi_lognorm_obs)/np.mean(gi_lognorm_obs))**2+1))

print("Simulated from: Lognormal(mean=", str(round(log_mean,3)),", std=", str(round(log_sd,3)),")",      "       where Normal(mean=", str(round(norm_mean,3)), ", std=", str(round(norm_sd,3)),")", sep="")

print("Actual:         Lognormal(mean=", str(round(np.mean(gi_lognorm_obs),3)), ", std=", str(round(np.std(gi_lognorm_obs),3)),")",      "     where Normal(mean=", str(round(act_norm_mean,3)), ", std=", str(round(act_norm_std,3)),")", sep="")


plt.figure(figsize = (10,5))
ax= sns.histplot(x=gi_lognorm_obs, binwidth=0.35,kde=True, line_kws={'lw': 1})
ax.set(xlabel='Time Since Infection', ylabel='Frequency', 
       title="Generation Interval Observations of Lognormal Distribution")
ax.lines[0].set_color('black')


# In[11]:


R0 = 2.3
w_log, l_log, b_log = solutions(gi_lognorm_obs, 1, 25, R0)


# In[12]:


lognorm_fig = make_subplots(rows=1, cols=1)

for beta,lamb in zip(b_log,l_log):
    data = beta_u_plot(lamb, beta)
    
    lognorm_fig.add_trace(
        go.Scatter(x=data['data'][0]['x'], y=data['data'][0]['y'],
                  name="GI of n="+str(len(beta))),
        row=1, col=1
    )

#-----STYLING--------
lognorm_fig.update_layout(
    title="Expected Infectious Curve of Lognormal Distribution",
    xaxis_title="Duration of Infection (days)",
    yaxis_title="Transmission Rate",
    legend_title="Legend",
    font=dict(size=14),
    width=900,
    height=500,
)
lognorm_fig.show()


# In[13]:


for w, l in zip(w_log,l_log):
    print("n=",str(len(w)))
    print("Obj: ",str(round(objective(list(w)+[l],gi_lognorm_obs),1)),          "        lambda:", str(round(l,2)))
    print("weights", str([round(val,2) for val in w]), "\n")


# <br><br><br><br><br>
# 
# ---
# 
# ###### Weibull Distribution:

# In[14]:


print("Simulated from: Weibull(shape=", str(round(weibull_shape,3)),
      ", scale=", str(round(weibull_scale,3)),")      where mean=", \
      str(round(weibull_mean,3)), "   and  std=",str(round(weibull_std,3)),sep="")

act_weibull_shape = (np.std(gi_weibull_obs)/np.mean(gi_weibull_obs))**(-1.086) 
act_weibull_scale = np.mean(gi_weibull_obs)/math.gamma(1+1/act_weibull_shape)
print("Actual:         Weibull(shape=", str(round(act_weibull_shape,3)),      ", scale=", str(round(act_weibull_scale,3)),")     where mean=",      str(round(np.mean(gi_weibull_obs),3))," and  std=", str(round(np.std(gi_weibull_obs),3)),sep="")

#visualize
plt.figure(figsize = (10,5))
ax= sns.histplot(x=gi_weibull_obs, binwidth=0.2,kde=True, line_kws={'lw': 1})
ax.set(xlabel='Time Since Infection', ylabel='Frequency', 
       title="Generation Interval Observations of Weibull Distribution")
ax.lines[0].set_color('black')


# In[15]:


R0 = 2.3
w_weibull, l_weibull, b_weibull = solutions(gi_weibull_obs, 1, 25, R0)


# In[16]:


weibull_fig = make_subplots(rows=1, cols=1)

for beta,lamb in zip(b_weibull,l_weibull):
    data = beta_u_plot(lamb, beta)
    
    weibull_fig.add_trace(
        go.Scatter(x=data['data'][0]['x'], y=data['data'][0]['y'],
                  name="GI of n="+str(len(beta))),
        row=1, col=1
    )

#-----STYLING--------
weibull_fig.update_layout(
    title="Expected Infectious Curve of Weibull Distribution",
    xaxis_title="Duration of Infection (days)",
    yaxis_title="Transmission Rate",
    legend_title="Legend",
    font=dict(size=14),
    width=900,
    height=500,
)
weibull_fig.show()


# In[17]:


for w, l in zip(w_weibull,l_weibull):
    print("n=",str(len(w)))
    print("Obj: ",str(round(objective(list(w)+[l],gi_weibull_obs),1)),          "        lambda:", str(round(l,2)))
    print("weights", str([round(val,2) for val in w]), "\n")

