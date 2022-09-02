#!/usr/bin/env python
# coding: utf-8

# # Covid-19 SI$_n$R Model
# 
# 
# First, import all necessary libraries

# In[1]:


import numpy as np
import ast
from os.path import exists
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


# ## Simulating the Generation Interval 

# In[2]:


#import previous generation intervals
gi_df = pd.read_excel('data\GI_Values.xlsx')

#find shape and scale for each
gi_df["shape"] = (gi_df["Mean"]**2)/(gi_df["SD"]**2)
gi_df["scale"] = (gi_df["SD"]**2)/(gi_df["Mean"])

#sort from smallest to largest 
gi_df.sort_values(by=['Dist_Type','Mean'], inplace=True)
gi_df.reset_index(drop=True, inplace=True)

gamma_df = gi_df[gi_df["Dist_Type"]=="gamma"]
weibull_df = gi_df[gi_df["Dist_Type"]=="weibull"]
lognorm_df = gi_df[gi_df["Dist_Type"]=="lognormal"]


# In[3]:


#Settings:
np.random.seed(1234)
m=5000

gi_pop = 3
pop_mean = gamma_df.loc[gi_pop]["Mean"]
pop_std = gamma_df.loc[gi_pop]["SD"]


#=========GAMMA============
gamma_shape = (pop_mean**2)/(pop_std**2)
gamma_scale = (pop_std**2)/(pop_mean)
gi_gamma_obs = np.random.gamma(gamma_shape, gamma_scale, m)

#=========LOGNORMAL============
log_mean = pop_mean
log_sd = pop_std
log_var = log_sd**2
norm_mean = np.log(log_mean)-0.5*np.log((log_sd/log_mean)**2+1) #scale=e^norm_mean
norm_var = np.log((log_sd/log_mean)**2+1) 
norm_sd = np.sqrt(norm_var) # equivalent to the shape
gi_lognorm_obs = lognorm.rvs(s=norm_sd, scale=math.exp(norm_mean), size=m)


#=========WEIBULL============
weibull_mean = pop_mean
weibull_std = pop_std

def G(k): 
    return math.gamma(1+2/k)/(math.gamma(1+1/k)**2)
def f(k,b):
    return G(k)-b #function solves for k

b = (weibull_std**2)/(weibull_mean**2)+1
init = 1 # The starting estimate for the root of f(x) = 0.
weibull_shape = fsolve(f,init,args=(b))[0]
weibull_scale = weibull_mean/math.gamma(1+1/weibull_shape)
gi_weibull_obs = weibull_min.rvs(weibull_shape,scale=weibull_scale, size=m)


# In[4]:


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


# <br><br><br><br>
# 
# ---
# 
# ## Create Functions used to Minimize Objective Function
# 
# #### Step 1: Minimize 
# $\min (-\sum_{i=1}^m \ln(w_1+w_2\frac{(\lambda u_i)^1}{1!}+...+w_n\frac{(\lambda u_i)^{n-1}}{n-1!})-\lambda u_i)$ 
# 
# s.t.  $w_1+...+w_n = 1$ (***equality constraint***) <br>
#        $0 \le w_1 \le 1$ (***bounds***) <br>
#        ...<br>
#        $0 \le w_n \le 1$ (***bounds***)<br>
# 
# 
# <br>
# 
# #### Step 2: Calculate Transmission Rates ($\beta_j$) 
# 
# $\beta_j = R_0 \lambda w_j$
# 
# 
# <br>
# 
# #### Step 3: Visualize:
# 
# $\beta(u)= \sum_{j=1}^n \beta_j \frac{e^{-\lambda u}  (\lambda u)^{j-1}}{(j-1)!}$

# In[5]:


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
def solver(tau, R_0, n, dist_type):
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
    
    if dist_type == "gamma" or dist_type == "lognorm":
        shape = (np.mean(tau)**2)/np.var(tau) #shape of the disribution
        w2 = shape - math.trunc(shape)        #expected weight of "second" compartment 
        w1 = 1 - w2                           #expected weight of "first" compartment
        comps = [math.trunc(shape)-1, math.trunc(shape)] #location of the "first" and "second" compartments where weights should exceed one
        weights = [w1, w2]

        for c, w in zip(comps,weights):
            wl_0[c] = w

        wl_0[-1]= np.mean(tau)/np.var(tau)
    
    
    #elif dist_type == "lognorm":
    #    for i in range(n):
    #        wl_0[i] = 1/n#

    #    log_mean = np.mean(tau)
    #    log_std = np.std(tau)
    #    norm_mean = np.log(log_mean)-0.5*np.log((log_std/log_mean)**2+1)
    #    wl_0[-1]=  norm_mean
        
    elif dist_type == "weibull":
        for i in range(n):
            wl_0[i] = 1/n

        wl_0[-1]= np.std(tau)
    
    b = (0, 1)
    bnds=()
    for i in range(n):
        bnds = bnds + (b,)
        
    b_lamb = (0.00000000001, None)
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



#==========SOLVE FOR MULTIPLE COMPARTMENTS========
def solutions(gi_data,min_n, max_n, R_0, dist_type):
    
    weight = []
    lambda_n = []
    beta = []
    obj = []
    
    for n_val in list(range(min_n, max_n+1)):
        if n_val == min_n: 
            str_p = "Solving: "+ str(dist_type)+ " with R_0="+str(R_0)+" for n in "+ str(min_n)+",...,"+str(max_n)
            print(str_p)
            
        w, l, b = solver(gi_data, R_0, n_val, dist_type)
        o = objective(list(w)+[l],gi_data)
        
        if n_val == int((max_n+1-min_n)/4+min_n): 
            print("n=",str(n_val)," is done. 25% Done!")
            
        if n_val == int((max_n+1-min_n)/2+min_n): 
            print("n=",str(n_val)," is done. Half way there!")
        
        if n_val == int(3*(max_n+1-min_n)/4 + min_n): 
            print("n=",str(n_val)," is done. 75% Done!")
            
        if n_val == max_n: 
            print("Done!")

        weight.append(w)
        lambda_n.append(l)
        beta.append(b)
        obj.append(o)
    
    return weight, lambda_n, beta, obj


#=================== BETAS(u) ====================
def beta_u(u_i,beta_vals, lambda_n):
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
    
    for u in u_i:
        transmission=0
        for j in range(n):
            transmission = transmission + beta_vals[j]*((np.exp(-lambda_n*u)*(lambda_n*u)**j)/math.factorial(j))

        y.append(transmission)

    return(y)

#============Beta(u) Plot==================
def beta_u_plot(lambda_n, beta_vals):
    
    #create x axis
    x = np.linspace(0, 15, 100)
    
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


# 
# <br><br>
# 
# ---
# 
# 
# ## Transmission Rate Solutions:
# 
# The code below is based on the following:
# * The generation intervals $[\tau_1,...,\tau_m] $ follow mean of about 4.4.9 and standard deviation or about 1.99
# * Observations $m=$ 5,000
# * Basic reproduction number $R_0$ = 2.3
# * Seed = 1234

# In[6]:


file_exists = exists('results/results.xlsx')
dist_string = ["Gamma", "Lognormal","Weibull"]

if file_exists == False:
    #If file doesn't exist, run model and save data
    print("File doesn't exist.")
    max_n = 25
    R0 = 2.3
    w_gamma, l_gamma, b_gamma, o_gamma = solutions(gi_gamma_obs, 7, max_n, R0, "gamma")
    w_lognorm, l_lognorm, b_lognorm, o_lognorm = solutions(gi_lognorm_obs, 7, max_n, R0, "lognorm")
    w_weibull, l_weibull, b_weibull, o_weibull = solutions(gi_weibull_obs, 7, max_n, R0, "weibull")

    #save data in file
    data_info = str(pop_mean)+"|"+str(pop_std)+"|"+str(R0) 
    gamma_dict = {'n': list(range(7,max_n+1)), 'w': [list(w) for w in w_gamma], 'lambda': l_gamma, data_info: b_gamma} 
    lognorm_dict = {'n': list(range(7,max_n+1)), 'w': [list(w) for w in w_lognorm], 'lambda': l_lognorm, data_info: b_lognorm} 
    weibull_dict = {'n': list(range(7,max_n+1)), 'w': [list(w) for w in w_weibull], 'lambda': l_weibull, data_info: b_weibull} 

    gamma_results = pd.DataFrame(gamma_dict)
    lognorm_results = pd.DataFrame(lognorm_dict)
    weibull_results = pd.DataFrame(weibull_dict)
    
    writer = pd.ExcelWriter('results/results.xlsx', engine='xlsxwriter')
    gamma_results.to_excel(writer,sheet_name='Gamma_18AUG2022')
    lognorm_results.to_excel(writer,sheet_name='Lognorm_18AUG2022')
    weibull_results.to_excel(writer,sheet_name='Weibull_18AUG2022')
    writer.save()

else:
    #if file exists, pull data from results file
    w_all, l_all, b_all = [],[],[]
    for sheet in ['Gamma_18AUG2022','Lognorm_18AUG2022','Weibull_18AUG2022']:
        gi_results_df = pd.read_excel('results/results.xlsx', sheet_name=sheet)

        column_names = list(gi_results_df.columns)[-4:] #should be: ['n', 'w', 'lambda', 'b for R0=2.3']
        w_combine, b_combine = [],[]

        for w in gi_results_df[column_names[1]]: #weights
            w_list = ast.literal_eval(w)
            w_combine.append(w_list)

        l_combine = list(gi_results_df[column_names[2]])

        for b in gi_results_df[column_names[3]]: #transmission
            b_list = ast.literal_eval(b)
            b_combine.append(b_list)

        w_all.append(w_combine)
        l_all.append(l_combine)
        b_all.append(b_combine)
        
    #State what R0 is:
    r0_val = column_names[3]
    R0 = float(r0_val.split("|")[-2])
    
    #Name variables that will be used throughout the duration
    w_gamma, w_lognorm, w_weibull = w_all[0], w_all[1], w_all[2]
    l_gamma, l_lognorm, l_weibull = l_all[0], l_all[1], l_all[2]
    b_gamma, b_lognorm, b_weibull = b_all[0], b_all[1], b_all[2]
    print("File exists: Pulled data from results.xlsx")


# In[8]:


fig = make_subplots(rows=1, cols=3, 
                   subplot_titles=('&#x3B2;(&#x1D70F;) of Gamma',  '&#x3B2;(&#x1D70F;) of Lognormal',
                                   "&#x3B2;(&#x1D70F;) of Weibull"))

for beta,lamb in zip(b_gamma,l_gamma):
    data = beta_u_plot(lamb, beta)
    
    fig.add_trace(
        go.Scatter(x=data['data'][0]['x'], y=data['data'][0]['y'],
                  name="GI of n="+str(len(beta))),
        row=1, col=1
    )
    
for beta,lamb in zip(b_lognorm,l_lognorm):
    data = beta_u_plot(lamb, beta)
    
    fig.add_trace(
        go.Scatter(x=data['data'][0]['x'], y=data['data'][0]['y'],
                  name="GI of n="+str(len(beta))),
        row=1, col=2
    )
    
for beta,lamb in zip(b_weibull,l_weibull):
    data = beta_u_plot(lamb, beta)
    
    fig.add_trace(
        go.Scatter(x=data['data'][0]['x'], y=data['data'][0]['y'],
                  name="GI of n="+str(len(beta))),
        row=1, col=3
    )

#-----STYLING--------
fig.update_layout(
    title="Expected Infectious Curve of Distributions",
    yaxis_title="Transmission Rate",
    legend_title="Legend",
    font=dict(
        #family="Courier New, monospace",
        size=14
    ),
    width=1100, 
    height=400,
)

fig.update_yaxes(range=[-0.02,0.6])
fig.update_xaxes(title="Duration of Infection (days)", row=1, col=2)
fig.show()


# <br>
# 
# A compartment size of $n= 10$ was ultimately selected:

# In[9]:


n_choice = 3 #n=1 -> 7th compartment
b_n = [b_gamma[n_choice], b_lognorm[n_choice], b_weibull[n_choice]]
l_n = [l_gamma[n_choice], l_lognorm[n_choice], l_weibull[n_choice]]
w_n = [w_gamma[n_choice], w_lognorm[n_choice], w_weibull[n_choice]]
color=["skyblue","darkorange","green"]
count = 0

fig = make_subplots(rows=1, cols=1)

dist = ["Gamma", "Lognormal","Weibull"]

for beta,lamb in zip(b_n,l_n):
    data = beta_u_plot(lamb, beta)
    
    fig.add_trace(
        go.Scatter(x=data['data'][0]['x'], y=data['data'][0]['y'],
                  name=dist[count], line_color=color[count],
                  line=dict(width=3)),
        row=1, col=1
    )
    count+=1
    

#-----STYLING--------
fig.update_layout(
    title="Estimated Infectious Curve &#x3B2;(&#x1D70F;) of each Distribution",
    xaxis_title="Duration of Infection (days)",
    yaxis_title="Transmission Rate", legend_title="Legend", font=dict(size=14),
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)
fig.show()

for i in range(3):
    print(dist[i], ": ", [round(w,3) for w in w_n[i]], "    lambda=",l_n[i], sep="")


# In[10]:


gi_data1 = [gi_gamma_obs,gi_lognorm_obs,gi_weibull_obs]
for i in range(3):
    print(dist_string[i], ": ", [round(b,3) for b in b_n[i]], "           objective:", round(objective(list(w_n[i])+[l_n[i]],gi_data1[i]),3), sep="")


# <br><br><br><br>
# 
# ---
# 
# ## Compartmental Model
# 
# First, we create the function to calculate the SI$_n$R models

# In[11]:


#============SInR and SInRS Plots==================
def SInR(y, t, N, c, h, beta, lambda_n):
    '''
    Inputs:
        y: initial condition vector (S, I, R)
        t: time points (in days)
        N: total population
        h: waning immunity rate
        beta: transmission rate
        lambda_n: optimized holding period
        
    Outputs:
        pop_status: returns a list of how many people are in each compartment
    '''
    S = y[0]
    I = y[1:-1]
    R = y[-1]
    
    npI = np.array(I)
    npBeta = np.array(beta)
    
    #Calculate dSdt
    dSdt = -S/N * np.sum(npI * npBeta)*c+ h*R
    
    #Calculate dI1dt
    dI1dt = S/N* np.sum(npI * npBeta)*c - lambda_n* I[0]
    
    #Calculate dI_dt values from n=2,..,n
    dIndt = []
    for index in range(len(I)-1):
        dIndt.append(lambda_n* I[index] - lambda_n* I[index+1])

    #Calculate dRdt
    dRdt = lambda_n * I[len(I)-1] - h*R

    #Create list of results for S through R
    pop_status = [dSdt, dI1dt]
    pop_status.extend(dIndt)
    pop_status.append(dRdt)
    
    return pop_status


#----------- SInR PLOT -----------
def SInR_plot(y_t0, t, N, c, h, beta, lambda_n):
# Integrate the SIR equations over the time grid, t.
    ret = odeint(SInR, y_t0, t, args=(N, c, h, beta, lambda_n))

    S = ret.T[0]
    I = sum(ret.T[1:-1])
    R = ret.T[-1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S/N,name="Susceptible"))
    fig.add_trace(go.Scatter(x=t, y=I/N,name="Sum of Infected Compartments"))
    fig.add_trace(go.Scatter(x=t, y=R/N,name="Recovered"))#

    fig.update_layout(legend_title_text='Compartment')
    fig.update_xaxes(title_text='Time (Days)')#,nticks=20)
    fig.update_yaxes(title_text='Percentage of the Population')

    return fig


# <br><br><br>
# 
# ---
# 
# ### SI$_n$R Model
# 
# We now calculate the populations over period of time
# 
# Note: Givens include
# * Population $N$ of 67886011
# * 1 person initially infected ($I_1(0)=1$)
# * Entire population is susceptible ($S(0)=N-I_1(0)$)
# 
# *When $h=0$ the equation is a SI$_n$R model; however, whenever $h$ is greater than 0, then the model is a SI$_n$RS model.*

# In[12]:


period=180
count=0

for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    #SIR MODEL
    fig = SInR_plot(y_t0, t, N, 1, 0, b, l)
    fig.update_layout(title="SI\u2099R Model of "+dist[count]+ " Generation Interval",
                     legend=dict(yanchor="top", y=0.35, xanchor="left", x=0.01),font_size=14)
    fig.show()
    count+=1
    


# In[13]:


period=180
count=0

SInR_compare = go.Figure()
dash = ["dot","dash"]

for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    #SIR MODEL
    if count == 0:
        fig = SInR_plot(y_t0, t, N, 1, 0, b, l)
        
        s_data = list(fig['data'][0]['y'])
        i_data = list(fig['data'][1]['y'])
        r_data = list(fig['data'][2]['y'])
        
        SInR_compare.update_layout(title="SI\u2099R Model for all Generation Intervals",
                         legend=dict(yanchor="top", y=0.65, xanchor="left",x=0.01), font_size=14)
        
        SInR_compare.add_trace(go.Scatter(x=t, y= s_data, name= "Susceptible: " +dist[count]+" GI", 
                                 line_color="blue",))
        SInR_compare.add_trace(go.Scatter(x=t, y= i_data, name= "Infected:     " +dist[count]+" GI", 
                                 line_color="red",))
        SInR_compare.add_trace(go.Scatter(x=t, y= r_data, name= "Recovered:  " +dist[count]+" GI", 
                                 line_color="green"))
        
        
        count+=1
    
    else:
        fig2 = SInR_plot(y_t0, t, N, 1, 0, b, l)
        s_data = list(fig2['data'][0]['y'])
        i_data = list(fig2['data'][1]['y'])
        r_data = list(fig2['data'][2]['y'])
        
        SInR_compare.add_trace(go.Scatter(x=t, y= s_data, name= "Susceptible: " +dist[count]+" GI", 
                                 line_color="blue", line_dash=dash[count-1]))
        SInR_compare.add_trace(go.Scatter(x=t, y= i_data, name= "Infected:     " +dist[count]+" GI", 
                                 line_color="red", line_dash=dash[count-1]))
        SInR_compare.add_trace(go.Scatter(x=t, y= r_data, name= "Recovered:  " +dist[count]+" GI", 
                                 line_color="green", line_dash=dash[count-1]))
       
        
        count+=1
SInR_compare.update_xaxes(title_text='Time (Days)')#,nticks=20)
SInR_compare.update_yaxes(title_text='Percentage of the Population')
SInR_compare.show()


# In[15]:


#pull uk total cases data
compare_infected = go.Figure()


#-----ZOOMED OUT----
period=150
count=0
for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    fig = SInR_plot(y_t0, t, N, 1, 0, b, l)
    
    i_data = list(fig['data'][1]['y'])
    
    compare_infected.add_trace(
        go.Scatter(x=t, y= i_data, name= "Prevalence of " +dist[count]+" GI", 
                   line_color=color[count]))
    count+=1


#-----STYLING--------
compare_infected.update_layout(
    title="Expected Infection Curve in the UK", xaxis_title="Days",
    yaxis_title="Percentage of Population<br>Presently Infected",
    legend_title="Legend", font_size=14,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))


# In[16]:


#pull uk total cases data
prev_uk_df = pd.read_csv('data/prevalence_uk.csv')

infected_real = go.Figure()

#-----ZOOMED IN----
d_count=52
dates_added = ["Jan "+str(i)+", 2020" for i in range(29,32)]+              ["Feb "+str(i)+", 2020" for i in range(1,15)]+              list(prev_uk_df['date'])


period=180
count=0
for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    fig = SInR_plot(y_t0, t, N, 1, 0, b, l)
    
    i_data = list(fig['data'][1]['y'])
    i_data = [i*N for i in i_data]
    
    infected_real.add_trace(
        go.Scatter(x=dates_added[:d_count+17], y=i_data[:d_count+17],
                   name= "Estimated Prevalence of " +dist[count]+" GI", 
                   line_color=color[count]))
    count+=1



infected_real.add_trace(
    go.Scatter(x=prev_uk_df['date'][:d_count], y=prev_uk_df['prevalence'][:d_count],
              name= "Recorded Prevalence in UK",line_color="black"))



infected_real.add_annotation(x="Mar 23, 2020", y=200000, xref="x", yref="y",
                             text="First <br> Official <br> Lockdown", showarrow=False,
                             font=dict(size=14,color="black"), align="center",
                             bordercolor="#636363", borderwidth=2, borderpad=4,
                             bgcolor="#C7C7C7", opacity=1)


infected_real.add_vline(x="Mar 23, 2020", line_width=2, line_dash="dash", line_color="grey")

#-----STYLING--------
infected_real.update_layout(
    title="Expected vs. Real Prevalence of Covid in the UK",
    xaxis_title="Date", yaxis_title="Prevalence", legend_title="Legend",
    font=dict(size=13),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))


# In[18]:


infected_real_zoom = go.Figure()


#-----ZOOMED IN----
d_count=40
dates_added = ["Jan "+str(i)+", 2020" for i in range(29,32)]+              ["Feb "+str(i)+", 2020" for i in range(1,15)]+              list(prev_uk_df['date'])
lockdown = []
zoom_1 = 35
period=180
count=0
for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    fig = SInR_plot(y_t0, t, N, 1, 0, b, l)
    
    i_data = list(fig['data'][1]['y'])
    i_data = [i*N for i in i_data]
    
    lockdown.append(i_data[zoom_1+19])
    
    
    infected_real_zoom.add_trace(
        go.Scatter(x=dates_added[zoom_1+17:d_count+17], y=i_data[zoom_1+17:d_count+17],
                   name= "Estimated Prevalence Given " +dist[count]+" GI", 
                   line_color=color[count],mode='lines'))
    count+=1



infected_real_zoom.add_trace(
    go.Scatter(x=prev_uk_df['date'][zoom_1:d_count], y=prev_uk_df['prevalence'][zoom_1:d_count],
              name= "Recorded Prevalence in UK", line_color="black", mode='lines'))



infected_real_zoom.add_annotation(x="Mar 23, 2020", y=22500, xref="x", yref="y",
                             text="First <br> Official <br> Lockdown", showarrow=False,
                             font=dict(size=14,color="black"), align="center",
                             bordercolor="#636363", borderwidth=2, borderpad=4,
                             bgcolor="#C7C7C7", opacity=1)


infected_real_zoom.add_vline(x="Mar 23, 2020", line_width=2, line_dash="dash", line_color="grey")

#-----STYLING--------
infected_real_zoom.update_layout(
    title="Expected vs. Real Prevalence of Covid in the UK",
    xaxis_title="Date", yaxis_title="Prevalence", legend_title="Legend",
    font=dict(size=13), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

print(lockdown)

infected_real_zoom.show()


# <br><br><br><br>
# 
# - - - - - - - -
# 
# #### Modifying contact rate $c$
# This is to see if changing the contact rate will better track the disease

# In[19]:


#pull uk total cases data
prev_uk_df = pd.read_csv('data/prevalence_uk.csv')

infected_real = go.Figure()


#-----ZOOMED IN----
dates_added = ["Jan "+str(i)+", 2020" for i in range(29,32)]+              ["Feb "+str(i)+", 2020" for i in range(1,15)]+              list(prev_uk_df['date'])
lockdown_c = []

period=180
count=0
d_count=52
zoom_c1 = 10
for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    
    contact=0.97
    fig = SInR_plot(y_t0, t, N, contact, 0, b, l)
    
    i_data = list(fig['data'][1]['y'])
    i_data = [i*N for i in i_data]
    
    infected_real.add_trace(
        go.Scatter(x=dates_added[zoom_c1+17:d_count+17], y=i_data[zoom_c1+17:d_count+17],
                   name= "Estimated Prevalence of " +dist[count]+" GI", 
                   line_color=color[count], mode='lines'))
    
    lockdown_c.append(i_data[zoom_1+19])
    count+=1


infected_real.add_trace(
    go.Scatter(x=prev_uk_df['date'][zoom_c1:d_count], y=prev_uk_df['prevalence'][zoom_c1:d_count],
              name= "Recorded Prevalence in UK", line_color="black", mode='lines'))



infected_real.add_annotation(x="Mar 23, 2020", y=80000, xref="x", yref="y",
                             text="First <br> Official <br> Lockdown", showarrow=False,
                             font=dict(size=14,color="black"), align="center",
                             bordercolor="#636363", borderwidth=2, borderpad=4,
                             bgcolor="#C7C7C7", opacity=1)


infected_real.add_vline(x="Mar 23, 2020", line_width=2, line_dash="dash", line_color="grey")

#-----STYLING--------
infected_real.update_layout(
    title="Expected vs. Real Prevalence of Covid in the UK where c=0.97",
    xaxis_title="Date", yaxis_title="Prevalence", legend_title="Legend",
    font=dict(size=13), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
infected_real.show()


print([round(l,2) for l in lockdown_c])


# In[20]:


period=180
count=0

SInR_C_compare_real = go.Figure()
dash = ["dot","dash"]
contact = 0.97

for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    #SIR MODEL
    if count == 0:
        fig = SInR_plot(y_t0, t, N, contact, 0, b, l)
        
        s_data = list(fig['data'][0]['y'])
        i_data = list(fig['data'][1]['y'])
        r_data = list(fig['data'][2]['y'])
        
        SInR_C_compare_real.update_layout(title="SI\u2099R Model for all Generation Intervals Contact=0.97",
                         legend=dict(yanchor="top", y=0.65, xanchor="left",x=0.01), font=dict(size=13))
        
        SInR_C_compare_real.add_trace(go.Scatter(x=t, y= s_data, name= "Susceptible: " +dist_string[count]+" GI", 
                                 line_color="blue",))
        SInR_C_compare_real.add_trace(go.Scatter(x=t, y= i_data, name= "Infected:     " +dist_string[count]+" GI", 
                                 line_color="red",))
        SInR_C_compare_real.add_trace(go.Scatter(x=t, y= r_data, name= "Recovered:  " +dist_string[count]+" GI", 
                                 line_color="green"))
        
        
        count+=1
    
    else:
        fig2 = SInR_plot(y_t0, t, N, contact, 0, b, l)
        s_data = list(fig2['data'][0]['y'])
        i_data = list(fig2['data'][1]['y'])
        r_data = list(fig2['data'][2]['y'])
        
        SInR_C_compare_real.add_trace(go.Scatter(x=t, y= s_data, name= "Susceptible: " +dist_string[count]+" GI", 
                                 line_color="blue", line_dash=dash[count-1]))
        SInR_C_compare_real.add_trace(go.Scatter(x=t, y= i_data, name= "Infected:     " +dist_string[count]+" GI", 
                                 line_color="red", line_dash=dash[count-1]))
        SInR_C_compare_real.add_trace(go.Scatter(x=t, y= r_data, name= "Recovered:  " +dist_string[count]+" GI", 
                                 line_color="green", line_dash=dash[count-1]))
       
        
        count+=1
        
SInR_C_compare_real.update_xaxes(title_text='Time (Days)')#,nticks=20)
SInR_C_compare_real.update_yaxes(title_text='Percentage of the Population')
SInR_C_compare_real.show()


# In[21]:


#pull uk total cases data
var_c = go.Figure()


#-----ZOOMED OUT----
period=600
count=0
c_vals = [round(i,1) for i in list(np.linspace(1,0.1,10))]
for c in c_vals:
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    fig = SInR_plot(y_t0, t, N, c, 0, b_n[0], l_n[0])
    
    i_data = list(fig['data'][1]['y'])
    
    var_c.add_trace(
        go.Scatter(x=t, y= i_data, name= "c="+str(c), ))
                  #line_color=color[count]))
    count+=1


#-----STYLING--------
var_c.update_layout(
    title="Changing Contact Rate c values of Gamma", xaxis_title="Days",
    yaxis_title="Percentage of Population<br>Presently Infected",
    legend_title="Legend", font_size=14,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.87))


# <br><br><br>
# 
# ##### Estimating effective $c$

# In[22]:


#----------- SInR WITH I compartment Seperated -----------
def SInR_Is(y_t0, t, N, c, h, beta, lambda_n):
# Integrate the SIR equations over the time grid, t.
    ret = odeint(SInR, y_t0, t, args=(N, c, h, beta, lambda_n))

    S = ret.T[0]
    I = ret.T[1:-1]
    R = ret.T[-1]

    return S,I,R


# In[31]:


#Find number of people in each compartment on the day of lockdown (54 days into pandemic)
prior_contact = 0.97
lock_day = 54
placement_lockdown1 = []
for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))
    
    S_c,I_c, R_c, = SInR_Is(y_t0, t, N, prior_contact, 0, b, l)
    
    I_lock = [item[lock_day] for item in list(I_c)]
    
    placement_lockdown1.append([S_c[54]]+I_lock+[R_c[54]])
    
ld_contact1 = 0
lock_day = 54
placement_lockdown1 = []
for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))
    
    S_c,I_c, R_c, = SInR_Is(y_t0, t, N, prior_contact, 0, b, l)
    
    I_lock = [item[lock_day] for item in list(I_c)]
    
    placement_lockdown1.append([S_c[54]]+I_lock+[R_c[54]])
#Place these people into the compartments 


# In[32]:


lockdown_fig = go.Figure()
lockdown_contact = 0.65
count = 0

zoom_val = 40

for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    y_t0 = placement_lockdown1[0]
    t = np.array(list(range(0,period+1)))
    
    fig = SInR_plot(y_t0, t, N, lockdown_contact, 0, b, l)
    
    i_data = list(fig['data'][1]['y'])
    i_data = [i*N for i in i_data]
    
    
    lockdown_fig.add_trace(
        go.Scatter(x=prev_uk_df['date'][37:37+zoom_val], y=i_data[:zoom_val], name= "Estimated Prevalence of " + dist_string[count]+" GI", 
                   line_color=color[count], mode='lines'))
    
    count+=1
    
lockdown_fig.add_trace(
    go.Scatter(x=prev_uk_df['date'][37:37+zoom_val], y=prev_uk_df['prevalence'][37:37+zoom_val],
              name= "Recorded Prevalence in UK", line_color="black", mode='lines'))


#-----STYLING--------
lockdown_fig.update_layout(
    title="Lockdown Expected vs. Real Prevalence of Covid in the UK",
    xaxis_title="Date", yaxis_title="Prevalence", legend_title="Legend",
    font=dict(size=13), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.1))

lockdown_fig.show()


# In[ ]:





# In[ ]:





# <br><br>
# 
# 
# ### SI$_n$RS modeling
# 
# After examining the SI$_n$R models, we can explore the SI$_n$RS model.

# In[28]:


period=700
count=0
holding_time = 1/122 #waning imunity of 4 months

for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    #SIR MODEL
    fig = SInR_plot(y_t0, t, N, 1, holding_time, b, l)
    fig.update_layout(title="SI\u2099RS Model of "+dist[count]+" Generation Interval",
                     legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), 
                     font_size=14)
    fig.show()
    count+=1
    


# In[29]:


period=700
count=0

SInR_C_compare = go.Figure()
dash = ["dot","dash"]

for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    #SIR MODEL
    if count == 0:
        fig = SInR_plot(y_t0, t, N, 1, holding_time, b, l)
        
        s_data = list(fig['data'][0]['y'])
        i_data = list(fig['data'][1]['y'])
        r_data = list(fig['data'][2]['y'])
        
        SInR_C_compare.add_trace(go.Scatter(x=t, y= s_data, name= "Susceptible: " +dist[count]+" GI", 
                                 line_color="blue",))
        SInR_C_compare.add_trace(go.Scatter(x=t, y= i_data, name= "Infected:  " +dist[count]+" GI", 
                                 line_color="red",))
        SInR_C_compare.add_trace(go.Scatter(x=t, y= r_data, name= "Recovered:  " +dist[count]+" GI", 
                                 line_color="green"))
        
        
        count+=1
    
    else:
        fig2 = SInR_plot(y_t0, t, N, 1, holding_time, b, l)
        s_data = list(fig2['data'][0]['y'])
        i_data = list(fig2['data'][1]['y'])
        r_data = list(fig2['data'][2]['y'])
        
        SInR_C_compare.add_trace(go.Scatter(x=t, y= s_data, name= "Susceptible: " +dist[count]+" GI", 
                                 line_color="blue", line_dash=dash[count-1]))
        SInR_C_compare.add_trace(go.Scatter(x=t, y= i_data, name= "Infected:  " +dist[count]+" GI", 
                                 line_color="red", line_dash=dash[count-1]))
        SInR_C_compare.add_trace(go.Scatter(x=t, y= r_data, name= "Recovered:  " +dist[count]+" GI", 
                                 line_color="green", line_dash=dash[count-1]))
       
        
        count+=1
SInR_C_compare.update_xaxes(title_text='Time (Days)')#,nticks=20)
SInR_C_compare.update_yaxes(title_text='Percentage of the Population')


SInR_C_compare.update_layout(title="SI\u2099R Model for all Generation Intervals",
                             xaxis_range=['2020-01-30','2020-05-31'],
                             legend=dict(x=.1, y=-0.15),legend_orientation="h",
                             height=600, width=1050, font_size=14)

#SInR_C_compare.update_layout(title="SI\u2099R Model for all Generation Intervals",
#                         legend=dict(yanchor="top", y=0.6, xanchor="left",x=0.01))


SInR_C_compare.show()


# In[30]:


compare_infected = go.Figure()

#-----ZOOMED OUT----
period=500
count=0
for b,l in zip(b_n, l_n):
    n_val=len(b)
    N = 67886011
    R_t0 = 0
    I_t0 = [1]+(n_val-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,period+1)))

    fig = SInR_plot(y_t0, t, N, 1, holding_time, b, l)
    
    i_data = list(fig['data'][1]['y'])
    
    compare_infected.add_trace(
        go.Scatter(x=t, y=i_data,
                   name= "Estimated Prevalence given " +dist[count]+" GI", 
                   line_color=color[count]))
    count+=1


#-----STYLING--------
compare_infected.update_layout(
    title="Expected Infection Curve in the UK", xaxis_title="Days",
    yaxis_title="Percentage of Population <br>Presently Infected",
    legend_title="Legend", font=dict(size=12),
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    font_size=14)

