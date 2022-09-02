#!/usr/bin/env python
# coding: utf-8

# ## App Creation
# 
# First, import all necessary libraries:

# In[1]:


#App Libraries
import json
import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

#Distributions
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import weibull_min

#Calculation libraries
import math
import pandas as pd
import numpy as np
import ast
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats 
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.optimize import fsolve
#from sympy import symbols, Eq, solve

#Plot libraries
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# In[2]:


#==================================================================#
#                 CREATE GENERATION INTERVAL DATA                  #
#==================================================================#
def create_gi(pop_mean, sd, m):
    '''
    pop_mean: population mean of the standard deviation
    '''
    
    #Set seed for consistency:
    np.random.seed(1234)
    
    #=========GAMMA============    
    gamma_shape = (pop_mean**2)/(sd**2)
    gamma_scale = (sd**2)/(pop_mean)
    gi_gamma_obs = np.random.gamma(gamma_shape, gamma_scale, m)

    #=========LOGNORMAL============
    log_mean = pop_mean
    log_sd = sd
    log_var = log_sd**2
    norm_mean = np.log(log_mean)-0.5*np.log((log_sd/log_mean)**2+1) #scale=e^norm_mean
    norm_var = np.log((log_sd/log_mean)**2+1) 
    norm_sd = np.sqrt(norm_var) # equivalent to the shape
    gi_lognorm_obs = lognorm.rvs(s=norm_sd, scale=math.exp(norm_mean), size=m)


    #=========WEIBULL============
    weibull_mean = pop_mean
    weibull_std = sd

    def G(k): 
        return math.gamma(1+2/k)/(math.gamma(1+1/k)**2)
    def f(k,b):
        return G(k)-b #function solves for k

    b = (weibull_std**2)/(weibull_mean**2)+1
    
    init = 1 # The starting estimate for the root of f(x) = 0.
    weibull_shape = fsolve(f,init,args=(b))[0]
    weibull_scale = weibull_mean/math.gamma(1+1/weibull_shape)
    gi_weibull_obs = weibull_min.rvs(weibull_shape,scale=weibull_scale, size=m)
    
    return gi_gamma_obs, gi_lognorm_obs, gi_weibull_obs


#==================================================================#
#               VISUALIZE GENERATION INTERVAL DATA                 #
#==================================================================#

def gi_visualize(gi_gamma, gi_lognorm, gi_weibull):
    
    color=["skyblue","darkorange","green"]

    fig = make_subplots(rows=2, cols=2,)

    fig.append_trace(go.Histogram(x=gi_gamma, histnorm='percent', name='Gamma',
                                marker_color=color[0], opacity=1,),row=1,col=1)

    fig.append_trace(go.Histogram(x=gi_lognorm, histnorm='percent', name='Lognorm',
                                marker_color=color[1], opacity=1), row=1,col=2)

    fig.append_trace(go.Histogram(x=gi_weibull, histnorm='percent', name='Weibull',
                                marker_color=color[2], opacity=1), row=2,col=1)


    group_labels = ['Gamma Curve', 'Lognormal Curve', 'Weibull Curve']
    hist_data = [gi_gamma, gi_lognorm, gi_weibull]
    distplfig = ff.create_distplot(hist_data, group_labels, colors=color,
                             bin_size=.2, show_hist=False, show_rug=False)

    for k in range(len(distplfig.data)):
        fig.append_trace(distplfig.data[k],
        row=2, col=2
    )
    fig.update_layout(barmode='overlay')
    
    return(fig)



#==================================================================#
#                          OBJECTIVE FUNCTION                      #
#==================================================================#

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


#==================================================================#
#                         CONSTRAINT FUNCTION                      #
#==================================================================#
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


#==================================================================#
#              CALCULATE WEIGHTS, HOLDING PERIOD, RATES            #
#==================================================================#
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
    #        wl_0[i] = 1/n

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



#==================================================================#
#                          OBJECTIVE FUNCTION                      #
#==================================================================#
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



#==================================================================#
#                       EXPECTED INFECTIOUS CURVE                  #
#==================================================================#
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

    
#==================================================================#
#                 VISUALIZE EXPECTED INFECTIOUS CURVE              #
#==================================================================#
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


#==================================================================#
#               VISUALIZE EXPECTED INFECTIOUS CURVE                #
#==================================================================#
def plot_beta_dist(betas, lambdas):
    
    color=["skyblue","darkorange","green"]
    dist = ["Gamma", "Lognormal","Weibull"]
    count = 0

    fig = make_subplots(rows=1, cols=1)

    for beta,lamb in zip(betas,lambdas):
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
        title="Estimated Infectious Curve  of each Distribution", #&#x3B2;(&#x1D70F;)
        xaxis_title="Duration of Infection (days)",
        yaxis_title="Transmission Rate", legend_title="Legend", font=dict(size=14),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    
    return(fig)


#==================================================================#
#                          SInR(S) MODEL                           #
#==================================================================#
def SInR(y, t, N, c, h, beta, lambda_n):
    '''
    Inputs:
        y: initial condition vector (S, I, R)
        t: time points (in days)
        N: total population
        c: contact rate
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



#==================================================================#
#                      VISUALIZE SInR(S) MODEL                     #
#==================================================================#
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

    return(fig)


# In[3]:


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

##########################################################################
#                           HELPER FUNCTIONS                             #
##########################################################################
def create_dropdown_options(series):
    options = [{'label': i, 'value': i} for i in series.sort_values().unique()]
    return options
def create_dropdown_value(series):
    value = series.sort_values().unique().tolist()
    return value
def create_slider_marks(values): #NEED
    marks = {i: {'label': str(i)} for i in values}
    return marks


##########################################################################
#                    ADD ONS TO APP (images, files,etc)                  #
##########################################################################
#pull Gamma Data
gi_df = pd.read_excel ('GI_Values.xlsx')[["Source", "Mean","SD", "Dist_Type"]]
gi_df.sort_values(by=["Dist_Type",'Mean'], inplace=True)
gi_df.reset_index(drop=True, inplace=True)

colors = {'background': '#111111','text': 'black'}
subheader_size = 20

##########################################################################
#                             DASHBOARD APP                              #
##########################################################################
app.layout = html.Div(children=[
    dcc.Location(id="url",refresh=False),
    html.Div(id="output-div")
])

##########################################################################
#                               HOME PAGE                                #
##########################################################################
home_layout = html.Div(
    
    #==========Create "Header" Data================= 
    children=[
        
    html.Div(
        [
            html.Img(src=app.get_asset_url("edinburgh.png"), height=50),
            html.Div([html.H4(children=('SI',html.Sub("n"),'R Covid-19 Modeling'), style={'textAlign': 'center', "font-weight": "bold"}),]),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
    ),
    
    #---Page Links---
    html.Div([dbc.Row(
    [
        dbc.Col(html.Div(dcc.Link('Home',href="/")), style={'textAlign': 'center'}),
        dbc.Col(html.Div(dcc.Link('Simulate Data',href="/simulate")), style={'textAlign': 'center'}),
        dbc.Col(html.A("Github Code", href='https://github.com/annette-bell/SInR-Covid-Dissertation', 
                       target="_blank"), style={'textAlign': 'center'}),
    ],
    className="g-0",
    )]),
    
    #---Line Break---    
    html.Div([html.Hr(style={'borderWidth': "0.3vh", "color": "#FEC700"}),]),
    
    #===============Home Page Information==========
    html.Div(
            [
                html.H6("About this app", style={"margin-top": "0","font-weight": "bold","text-align": "left"}),
                #html.Hr(),
                html.P("The Susceptible-Infected-Recovered (SIR) compartmental model is used in epidemiology to identify\
                        and categorize members of a population based on their status with regards to a disease. Less\
                        studied variations of this problem are the SInR and SInRS models. These models, which have applications\
                        in latent infection and varying transmission rates, will be used on three different generation\
                        interval—the time between primary exposure and secondary infection—distributions: gamma, lognormal,\
                        and Weibull. The distributions are ultimately tested against one another to see not only \
                        which provides most realistic data, but how these data-sets interact.\
                        This app is meant to help people understand dynamics of COVID-19 modeling through a simply dashboard application.\
                        To see a more in-depth explanation, please see the Github repository which includes my dissertation.", 
                       className="control_label",style={"text-align": "justify"}),               
            ],
            className="pretty_container almost columns",
        ),        
        
        
    #============AUTHOR=============
    html.Div(
            [
                html.H6("Authors", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                
                html.P("Annette Bell (nettebell97@gmail.com)", style={"text-align": "center", "font-size":"10pt"}),
                
            ],
            className="pretty_container almost columns",
        ),
        
    #============ACKNOWLEDGEMENTS=============
    html.Div(
            [
                html.H6("Acknowledgements", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                
                html.P("John Dagpunar: Dr. Dagpunar was my thesis advisor and extremely helpful throughout the project.)", style={"text-align": "left", "font-size":"10pt"}),
                
            ],
            className="pretty_container almost columns",
        ),
        
    #============SOURCES=============
    html.Div(
            [
                html.H6("Sources", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                dcc.Markdown(
                    """\
                         - Code Layout was used from Plotly public dash application: https://dash.gallery/dash-food-consumption/
                         - I examined another dash application to better understand how to use it. In addition to dash application resources, I analyzed the source code to clarify how to implement dash: https://github.com/FranzMichaelFrank/health_eu
                         - Edinburgh PNG: https://uploads-ssl.webflow.com/5eb13d58c8c08b73773d6b1c/600ea3810bde89c60e317be7_uni-of-edinburgh.png
                        """
                ,style={"font-size":"10pt"}),
                
            ],
            className="pretty_container almost columns",
        ),
        
    
    ])



##########################################################################
#                           SIMULATION PAGE                             #
##########################################################################
sim_layout = html.Div(
    
    #==========Create "Header" Data================= 
    children=[
    
    #---Title---
    html.Div(
        [
            html.Img(src=app.get_asset_url("edinburgh.png"), height=50),
            html.Div([html.H4(children=('SI',html.Sub("n"),'R Covid-19 Modeling'), style={'textAlign': 'center', "font-weight": "bold"}),]),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
    ),
    
    #---Page Links---
    html.Div([dbc.Row(
    [
        dbc.Col(html.Div(dcc.Link('Home',href="/")), style={'textAlign': 'center'}),
        dbc.Col(html.Div(dcc.Link('Simulate Data',href="/simulate")), style={'textAlign': 'center'}),
        dbc.Col(html.A("Github Code", href='https://github.com/annette-bell/SInR-Covid-Dissertation', 
                       target="_blank"), style={'textAlign': 'center'}),
    ], className="g-0",
    )]),
    
    #---Line Break---    
    html.Div([html.Hr(style={'borderWidth': "0.3vh", "color": "#FEC700"}),]),
        
    #============OVERIEW OF THE SIMULATION DATA=================
        html.Div(
            [
                html.H6(["Overview of Distributions", html.Br()], style={"margin-top": "0","font-weight": "bold","text-align": "left"}),

                html.Div(
                [
                dbc.Row(
                    [#---Table of Previous Analysis---:
                        dbc.Col(dash_table.DataTable(gi_df.to_dict('records'), [{"name": i, "id": i} for i in gi_df.columns], 
                                                     style_header={'text-align': 'center', 'fontWeight': 'bold',},
                                                     style_table={'height': '200px', 'overflowY': 'auto'},
                                                     style_cell={'textAlign': 'left'},), width=3),

                    #---Commentary on the table
                        dbc.Col(html.Div([html.P("There are three main commonly used distributions to model the generation interval-\
                        the time from primary exposure to secondary infectiousness. These distributions include gamma, weibull, and log-normal.\
                        To the left, you can see a table of means and standard deviations from others previous work.",className="control_label",style={"text-align": "justify"}),]))]),]),
            ],
            className="pretty_container",
        ),   
        
    #================= GENERATION INTERVAL SIMULATION =====================
        html.Div(
            [   
                html.Div(
                    #----------------INPUTS-----------
                    [
                    html.H6("Generation Interval Distribution:", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                    html.P("Please select the distribution you wish to base your simulated generation interval data off of. Note: Seed=1234.", className="control_label",style={"text-align": "justify"}),
                    html.Br(),
                        
                    #Shows the inputs for the specified distribution
                    html.Div(
                        [ 
                            
                            dbc.Row([
                                dbc.Col(html.Div([
                                    #---Mean---
                                    html.P("Input the population mean:", className="control_label", style={"font-weight": "bold", "text-align": "left"}),
                                    dcc.Input(id='pop_mean',placeholder='', type='number', value= 4.9),

                                ])),
                                dbc.Col(html.Div([
                                    #---SD---
                                    html.P("Input the standard deviation:", className="control_label", style={"font-weight": "bold", "text-align": "left"}),
                                    dcc.Input(id='stan_dev',placeholder='', type='number', value= 2),

                                ])),
                            ],),
                            
                            
                            #---Data Set Size---
                            html.P("Select size of data:", className="control_label", style={"font-weight": "bold", "text-align": "center"}),
                            dcc.Slider(id='gi_size', min=1000, max=10000, step=500, value=5000,
                                       marks=create_slider_marks(list(range(1000,10001))[::1000])),

                            html.Br(),
                            
                            #---Update Button---
                            html.Button(id='update_button', children="Simulate", n_clicks=0,style=dict(width='220px')),
                        ],),
                        
                        
                    ],
                    className="pretty_container four columns",
                ),

                #----------------GI Plot-----------
                html.Div(
                    [   
                        html.H6("Generation Interval Simulations", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                     
                        #---Information Regarding Shape and Scale---
                        html.P(id='shape_scale', style={"text-align": "justify"}),
                     
                        #---GI Histogram---
                        html.Div(id='gammaplot_container',children=[
                        dcc.Graph(id="gi_plots", style={'height': '80vh'}),
                        ]),
                    
                    ],
                    
                    className="pretty_container seven columns",
                ),

                
            ],
            className="row flex-display",
        ),
    
        
        
    #===============Transmission Rate==========
    html.Br(),
    html.Div([dbc.Row(
    [
        dbc.Col(html.Div(html.Hr(style={'borderWidth': "0.3vh", "color": "#FEC700", 'align':'right'},)), width=5),
        dbc.Col(html.Div(html.H6("Transmission Rate", style={"text-align": "center", "font-weight": "bold",
                                                            "margin-top": "14px", "color": "#384142"})), width=1.5),
        dbc.Col(html.Div(html.Hr(style={'borderWidth': "0.3vh", "color": "#FEC700", 'align':'left'},)), width=5),
    ],
    )]),        
    
    html.Div(
            [   
                html.Div(
                    [#----------------Parameters of Beta(u)-----------
                        html.H6("Parameters of \u03b2(u):", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                        html.P("As the transmission rate is not constant, we create a function that simulates transmission a non constant transmission rate.", className="control_label",style={"text-align": "justify"}),

                        #---R_0---
                        html.P("Input the basic reproduction number:", className="control_label", style={"font-weight": "bold", "text-align": "center"}),
                        dcc.Input(id='R0', placeholder='', type='number', value= 2.3),
                                               
                        html.Br(),
                        html.Br(),
                        
                        #---Update Button---
                        html.Button(id='beta_update_button', children="Calculate B(u)", n_clicks=0, style=dict(width='220px')),                
                        
                        ], className="pretty_container four columns",
                ),

                
                html.Div(
                    [#----------------Beta(u) Plot-----------
                        html.H6("Expected Infectious Curve", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                        #html.P("Visualize the transmission rates. Note: the following results are based on the parameters and the GI Data simulated above.", className="control_label",style={"text-align": "justify"}),
                        
                        html.Br(),
                        
                        #---return weights and betas---
                        #html.P(id='weights_beta_info', style={"text-align": "justify"}), 
                        #html.P(id='lambdas', style={"text-align": "justify"}), 
                        #html.P(id='weights', style={"text-align": "justify"}), 
                        #html.P(id='betas', style={"text-align": "justify"}), 
                        
                        dcc.Graph(id="beta_plot", style={'height': '60vh'}),
                    ],
                    className="pretty_container seven columns",
                ),                
            ],
            className="row flex-display",
        ), 
    
    
    #=====================SI(n)R Model=====================
        
    html.Br(),
    html.Div([dbc.Row(
    [
        dbc.Col(html.Div(html.Hr(style={'borderWidth': "0.3vh", "color": "#FEC700", 'align':'right'},)), width=5),
        dbc.Col(html.Div(html.H6("Modeling COVID-19", style={"text-align": "center", "font-weight": "bold",
                                                            "margin-top": "14px", "color": "#384142"})), width=1.5),
        dbc.Col(html.Div(html.Hr(style={'borderWidth': "0.3vh", "color": "#FEC700", 'align':'left'},)), width=5),
    ],
    )]),
        
    
    html.Div(
            [   
                html.Div(
                    [#----------------Parameters of SInR Model-----------
                        html.H6("Parameters of the Model:", style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                        #html.P("Beta(u) was last calculated using the following:", className="control_label",style={"text-align": "justify"}),
                        
                        html.Div([
                        dcc.RadioItems(
                            id='distribution-dropdown',
                            options=[
                                {'label': 'gamma', 'value': 'gamma'},
                                {'label': 'weibull', 'value': 'weibull'},
                                {'label': 'log-normal','value': 'lognorm'},
                                {'label': 'all','value': 'all'}],
                            
                            value='all',
                            
                            labelStyle={"display": "inline-block"},
                            style={"font-weight": "bold", "text-align": "center"},
                        ),],),
                        
                        dbc.Row(
                            [
                                dbc.Col(html.Div([
                                     #---Total Population---
                                    html.P("Total Population (N):", className="control_label", style={"font-weight": "bold", "text-align": "left"}),
                                    dcc.Input(id='N_size', placeholder='', type='number', value = 67886011),
                                ])),
                                dbc.Col(html.Div([
                                    #---simulated days---
                                    html.P("Total Days to Simulate Over:", className="control_label", style={"font-weight": "bold", "text-align": "left"}),
                                    dcc.Input(id='t_days', placeholder='', type='number', value = 180),

                                ])),
                                
                            ], className="g-0",
                            ),
                        
                        dbc.Row(
                            [
                                dbc.Col(html.Div([
                                    #---Recovered---
                                    html.P("Initial Recovered (R):", className="control_label", style={"font-weight": "bold", "text-align": "left"}),
                                    dcc.Input(id='R_size', placeholder='', type='number', value = 0),

                                ])),
                                dbc.Col(html.Div([
                                    #---simulated days---
                                    html.P("Contact Rate:", className="control_label", style={"font-weight": "bold", "text-align": "left"}),
                                    dcc.Input(id='c', placeholder='', type='number', value = 1),

                                ])),
                            ], className="g-0",
                            ),
                        
                        dbc.Row(
                            [
                                dbc.Col(html.Div([
                                    #---Infected---
                                    html.P("Initail Infected (I):", className="control_label", style={"font-weight": "bold", "text-align": "left"}),
                                    dcc.Input(id='I_size', placeholder='', type='number', value = 1),
                        
                                ])),
                                dbc.Col(html.Div([
                                    #---simulated days---
                                    html.P("Waning Immunity Rate:", className="control_label", style={"font-weight": "bold", "text-align": "left"}),
                                    dcc.Input(id='h', placeholder='', type='number', value = 0),

                                ])),
                            ], className="g-0",
                            ),
                        
                        #---n_slider---
                        html.P("Select the compartment size: ", className="control_label", style={"font-weight": "bold", "text-align": "center"}),
                        dcc.Slider(1, 20, step=1, value=10, id='n_val'),
                        
                        #---SInR Button---
                        html.Br(),
                        html.Br(),
                        html.Button(id='model_button', children="Model", n_clicks=0, style=dict(width='220px')),
                        
                        ], className="pretty_container four columns",
                ),

                
                html.Div(
                    [#----------------SInR Plot-----------
                        html.H6(('SI',html.Sub("n"),'R Covid-19 Modeling'), style={"margin-top": "0","font-weight": "bold","text-align": "center"}),
                        html.P(id='model_parameters', style={"text-align": "justify"}), 
                        #html.P("Visualize the how th population shifts.", className="control_label",style={"text-align": "justify"}),
                        
                        dcc.Graph(id="SInR_plot", style={'height': '60vh'}),
                        
                        html.Div([ ], id='plot1'),
                    ],
                    className="pretty_container seven columns",
                ),                
            ],
            className="row flex-display",
        ),
        
    
        
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


        
##########################################################################
#                          LINK TO EACH PAGE                             #
##########################################################################
@app.callback(
    Output(component_id="output-div",component_property="children"),
    Input(component_id="url",component_property="pathname"))

def update_page_layout(pathname):
    if pathname == "/simulate":
        return  sim_layout
    else:
        return home_layout

    
        
##########################################################################
#             SIMULATE AND VISUALIZE DISTRIBUTION DATA SETS              #
##########################################################################
@app.callback(
    [Output('gi_plots', 'figure'),
     Output('shape_scale', 'children')],
    [Input(component_id='update_button', component_property='n_clicks')],
    [State('stan_dev', 'value'),
    State('pop_mean', 'value'),
    State('gi_size', 'value')]
)

def update_sim_gi(n_clicks, sd, mean, size):
    '''
    This callback and function combination simulates
    a desired distribution (either gamma, weibull, or log-normal) 
    given the information.
    '''
    #----------CREATE DISTRIBUTIONS---------
    gamma_data, lognorm_data, weibull_data = create_gi(mean, sd, size)
    
    mean_vals = [np.mean(gamma_data), np.mean(lognorm_data), np.mean(weibull_data)]
    std_vals = [np.std(gamma_data), np.std(lognorm_data), np.std(weibull_data)]
    
    #--------------VISUALIZE----------------
    gi_fig = gi_visualize(gamma_data, lognorm_data, weibull_data)
    


    return gi_fig, f'Given the input mean and standard deviation of {mean} and {sd} respectively,    the distributions are as follows: Gamma (x\u0305={round(mean_vals[0],3)}, s={round(std_vals[0],3)}).    Lognormal(x\u0305 ={round(mean_vals[1],3)}, s={round(std_vals[1],3)}).    Weibull(x\u0305={round(mean_vals[2],3)}, s={round(std_vals[2],3)}).'


##########################################################################
#                         CREATE AND PLOT BETA(u)                        #
##########################################################################
@app.callback(
    [Output('beta_plot', 'figure')],  #Output('weights_beta_info', 'children'), Output('lambdas', 'children'), Output('weights', 'children'), Output('betas', 'children')],
    [Input(component_id='beta_update_button', component_property='n_clicks')],
    [State('R0', 'value'),  #DISTRIBUTION STATES
     State('stan_dev', 'value'),
     State('pop_mean', 'value'),
     State('gi_size', 'value'),
    ]
)

def update_beta_u_plot(n_click, R0, sd, mean, size):
    '''
    Function will run beta_u function once "Calculate Beta(u)" button is clicked
    '''
    
    #----------CREATE DISTRIBUTIONS---------
    gamma_data, lognorm_data, weibull_data = create_gi(mean, sd, size)
    
    #----determine mimnimal acceptable size -----
    g_min_comp = math.ceil((np.mean(gamma_data)**2)/(np.var(gamma_data)))
    l_min_comp = math.ceil((np.mean(lognorm_data)**2)/(np.var(lognorm_data)))
    w_min_comp = math.ceil((np.mean(weibull_data)**2)/(np.var(weibull_data)))
    
    min_acceptable = max(g_min_comp, l_min_comp, w_min_comp)
        
    #----------------CALC VALS------------------
    w_gamma, l_gamma, b_gamma = solver(gamma_data, R0, min_acceptable, "gamma")
    w_lognorm, l_lognorm, b_lognorm = solver(lognorm_data, R0, min_acceptable, "lognorm")
    w_weibull, l_weibull, b_weibull = solver(weibull_data, R0, min_acceptable, "weibull")
    
    #----------------PLOT Beta(u)------------------
    b_n = [b_gamma, b_lognorm, b_weibull]
    l_n = [l_gamma, l_lognorm, l_weibull]
    w_n = [w_gamma, w_lognorm, w_weibull] 
    
    beta_plot = plot_beta_dist(b_n, l_n)
  

    return [go.Figure(data=beta_plot)]


##########################################################################
#                           UPDATE SInR MODEL                            #
##########################################################################
@app.callback(
    [Output('SInR_plot', 'figure'),],#Output(component_id="plot1", component_property="children"),],
    #Output(component_id='model_parameters', component_property='children')],
    [Input(component_id='model_button', component_property='n_clicks')],
    [State('stan_dev', 'value'),
    State('pop_mean', 'value'),
    State('gi_size', 'value'),
    State('distribution-dropdown', 'value'),
    State('N_size', 'value'),
    State('I_size', 'value'),
    State('R_size', 'value'),
    State('t_days', 'value'),
    State('n_val', 'value'),
    State('R0', 'value'),
    State('c', 'value'),
    State('h', 'value')]
)


def update_SInR_plot(n_click, sd, mean, size, show, N, I1_t0, R_t0, days, n, R0, c_val, h_val):
    '''
    Visualize the SInR(S) plot
    '''
    
    gamma_data, lognorm_data, weibull_data = create_gi(mean, sd, size)

    #----determine mimnimal acceptable size -----
    g_min_comp = math.ceil((np.mean(gamma_data)**2)/(np.var(gamma_data)))
    l_min_comp = math.ceil((np.mean(lognorm_data)**2)/(np.var(lognorm_data)))
    w_min_comp = math.ceil((np.mean(weibull_data)**2)/(np.var(weibull_data)))

    #----------------CALC VALS------------------
    w_gamma, l_gamma, b_gamma = solver(gamma_data, R0, n, "gamma")
    w_lognorm, l_lognorm, b_lognorm = solver(lognorm_data, R0, n, "lognorm")
    w_weibull, l_weibull, b_weibull = solver(weibull_data, R0, n, "weibull")

    #-------Create lists of data-----
    b_n = [b_gamma, b_lognorm, b_weibull]
    l_n = [l_gamma, l_lognorm, l_weibull]
    w_n = [w_gamma, w_lognorm, w_weibull]
    
    print(b_n)
    
    color=["skyblue","darkorange","green"]
    dist = ["Gamma", "Lognormal","Weibull"]
    count=0
        
    #----specify compartments-------
    I_t0 = [I1_t0]+(n-1)*[0]
    S_t0 = N - sum(I_t0) - R_t0
    y_t0 = [S_t0]+ I_t0 +[R_t0]
    t = np.array(list(range(0,days+1)))
    
    #----specify model type-----
    if h_val== 0:
        model_type = "SI\u2099R Model of "
    else:
        model_type = "SI\u2099RS Model of "
        
    if show == "all":       
        SInR_compare = go.Figure()
        dash = ["dot","dash"]

        for b,l in zip(b_n, l_n):
            #SIR MODEL
            if count == 0:
                fig = SInR_plot(y_t0, t, N, c_val, h_val, b, l)

                s_data = list(fig['data'][0]['y'])
                i_data = list(fig['data'][1]['y'])
                r_data = list(fig['data'][2]['y'])

                SInR_compare.update_layout(title=model_type+" for all Generation Intervals",
                                 legend=dict(yanchor="top", y=-0.2, xanchor="left",x=0.02, orientation="h"), 
                                           font_size=14)

                SInR_compare.add_trace(go.Scatter(x=t, y= s_data, name= "Susceptible: " +dist[count]+" GI", 
                                         line_color="blue",))
                SInR_compare.add_trace(go.Scatter(x=t, y= i_data, name= "Infected:     " +dist[count]+" GI", 
                                         line_color="red",))
                SInR_compare.add_trace(go.Scatter(x=t, y= r_data, name= "Recovered:  " +dist[count]+" GI", 
                                         line_color="green"))


                count+=1

            else:
                fig2 = SInR_plot(y_t0, t, N, c_val, h_val, b, l)
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
        
        return [go.Figure(data=SInR_compare)]
    
    else:
        if show == "gamma":
            index = 0
        if show == "lognorm":
            index = 1
        else:
            index=2
        
        I_t0 = [I1_t0]+(n-1)*[0]
        S_t0 = N - sum(I_t0) - R_t0
        y_t0 = [S_t0]+ I_t0 +[R_t0]
        t = np.array(list(range(0,days+1)))

        #SIR MODEL
        fig = SInR_plot(y_t0, t, N, c_val, h_val, b_n[index], l_n[index])
        fig.update_layout(title=model_type+" of "+dist[index]+" Generation Interval",
                         legend=dict(yanchor="top", y=0.35, xanchor="left", x=0.01),font_size=14)
        return [go.Figure(data=fig)]


##########################################################################
#                             RUN THE APP                                #
##########################################################################
if __name__ == "__main__":
    app.run_server(debug=False)
    
    
    
#TABLE STYLING:
#https://dash.plotly.com/datatable/style
    


# <br><br><br><br>
