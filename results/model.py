from math import floor, exp, sqrt, pi
import cmath
import numpy
from numpy import e, cos, zeros, arange, roll, where, random, ones, mean, reshape, dot, array, flipud, pi, exp, dot, angle, degrees, shape, linspace
import matplotlib.pyplot as plt
from itertools import chain
import scipy
from scipy import special
import numpy as np 
import seaborn as sns
import time
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.signal
from scipy.optimize import curve_fit 
import pandas as pd
##import sdepy as sd

## Fucntions to use


def decode_rE(rE, a_ini=0, a_fin=360, N=512):
    #Population vector for a given rE
    # return ( angle in radians, absolut angle in radians, abs angle in degrees )
    N=len(rE)
    Angles = np.linspace(a_ini, a_fin, N) 
    angles=np.radians(Angles)
    rE = np.reshape(rE, (1,N))
    R = numpy.sum(np.dot(rE,exp(1j*angles)))/numpy.sum(rE)
    
    angle_decoded = np.degrees(np.angle(R))
    if angle_decoded<0:
        angle_decoded = 360+angle_decoded
    
    return angle_decoded
    #Mat.append(  [angle(R), abs(angle(R)) , degrees(abs(angle(R)))]  )
    #return round( np.degrees(abs(np.angle(R))), 2)


def circ_dist(a1,a2):
    ## Returns the minimal distance in angles between to angles 
    op1=abs(a2-a1)
    angs=[a1,a2]
    op2=min(angs)+(360-max(angs))
    options=[op1,op2]
    return min(options)


def most_frequent(List): 
    dict = {} 
    count, itm = 0, '' 
    for item in reversed(List): 
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count : 
            count, itm = dict[item], item 
    return(itm) 


def Interference_effects(target, response, reference):
    #input list of target, list of responses and list of references
    #Error_interference; positive for attraction and negative for repulsion
    #######
    #Decimals to get
    decimals=2
    ####
    interferences=[]
    for i in range(0, len(target)):
        angle_err_abs=abs(target[i] - response[i])
        if circ_dist(np.array(response)[i], np.array(reference)[i])<=circ_dist(np.array(target)[i], np.array(reference)[i]):
            Err_interference=round( angle_err_abs, decimals) 
        else:
            Err_interference=round( -angle_err_abs, decimals)
        interferences.append(Err_interference)
    
    return interferences


def viz_polymonial(X, y, poly_reg, pol_reg):
    plt.figure()
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Fit Bump')
    plt.xlabel('Neuron')
    plt.ylabel('rate')
    plt.show(block=False)
    return


def err_deg(a1,ref):
    ### Calculate the error ref-a1 in an efficient way in the circular space
    ### it uses complex numbers!
    ### Input in degrees (0-360)
    a1=np.radians(a1)
    ref=np.radians(ref)
    err = np.angle(np.exp(1j*ref)/np.exp(1j*(a1) ), deg=True) 
    err=round(err, 2)
    return err




def closest(lst, K):       
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

def model_I0E(center_angle, size_windows, N=512):
    inf, sup = np.radians(center_angle) - np.radians(size_windows/2), np.radians(center_angle) + np.radians(size_windows/2)
    new_I0E=[]
    theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)];
    for i in theta:
        if i < closest(theta, inf):
            new_I0E.append(0)
        elif i>= closest(theta, inf) and i <=closest(theta, sup):
            new_I0E.append(1)
        else:
            new_I0E.append(0)
    
    return np.reshape(np.array(new_I0E), (N,1))



def ornstein_uhlenbeck(t_final, delta_t , theta ):
    sigma = 1.
    # The time array of the trajectory
    time = np.arange(0, t_final, delta_t)
    # Initialise the array y
    y = np.zeros(time.size)
    # Generate a Wiener process
    dw = np.random.normal(loc = 0, scale = np.sqrt(delta_t), size = time.size)
    # Integrate the process
    for i in range(1,time.size):
        y[i] = y[i-1] - theta*y[i-1]*delta_t + sigma*dw[i]

    return(y) 






def model(totalTime, targ_onset_1, targ_onset_2, presentation_period, angle_target_i, angle_separation, tauE=9, 
          tauI=4,  n_stims=2, I0E=0.1, I0I=0.5, GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.5, sigI=1.6, k_noise=0.8,
          kappa_E=100, kappa_I=1.75, kappa_stim=100, N=512, plot_connectivity=False, plot_rate=False, 
          plot_hm=True , plot_fit=True, stim_strengthE=1., stim_strengthI=1., 
          phantom_st = 0.2, phantom_onset=500, phantom_on='off', phnatom_duration=200, just_final_re=False):
    #
    st_sim =time.time()
    dt=2
    nsteps=int(floor(totalTime/dt));
    origin = np.radians(angle_target_i)
    rE=zeros((N,1));
    rI=zeros((N,1));
    #Connectivities
    v_E=zeros((N));
    v_I=zeros((N));
    WE=zeros((N,N));
    WI=zeros((N,N));
    separation =  np.radians(angle_separation)
    angle_target=angle_target_i
    angle_distractor=angle_target_i-angle_separation
    if n_stims==1:
        separation=0


    theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 
    for i in range(0, N):
        v_E_new=[e**(kappa_E*cos(theta[f]))/(2*pi*scipy.special.i0(kappa_E)) for f in range(0, len(theta))]    
        v_I_new=[e**(kappa_I*cos(theta[f]))/(2*pi*scipy.special.i0(kappa_I)) for f in range(0, len(theta))]
        ###    
        vE_NEW=roll(v_E_new,i)
        vI_NEW=roll(v_I_new,i) #to roll
        ###    
        WE[:,i]=vE_NEW
        WI[:,i]=vI_NEW
    #
    # Plot of the connectivity profile
    if plot_connectivity ==True:
        plt.figure()
        plt.plot(WE[250, :], label='E')
        plt.plot(WI[250, :], label = 'I')
        plt.ylim(0,6)
        plt.gca().spines['right'].set_visible(False) #no right axis
        plt.gca().spines['top'].set_visible(False) #no  top axis
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()
        plt.show(block=False)
    ##
    # Stims
    if n_stims==2:
        stimulus1=zeros((N))
        stimulus2=zeros((N))
        for i in range(0, N):
            stimulus1[i]=e**(kappa_stim*cos(theta[i] + origin)) / (2*pi*scipy.special.i0(kappa_stim))
            stimulus2[i]=e**(kappa_stim*cos(theta[i] + origin + separation)) / (2*pi*scipy.special.i0(kappa_stim))
        #stimulus= (stimulus1 + stimulus2);
        #stimulus=reshape(stimulus, (N,1))
        stimulus_1=reshape(stimulus1, (N,1))
        stimulus_2=reshape(stimulus2, (N,1))
    elif n_stims==1:
        stimulus2=zeros((N));
        for i in range(0, N):
            stimulus2[i]=e**(kappa_stim*cos(theta[i] + origin)) / (2*pi*scipy.special.i0(kappa_stim))
        stimulus=stimulus2
        stimulus=reshape(stimulus, (N,1))
    ###
    ####
    stimon1 = floor(targ_onset_1/dt);
    stimoff1 = floor(targ_onset_1/dt) + floor(presentation_period/dt) ;
    stimon2 = floor(targ_onset_2/dt);
    stimoff2 = floor(targ_onset_2/dt) + floor(presentation_period/dt) ;
    #Simulation
    #generation of the noise and the connectivity between inhib and exitatory
    RE=zeros((N,nsteps));
    RI=zeros((N,nsteps));
    f = lambda x : x*x*(x>0)*(x<1) + reshape(array([cmath.sqrt(4*x[i]-3) for i in range(0, len(x))]).real, (N,1)) * (x>=1)

    background_s= I0E* ones((N,1))
    background_silent = -8. * ones((N,1))
    ##print(1)
    background_on = (I0E+phantom_st) * ones((N,1))
    background = background_s 
    ##noisee
    #numcores = multiprocessing.cpu_count()
    #noisepopE = np.array( Parallel(n_jobs = numcores)(delayed(ornstein_uhlenbeck)(t_final=totalTime, delta_t=dt, theta=k_noise)  for n in range(N)) )
    #noisepopI = np.array( Parallel(n_jobs = numcores)(delayed(ornstein_uhlenbeck)(t_final=totalTime, delta_t=dt, theta=k_noise)  for n in range(N)) )
    ## diferential equations
    for i in range(0, nsteps):
        #noiseE = np.reshape(sigE*noisepopE[:,i], (N,1))
        #noiseI = np.reshape(sigI*noisepopI[:,i], (N,1))
        #noiseE = sigE*random.randn(N,1);
        #noiseI = sigI*random.randn(N,1);
        noiseE = sigE*1.32*random.randn(N,1); #*1.44
        noiseI = sigI*1.32*random.randn(N,1);
        #differential equations for connectivity
        ## ## everything defining the excitatory current must end in the excitatory (GEE, GIE) and viceversa
        IE= GEE*dot(WE,rE) - GIE*dot(WI,rI) + background; 
        II= GEI*dot(WE,rE) +  (I0I-GII*mean(rI))*ones((N,1));
        
        # phantom condition
        if phantom_on == 'off':
            if i< stimon1:
                IE= GEE*dot(WE,rE) - GIE*dot(WI,rI) + background_silent; ##before 1st stim, inhibition of the network
        if phantom_on == 'on':
            if i< float(phantom_onset/dt):
                IE= GEE*dot(WE,rE) - GIE*dot(WI,rI) + background_silent;

        if i > float(phantom_onset/dt) and i < float(phantom_onset/dt) + float(phnatom_duration/dt) :
            background= background_on
        else:
            background= background_s
        ## stim condition
        if i>stimon1 and i<stimoff1:
            IE=IE+stim_strengthE*stimulus_1 + (background_on-background_s);
            II=II+stim_strengthI*stimulus_1;
        if i>stimon2 and i<stimoff2:
            IE=IE+stim_strengthE*stimulus_2 + (background_on-background_s);
            II=II+stim_strengthI*stimulus_2;
        #
        #rates of exit and inhib
        rE = rE + (f(IE) - rE + noiseE)*dt/tauE;
        rI = rI + (f(II) - rI + noiseI)*dt/tauI;
        rEr=reshape(rE, N)
        rIr=reshape(rI, N)
        #drawnow
        RE[:,i] = rEr;
        RI[:,i] = rIr;
    #
    ## metrics
    if n_stims==2:
        interference = Interference_effects( [decode_rE(stimulus1)], [decode_rE(rE)], [decode_rE(stimulus2)])[0]

    p_targ1 = int((N * np.degrees(origin))/360)
    p_targ2 = int((N * np.degrees(origin + separation))/360)
    #
    if plot_rate==True:
        #### plot dynamics
        fig = plt.figure()
        plt.title('Rate dynamics')
        plt.plot(RE[N-p_targ1, :], 'b', label='target1')
        plt.plot(RE[N-p_targ2, :], 'r', label='target2')
        plt.xlabel('time (ms)')
        plt.ylabel('rate (Hz)')
        plt.legend()
        plt.show(block=False)
    if plot_hm==True:
        #### plot heatmap
        RE_sorted=flipud(RE)
        sns.set_context("poster", font_scale=1.1)
        sns.set_style("ticks")
        plt.figure(figsize=(10,5))
        #sns.heatmap(RE_sorted, cmap='binary', vmax=8)
        sns.heatmap(RE_sorted, cmap='RdBu_r')
        #plt.title('BUMP activity') # remove title
        #plt.ylabel('Angle')
        plt.xlabel('time (s)')
        #plt.ylim(int(3*N/8), int(5*N/8))
        plt.plot([stimon1, nsteps], [p_targ1, p_targ1], '--k', linewidth=2) ## flipped, so it is p_target 
        plt.plot([stimon2, nsteps], [p_targ2, p_targ2], '--k', linewidth=2) ## flipped, so it is p_target 
        #plt.plot([stimon, nsteps], [p_targ1, p_targ1], '--r',) ## flipped, so it is p_target 
        plt.yticks([])
        plt.xticks([])
        plt.yticks([N/8, 3*N/8, 5*N/8, 7*N/8 ] ,['45','135','225', '315'])
        #plt.ylim( 3*N/8, 5*N/8)
        #plt.plot([stimon, stimon,], [0+20, N-20], 'k-', label='onset')
        #plt.plot([stimoff, stimoff,], [0+20, N-20], 'k--', label='offset')
        #plt.plot([stimon, stimon,], [0+20, N-20], 'k-', linewidth=0.5)
        #plt.plot([stimoff, stimoff,], [0+20, N-20], 'k-', linewidth=0.5)
        #plt.legend()
        plt.show(block=False)
    
    
    ## print time consumed
    end_sim =time.time()
    total_time= end_sim - st_sim 
    total_time = round(total_time, 1)
    #print('Simulation time: ' + str(total_time) + 's')
    ####
    decode = decode_rE(flipud(rE))
    err=err_deg(decode, angle_target_i)
    err_2 = err_deg(decode, (angle_target_i+angle_separation) )

    if just_final_re==True:
        return rE
    else:
        return(decode, err, err_2, rE, RE, total_time) #bias_b1, bias_b2)


###