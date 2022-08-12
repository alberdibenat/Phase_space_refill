import numpy as np
#import pandas as pd
import numpy.matlib
import random
import math
from scipy import special
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.kde import gaussian_kde
#import time
import timeit

def gaussian(x,sigma,mu):
    return np.exp(-(x-mu)**2/2/(sigma**2))#/(np.sqrt(2*np.pi)*sigma)

def plot_distribution(dist_filename):
    new_Dist_array = np.loadtxt(dist_filename)

    #Filter particles depending on their status
    new_a = new_Dist_array[new_Dist_array[:,9]>0]
    #Get the positions (x,y,z) and momentums (px,py,pz) as numpy arrays from the distributiuon 
    x = np.copy(new_a[:,0])
    y = np.copy(new_a[:,1])
    z = np.copy(new_a[:,2])
    px = np.copy(new_a[:,3])
    py = np.copy(new_a[:,4])
    pz = np.copy(new_a[:,5])
    
    me = 511e3 #eV
    c = 3e8 #m/s
    momentum_ref = np.sqrt(px[0]**2+py[0]**2+pz[0]**2) #eV/c
    energy_ref = np.sqrt((momentum_ref)**2 + (me)**2)
    gamma_ref = energy_ref/me
    beta_ref = np.sqrt(gamma_ref**2-1)/gamma_ref
    
    z_ref = z[0]
    pz_ref = pz[0]
    z[0] = z[0]-z_ref
    pz[0] = pz[0] - pz_ref
    
    delta_pz = pz
    delta_z = z
    z = z + z_ref
    pz = pz + pz_ref
    
    momentum = np.sqrt(px**2+py**2+pz**2) #eV/c
    energy = np.sqrt(momentum**2 + me**2)
    delta_e = energy - energy_ref*np.ones(len(energy))


    #Geometrically it would make more sense to use the angle coming from acos(px/pxz) for example, but astra uses px/pz instead 
    #x_prime = np.arccos(pz/np.sqrt(pz**2+px**2))
    #y_prime = np.arccos(pz/np.sqrt(pz**2+py**2))
    x_prime = px/pz
    y_prime = py/pz

    #We calculate kernel densities for the density-contour plots 
    k_x_px = gaussian_kde(np.vstack([x, px]))
    a_x_px, b_x_px = np.mgrid[x.min():x.max():x.size**0.5*1j,px.min():px.max():px.size**0.5*1j]
    z_x_px = k_x_px(np.vstack([a_x_px.flatten(), b_x_px.flatten()]))

    k_y_py = gaussian_kde(np.vstack([y, py]))
    a_y_py, b_y_py = np.mgrid[y.min():y.max():y.size**0.5*1j,py.min():py.max():py.size**0.5*1j]
    z_y_py = k_y_py(np.vstack([a_y_py.flatten(), b_y_py.flatten()]))

    k_z_pz = gaussian_kde(np.vstack([delta_z, delta_pz]))
    a_z_pz, b_z_pz = np.mgrid[delta_z.min():delta_z.max():delta_z.size**0.5*1j,delta_pz.min():delta_pz.max():delta_pz.size**0.5*1j]
    z_z_pz = k_z_pz(np.vstack([a_z_pz.flatten(), b_z_pz.flatten()]))
    
    k_x_prime = gaussian_kde(np.vstack([x, x_prime]))
    a_x_prime, b_x_prime = np.mgrid[x.min():x.max():x.size**0.5*1j,x_prime.min():x_prime.max():x_prime.size**0.5*1j]
    z_x_prime = k_x_prime(np.vstack([a_x_prime.flatten(), b_x_prime.flatten()]))

    k_y_prime = gaussian_kde(np.vstack([y, y_prime]))
    a_y_prime, b_y_prime = np.mgrid[y.min():y.max():y.size**0.5*1j,y_prime.min():y_prime.max():y_prime.size**0.5*1j]
    z_y_prime = k_y_prime(np.vstack([a_y_py.flatten(), b_y_prime.flatten()]))

    k_z_delta = gaussian_kde(np.vstack([delta_z,delta_e ]))
    a_z_delta, b_z_delta = np.mgrid[delta_z.min():delta_z.max():delta_z.size**0.5*1j,delta_e.min():delta_e.max():delta_e.size**0.5*1j]
    z_z_delta = k_z_delta(np.vstack([a_z_delta.flatten(), b_z_delta.flatten()]))
    

    colors = ['tab:blue','tab:red','tab:green']
    fig, ax = plt.subplots(2,3,figsize = (18,10))
    ax[0,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[0,0].set_title('x-px phase space')
    ax[0,0].set_xlabel(r'$\Delta$x [m]')
    ax[0,0].set_ylabel(r'$\Delta$px [eV/c]')
    ax[0,0].scatter(x,px,s = 1,color=colors[0])
    ax[0,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[0,1].set_title('y-py phase space')
    ax[0,1].set_xlabel(r'$\Delta$y [m]')
    ax[0,1].set_ylabel(r'$\Delta$py [eV/c]')
    ax[0,1].scatter(y,py,s = 1,color=colors[1])
    ax[0,2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0,2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[0,2].set_title('z-pz phase space')
    ax[0,2].set_xlabel(r'$\Delta$z [m]')
    ax[0,2].set_ylabel(r'$\Delta$pz [eV/c]')
    ax[0,2].scatter(delta_z,delta_pz,s = 1,color=colors[2])
    ax[1,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[1,0].set_title('x-x\' phase space')
    ax[1,0].set_xlabel(r'$\Delta$x [m]')
    ax[1,0].set_ylabel('x\' [rad]')
    ax[1,0].scatter(x,x_prime,s = 1,color=colors[0])
    ax[1,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[1,1].set_title('y-y\' phase space')
    ax[1,1].set_xlabel(r'$\Delta$y [m]')
    ax[1,1].set_ylabel('y\' [rad]')
    ax[1,1].scatter(y,y_prime,s = 1,color=colors[1])
    ax[1,2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[1,2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[1,2].set_title(r'z-$\delta_e$ phase space')
    ax[1,2].set_xlabel(r'$\Delta$z [m]')
    ax[1,2].set_ylabel(r'$\delta_e$ [eV]')
    ax[1,2].scatter(delta_z,delta_e,s = 1,color=colors[2])
    plt.tight_layout()
    fig.savefig(dist_filename + '.png')
#    plt.close()
    

    fig1, ax1 = plt.subplots(2,3,figsize = (17,10))
    fig1.subplots_adjust(left=0.05)
    fig1.subplots_adjust(right=0.93)
    fig1.subplots_adjust(top=0.9)
    fig1.subplots_adjust(bottom=0.1)
    ax1[0,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1[0,0].set_title('x-px phase space')
    ax1[0,0].set_xlabel(r'$\Delta$x [m]')
    ax1[0,0].set_ylabel(r'$\Delta$px [eV/c]')
    cmap = ax1[0,0].contourf(a_x_px,b_x_px,z_x_px.reshape(a_x_px.shape)/np.amax(z_x_px.reshape(a_x_px.shape)),levels = 10)
    ax1[0,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1[0,1].set_title('y-py phase space')
    ax1[0,1].set_xlabel(r'$\Delta$y [m]')
    ax1[0,1].set_ylabel(r'$\Delta$py [eV/c]')
    ax1[0,1].contourf(a_y_py,b_y_py,z_y_py.reshape(a_y_py.shape)/np.amax(z_y_py.reshape(a_x_px.shape)),levels = 10)
    ax1[0,2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1[0,2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1[0,2].set_title('z-pz phase space')
    ax1[0,2].set_xlabel(r'$\Delta$z [m]')
    ax1[0,2].set_ylabel(r'$\Delta$pz [eV/c]')
    ax1[0,2].contourf(a_z_pz,b_z_pz,z_z_pz.reshape(a_z_pz.shape)/np.amax(z_z_pz.reshape(a_x_px.shape)), levels = 10)
    ax1[1,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1[1,0].set_title('x-x\' phase space')
    ax1[1,0].set_xlabel(r'$\Delta$x [m]')
    ax1[1,0].set_ylabel('x\' [rad]')
    ax1[1,0].contourf(a_x_prime,b_x_prime,z_x_prime.reshape(a_x_prime.shape)/np.amax(z_x_prime.reshape(a_x_px.shape)), levels = 10)
    ax1[1,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1[1,1].set_title('y-y\' phase space')
    ax1[1,1].set_xlabel(r'$\Delta$y [m]')
    ax1[1,1].set_ylabel('y\' [rad]')
    ax1[1,1].contourf(a_y_prime,b_y_prime,z_y_prime.reshape(a_y_prime.shape)/np.amax(z_x_prime.reshape(a_x_px.shape)), levels = 10)
    ax1[1,2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1[1,2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1[1,2].set_title(r'z-$\delta_e$ phase space')
    ax1[1,2].set_xlabel(r'$\Delta$z [m]')
    ax1[1,2].set_ylabel(r'$\delta_e$ [eV]')
    ax1[1,2].contourf(a_z_delta,b_z_delta,z_z_delta.reshape(a_z_delta.shape)/np.amax(z_z_delta.reshape(a_x_px.shape)), levels = 10)
    cbar_ax = fig1.add_axes([0.945, 0.15, 0.013, 0.7])
    cbar = fig1.colorbar(cmap, cax=cbar_ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Particle density (arb. unit)', rotation=270)
    #plt.tight_layout()
    fig1.savefig(dist_filename + '_density.png')
    
    fig2, ax2 = plt.subplots(1,1,figsize = (5,5))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.set_title('x-y space')
    ax2.set_xlabel(r'$\Delta$x [m]')
    ax2.set_ylabel(r'$\Delta$y [m]')
    ax2.scatter(x,y,s = 1,color='darkorange')
    plt.tight_layout()
    fig2.savefig(dist_filename + '_XY.png')

    plt.show()

    return 0 
    


def refill_distribution(dist_filename,R_aperture,limit_number_particles):
    #### INPUTS: 
    ####        ->dist_filename: filename of distribution
    ####        ->R_aperture: Radius of aperture [m]
    ####        ->limit_number_particles: Number of core-particles that one wants to include for the twiss parameter calculations. 1.0 for all of them, 0.1 for the particles located in the region with normalized density between 0.9 and 1.0.
    #### OUTPUTS: 
    ####        ->collimated_dist: Collimated and refilled distribution, same length as input distribution






    #### Plot phase spaces first
    status_plot = plot_distribution(dist_filename)

    #### EMITTANCE AND OPTICAL FUNCTION CALCULATIONS ARE BASED ON THE ARTICLE:
    #### 'Some basic features of beam emittance' by Klaus Floettmann, Physical Review Accelerators and Beams, Volume 6, 034202 (2003) 
    start = timeit.default_timer()

    Dist_array = np.loadtxt(dist_filename)

    #Filter particles depending on their status
    a = Dist_array[Dist_array[:,9]>0]
     
    #Get the positions (x,y,z) and momentums (px,py,pz) as numpy arrays from the distributiuon 
    x = np.copy(a[:,0])
    y = np.copy(a[:,1])
    z = np.copy(a[:,2])
    px = np.copy(a[:,3])
    py = np.copy(a[:,4])
    pz = np.copy(a[:,5])
    

    me = 511e3 #eV
    c = 3e8 #m/s
    momentum_ref = np.sqrt(px[0]**2+py[0]**2+pz[0]**2) #eV/c
    energy_ref = np.sqrt((momentum_ref)**2 + (me)**2)
    gamma_ref = energy_ref/me
    beta_ref = np.sqrt(gamma_ref**2-1)/gamma_ref

    z_ref = z[0]
    pz_ref = pz[0]
    z[0] = z[0]-z_ref
    pz[0] = pz[0] - pz_ref
    
    delta_z = z
    delta_pz = pz
    z = z + z_ref
    pz = pz + pz_ref
    
    momentum = np.sqrt(px**2+py**2+pz**2) #eV/c
    energy = np.sqrt(momentum**2 + me**2)
    delta_e = energy - energy_ref*np.ones(len(energy))

    #Geometrically it would make more sense to use the angle coming from acos(px/pxz) for example, but astra uses px/pz instead 
    #x_prime = np.arccos(pz/np.sqrt(pz**2+px**2))
    #y_prime = np.arccos(pz/np.sqrt(pz**2+py**2))
    x_prime = px/pz
    y_prime = py/pz


    #Now we choose the density of particles we want to use for the calculation of twiss parameters
    limit = limit_number_particles #1.0 would be to include all particles, 0.1 would mean we are taking the ones located in the region with normalized density between 0.9 and 1.0
    density = 1.0 - limit
    print('\n')
    print('Including particles located in the region with normalized density between ' + str(density) + ' and 1.0 in the emittance and twiss parameter calculations:')
    print('\n')

    #We calculate kernel densities for the different phase-spaces and the density value for the location of each particle 
    k_x_px = gaussian_kde(np.vstack([x, px]))
    a_x_px, b_x_px = np.mgrid[x.min():x.max():x.size**0.5*1j,px.min():px.max():px.size**0.5*1j]
    z_x_px = k_x_px(np.vstack([a_x_px.flatten(), b_x_px.flatten()]))
    #Here is where we calculate the density value and we normalize, it goes from 0.0 to 1.0
    values_x_px = k_x_px.evaluate(np.array([x,px]))/np.amax(z_x_px.reshape(a_x_px.shape))

    k_y_py = gaussian_kde(np.vstack([y, py]))
    a_y_py, b_y_py = np.mgrid[y.min():y.max():y.size**0.5*1j,py.min():py.max():py.size**0.5*1j]
    z_y_py = k_y_py(np.vstack([a_y_py.flatten(), b_y_py.flatten()]))
    density_y_py = z_y_py.reshape(a_y_py.shape)/np.amax(z_y_py.reshape(a_y_py.shape))
    values_y_py = k_y_py.evaluate(np.array([y,py]))/np.amax(z_y_py.reshape(a_y_py.shape))

    k_z_pz = gaussian_kde(np.vstack([delta_z, delta_pz]))
    a_z_pz, b_z_pz = np.mgrid[delta_z.min():delta_z.max():delta_z.size**0.5*1j,delta_pz.min():delta_pz.max():delta_pz.size**0.5*1j]
    z_z_pz = k_z_pz(np.vstack([a_z_pz.flatten(), b_z_pz.flatten()]))
    density_z_pz = z_z_pz.reshape(a_z_pz.shape)/np.amax(z_z_pz.reshape(a_z_pz.shape))
    values_z_pz = k_z_pz.evaluate(np.array([delta_z,delta_pz]))/np.amax(z_z_pz.reshape(a_z_pz.shape))
    
    ### Now finally we filter our initial numpy arrays depending on the density we want to use for our calculations
    ### ACHTUNG!! We will only filter by one of the two spaces, in our case we choose the x-px, y-py, z-pz as the real emittances (instead of x-x_prime, y-y_prime, z-delta_e)
    x = x[values_x_px>=density]
    px = px[values_x_px>=density]
    x_prime = x_prime[values_x_px>=density] #this is where we also filter x_prime depending on x-px space. Otherwise we would have to define two different x variables
    y = y[values_y_py>=density]
    py = py[values_y_py>=density]
    y_prime = y_prime[values_y_py>=density] #Same here
    z = z[values_z_pz>=density]
    pz = pz[values_z_pz>=density]
    delta_e = delta_e[values_z_pz>=density] #Same here

    #We can now continue with the emittance calculations
    x_ave = np.mean(x)
    y_ave = np.mean(y)
    x_prime_ave = np.mean(x_prime)
    y_prime_ave = np.mean(y_prime)

    x_squared_var = np.sum([(i-x_ave)**2 for i in x])/len(x)  ### = x_rms
    y_squared_var = np.sum([(i-y_ave)**2 for i in y])/len(y)  ### = y_rms
    x_squared_var_2 = np.sum([i**2 for i in x])/len(x) - (np.sum(x))**2/(len(x)**2) 
    y_squared_var_2 = np.sum([i**2 for i in y])/len(y) - (np.sum(y))**2/(len(y)**2) 
    #The two ways to calculate x/y_squared_var are equivalent, they agree until 15th decimal. The only difference is that one uses numpy predefined functions and the other does not
    px_squared_var = np.sum([i**2 for i in px])/len(px) - (np.sum(px))**2/(len(px)**2) 
    py_squared_var = np.sum([i**2 for i in py])/len(py) - (np.sum(py))**2/(len(py)**2) 
    x_prime_squared_var = np.sum([(i-x_prime_ave)**2 for i in x_prime])/len(x_prime)
    y_prime_squared_var = np.sum([(i-y_prime_ave)**2 for i in y_prime])/len(y_prime)
    x_px_var = np.sum([x[i]*px[i] for i in range(len(x))])/len(x) - np.sum(x)*np.sum(px)/(len(x))**2
    y_py_var = np.sum([y[i]*py[i] for i in range(len(y))])/len(y) - np.sum(y)*np.sum(py)/(len(y))**2
    x_x_prime_var = np.sum([(x[i]-x_ave)*(x_prime[i]-x_prime_ave) for i in range(len(x))])/len(x)
    y_y_prime_var = np.sum([(y[i]-y_ave)*(y_prime[i]-y_prime_ave) for i in range(len(y))])/len(y)

    #Calculate the normalized transverse emittances from the distribution
    norm_emittance_x = np.sqrt(x_squared_var_2*px_squared_var - x_px_var**2)/(me)
    norm_emittance_y = np.sqrt(y_squared_var_2*py_squared_var - y_py_var**2)/(me)
    print('Norm. beam emittance X = ' + str(norm_emittance_x) + ' pi m rad')
    print('Norm. beam emittance Y = ' + str(norm_emittance_y) + ' pi m rad') 
    
    #Calculate the transverse emittances from the distribution
    emittance_x = norm_emittance_x/(gamma_ref*beta_ref)
    emittance_y = norm_emittance_y/(gamma_ref*beta_ref)
    #print('Beam emittance X = ' + str(emittance_x) + ' pi m rad')
    #print('Beam emittance Y = ' + str(emittance_y) + ' pi m rad') 
    
    #Calculate the transverse trace-emittances from the distribution
    trace_emittance_x = np.sqrt(x_squared_var*x_prime_squared_var - x_x_prime_var**2)
    trace_emittance_y = np.sqrt(y_squared_var*y_prime_squared_var - y_y_prime_var**2)
    #print('Beam trace-emittance X = ' + str(trace_emittance_x) + ' pi m rad')
    #print('Beam trace-emittance Y = ' + str(trace_emittance_y) + ' pi m rad') 

    #Calculate the normalized trace-space transverse emittances
    norm_trace_emittance_x = beta_ref*gamma_ref*trace_emittance_x
    norm_trace_emittance_y = beta_ref*gamma_ref*trace_emittance_y
    #print('Norm. beam trace-emittance X = ' + str(norm_trace_emittance_x) + ' pi m rad')
    #print('Norm. beam trace-emittance Y = ' + str(norm_trace_emittance_y) + ' pi m rad') 

    #Correlated divergence (=-alpha/beta)
    cor_div_x = x_x_prime_var/x_squared_var_2 #x_px_var/x_squared_var_2
    cor_div_y = y_y_prime_var/y_squared_var_2 #y_py_var/y_squared_var_2
    #print(cor_div_x)
    #print(cor_div_y)

    beta_x = x_squared_var/emittance_x
    alpha_x = -x_x_prime_var/emittance_x
    gamma_x = x_prime_squared_var/emittance_x
    print('Beta_X before collimation = '+str(beta_x) + ' m')
    print('Alpha_X before collimation = '+str(alpha_x) )
    print('Gamma_X before collimation = '+str(gamma_x) + ' m^(-1)' )
    
    beta_y = y_squared_var/emittance_y
    alpha_y = -y_y_prime_var/emittance_y
    gamma_y = y_prime_squared_var/emittance_y
    print('Beta_Y before collimation = '+str(beta_y) + ' m')
    print('Alpha_Y before collimation = '+str(alpha_y) )
    print('Gamma_Y before collimation = '+str(gamma_y) + ' m^(-1)' )

    print('\n')
    print('Refilling phase space for an aperture of R = '+str(R_aperture)+' m...')
    print('\n')
    #Now we want to keep the shape of the distribution in x and y phase spaces after collimation, but only with particles going through the collimator
    b = a[np.sqrt(a[:,0]**2+a[:,1]**2)<R_aperture]   #Surviving particles after collimation
    c = a[np.sqrt(a[:,0]**2+a[:,1]**2)>=R_aperture]  #Dead particles after collimation

    #We will replenish the x and y phase spaces using the longitudinal coordinates of the dead particles (this code assumes no correlation between transverse and long. phase-spaces) 
    z_dead = c[:,2] 
    pz_dead = c[:,5] 
    rest = c[:,6:]
    n_dead_part = len(z_dead)

    sig_x = np.sqrt(beta_x*emittance_x)
    #print('sig_x = '+str(sig_x))
    limits_x = [-10.0*sig_x,10.0*sig_x]
    norm_x = []
    norm_x_prime = []
    collimated_x = []
    collimated_x_prime = [] 
    
    sig_y = np.sqrt(beta_y*emittance_y)
    #print('sig_y = '+str(sig_y))
    limits_y = [-10.0*sig_y,10.0*sig_y]
    norm_y = []
    norm_y_prime = []
    collimated_y = []
    collimated_y_prime = [] 

    ### Now we need to fill the new distributions with the same twiss parameters, but making sure that x**2+y**2<=Radius_collimator
    ### A random walker can easily fill a 2d distribution with certain rms values in both axes, but the problem resides in the correlation between these two axes.
    ### To fix this, we will make use of the fact that the trajectories in the phase space x - (beta*x'+alpha*x) are circular, and we will assume gaussian distributions in both axes. 
    ### This solution was found in: http://nicadd.niu.edu/~syphers/tutorials/analyzeTrack.html , and can be used for any kind of distribution in phase_space
    #Later we could instead change this to any random distribution function by interpolating the histogram of the initial distriubtion in both axes and using that as density function.
    count=0

    sig_x_norm = sig_x#/np.sqrt(beta_x)
    limits_x_norm = limits_x#/np.sqrt(beta_x)
    sig_y_norm = sig_y#/np.sqrt(beta_y)
    limits_y_norm = limits_y#/np.sqrt(beta_y)
    

    while count < n_dead_part:
        for i in range(10000):
            x_candidate = np.random.random()*(limits_x_norm[1]-limits_x_norm[0])+limits_x_norm[0]
            x_value  = gaussian(x_candidate,sig_x_norm,0.0)
            prob_x = np.random.random()
            if prob_x <= x_value:
                x_norm = x_candidate
                break
        for i in range(10000):
            px_candidate = np.random.random()*(limits_x_norm[1]-limits_x_norm[0])+limits_x_norm[0]
            px_value  = gaussian(px_candidate,sig_x_norm,0.0)
            prob_px = np.random.random()
            if prob_px <= px_value:
                x_prime_norm = px_candidate
                break
        for i in range(10000):
            y_candidate = np.random.random()*(limits_y_norm[1]-limits_y_norm[0])+limits_y_norm[0]
            y_value  = gaussian(y_candidate,sig_y_norm,0.0)
            prob_y = np.random.random()
            if prob_y <= y_value:
                y_norm = y_candidate
                break
        for i in range(10000):
            py_candidate = np.random.random()*(limits_y_norm[1]-limits_y_norm[0])+limits_y_norm[0]
            py_value  = gaussian(py_candidate,sig_y_norm,0.0)
            prob_py = np.random.random()
            if prob_py <= py_value:
                y_prime_norm = py_candidate
                break

        #Keep track of the distributions be create before applying the collimator
        norm_x.append(x_norm)
        norm_y.append(y_norm)
        norm_x_prime.append(x_prime_norm)
        norm_y_prime.append(y_prime_norm)

    
        #Apply the collimator
        x0 = x_norm#*np.sqrt(beta_x)
        y0 = y_norm#*np.sqrt(beta_y)
        if np.sqrt(x0**2+y0**2) < R_aperture:
            x_prime0 = (x_prime_norm-alpha_x*x0)/beta_x
            y_prime0 = (y_prime_norm-alpha_y*y0)/beta_y
            collimated_x.append(x0)
            collimated_y.append(y0)
            collimated_x_prime.append(x_prime0)
            collimated_y_prime.append(y_prime0)
            count += 1
            #print(count)


    #The following 4 arrays are just to check the distributions we get in the previous section
    norm_x = np.asarray(norm_x)
    norm_x_prime = np.asarray(norm_x_prime)
    norm_y = np.asarray(norm_y)
    norm_y_prime = np.asarray(norm_y_prime)

    
    new_x_prime = (norm_x_prime-alpha_x*norm_x)/beta_x
    new_y_prime = (norm_y_prime-alpha_y*norm_y)/beta_y
    #print(np.sqrt(np.mean(np.square(norm_x))))
    #print(np.sqrt(np.mean(np.square(x_prime))))
    #print(np.sqrt(np.mean(np.square(norm_y))))
    #print(np.sqrt(np.mean(np.square(y_prime))))
    collimated_x = np.asarray(collimated_x)
    collimated_y = np.asarray(collimated_y)
    collimated_x_prime = np.asarray(collimated_x_prime)
    collimated_y_prime = np.asarray(collimated_y_prime)

    #Just to plot all the new distributions (without collimation)
    #sns.distplot(norm_x)
    #sns.distplot(new_x_prime)
    #sns.distplot(norm_y)
    #sns.distplot(new_y_prime)

    collimated_dist = np.stack((collimated_x,collimated_y,z_dead,collimated_x_prime*(pz_dead+pz_ref),collimated_y_prime*(pz_dead+pz_ref),pz_dead),axis=1)
    collimated_dist = np.concatenate((collimated_dist,rest),axis=1)
    collimated_dist = np.concatenate((b,collimated_dist),axis=0)

    stop = timeit.default_timer()
    needed_time = stop - start
    #print(np.amax(np.sqrt(new_dist[:,0]**2+new_dist[:,1]**2)))
    return collimated_dist, needed_time





#distname = 'Initial_Dist.ini'
distname = 'Gun.0174.001'
return_distribution, timing = refill_distribution(distname,0.0001,0.25) #This will return a distribution with the same number of particles as the original one, but collimated
    #### INPUTS: 
    ####        ->dist_filename: filename of distribution
    ####        ->R_aperture: Radius of aperture [m]
    ####        ->limit_number_particles: Number of core-particles that one wants to include for the twiss parameter calculations. 1.0 for all of them, 0.1 for the particles located in the region with normalized density between 0.9 and 1.0.
    #### OUTPUTS: 
    ####        ->collimated_Dist: Collimated and refilled distribution, same length as input distribution
with open('Collimated_Dist.ini','w') as final_dist:
    np.savetxt(final_dist,return_distribution)
    final_dist.close()


status_plot = plot_distribution('Collimated_Dist.ini')


print('\n')
print('Needed time for refilling: ' + str(timing) + ' s' )  
print('\n')

