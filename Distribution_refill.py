import numpy as np
#import pandas as pd
import numpy.matlib
import random
import math
from scipy import special


def search_neighbors(x,y,s):
    
    """
    :param y: matrix of sorted labels
    :param k: number of nearest neighbours
    """
    
    N1, N2 = y.shape[0], x.shape[0]
    closest = []
    for i in range(N1):
        dist = np.zeros(N2)
        for j in range(N2):
            for k in range(len(y[i])):
                dist[j] += (x[j,k]-y[i,k])**2
            dist[j] = np.sqrt(dist[j])
    
        #Now find the index of the 'k' elements with smallest distance for each y element
        closest.append(np.argsort(dist)[:s])
    return closest


def multiprod(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # First check if we have been given just one matrix
    if len(np.shape(A)) == 2:
        return np.dot(A, B)

    # Old (slower) implementation:
    # a = A.reshape(np.hstack([np.shape(A), [1]]))
    # b = B.reshape(np.hstack([[np.shape(B)[0]], [1], np.shape(B)[1:]]))
    # return np.sum(a * b, axis=2)

    # Approx 5x faster, only supported by numpy version >= 1.6:
    return np.einsum('ijk,ikl->ijl', A, B)


def refill_distribution(dist_filename,R_aperture,Num):
    
    #Dist_dataframe = pd.read_csv(dist_filename,header=None, delim_whitespace = True)
    #Dist_dataframe.columns=['x','y','z','px','py','pz','clock','macro_charge','particle_index','status']
    #delta_z = np.array(bunch_dataframe['z'].astype(float).tolist())
    Dist_array = np.loadtxt(dist_filename)

    #We filter the dataframe depending on particle status
    #a = Dist_dataframe[Dist_dataframe['status']>0]
    a = Dist_array[Dist_array[:,9]>0]

    #We transfer the filtered information to a matrix
    Dist_matrix  = a
   
    #We save the coordenates x,x',y,y',z and z' of the filtered particles in P and the amount of active    particles in n
    P = Dist_matrix[1:,0:7]
    n = P.shape[0]

    #Get the max value and min value of each column from P: the limits of the phase space, then we save the averages in MAX and the standard deviation of each column in RMS
    MAX = (P.max(0) - P.min(0))/2.0
    RMS = P.std(0)[0:2]     #Only save rms x and y, not sure about this

    #Build a matrix with the same number of elements as P but with the average values of each column repeated in every row
    averages = numpy.matlib.repmat(MAX,n,1)

    #Divide P element by element with this new matrix, all elements in P0 are going to be between -1 and 1
    P0 = np.divide(P,averages)

    #Now filter the elements in P depending on the radius, only take particles with x**2+y**2 smaller than R
    Pa = np.array([np.array(x) for x in P if np.sqrt(x[0]**2+x[1]**2) <= R_aperture])
    Pb = np.array([np.array(x) for x in P if np.sqrt(x[0]**2+x[1]**2) <= np.min([10.0*R_aperture,np.mean(RMS)])])   #Not completely sure about this
    

    nb = Pb.shape[0]
    averages_b = numpy.matlib.repmat(MAX,nb,1)
    Pb_0 = np.divide(Pb,averages_b)

    
    NDist = Pa.shape[0]

    #Initialize the new array we will use for the output with the reference particle
    Dist1 = Dist_matrix[0,0:7]

    #Attach all the particles that have x**2+y**2 smaller than R to the output distribution
    Dist1 = np.vstack([Dist1,Pa])

    Np = 100
   
    #try:
   
    #---------------------NEW FROM HERE-------------------
    # Find the 7 closest neighbors to every particle in Pb_0 (we find 8, but the closest one is always the particle itself). Save restults in closest_neighbor [pb_0.shape[0],7] dimension
    # import time
    # start = time.process_time()
    Dist = np.asarray(search_neighbors(Pb_0,Pb_0,8))
    closest_neighbors = Dist[:,1:8]

    #We calculate the coordinates of those 7 neighbors respect to the particle itself (thats why we subtract the coordinates of the particle itself to the koordinates of every neighbor). 
    #We save the results in koord, which will be a [Pb_0.shape[0],7,6] dimension matrix
    koord = []
    for particles in range(Pb_0.shape[0]):
        koord.append([])
        for i in range(7):
            koord[particles].append(np.subtract(Pb_0[closest_neighbors[particles,i]],Pb_0[particles]))
    koord = np.asarray(koord)
  
    #--------------------------------
    k=0
    while NDist < Num :
        Faktor = np.random.rand(nb*7*Np,1)*2-1
        FaktorERF = np.random.rand(nb*7*Np,1)
        FaktorInd = np.where(FaktorERF <= (special.erf(Faktor)+1)/2.0)
        Faktor = Faktor[FaktorInd]
        #print(Pb_0.shape)
        Faktor_reshape = Faktor[:nb*7*20]
        Faktor = Faktor_reshape.reshape(nb,7,20)
   
        #print(koord.shape)
        #print(Faktor.shape)

        PS = multiprod(koord,Faktor) 



        #----------PROBLEM IS IN NEXT BLOCK!!!!!!!!!
        carrier = []
        for j in range(Pb_0.shape[0]):
            carrier.append([])
            for l in range(Pb_0.shape[1]):
                carrier[j].append([])
                for i in range(20):
                    carrier[j][l].append(Pb_0[j,l])
        #----------PROBLEM IS IN PREVIOUS BLOCK!!!!!!!!!

        PS = PS + np.asarray(carrier) #np.tile(Pb_0,[1,1,20]) #np.tile is the equivalent of repmat in python, multiprod is not exactly as the one in matlab (maybe some changes needed here)
        #print(PS.shape)
        PS = PS.transpose(0, 2, 1).reshape(nb*20,7)

        #Works until here
        for_multi = []
        for i in range(PS.shape[0]):
            for_multi.append(MAX)
        for_multi = np.asarray(for_multi)
        Pa = np.multiply(PS,for_multi)
        #print(Pa)

        Pa_radius = np.asarray([np.sqrt(i[0]**2+i[1]**2) for i in Pa])
        Pa_Ind = np.where(Pa_radius <= R_aperture)
        #print(Pa_Ind)
        #Pa = Pa[Pa_radius <= R_aperture]
        Pa = Pa[Pa_Ind]
        #print(Pa.shape)

        Dist1 = np.vstack([Dist1,Pa])
        NDist = Dist1.shape[0]

    #We calculate how much charge has gone through the aperture
    Qpass = np.sum(np.array([x[7] for x in Dist_matrix if np.sqrt(x[0]**2+x[1]**2) <= R_aperture]))
    #print(Qpass)

    #Now, as Dist1 contains more particles than the desired output number Num we will take Num random integers from 0 to NDist-1 and take those particles for the output distribution
    
    x = np.random.randint(0,NDist-1,size=(Num-1))
    x = np.asarray(x)
    Dist_return = Dist1[0]
    Dist_return = np.vstack([Dist_return,Dist1[x+1]])
    #print(Dist_return)

    #Finally, we add the columns corresponding to charge, index and status
    New_macroCharge = np.ones(Num)*(Qpass/Num)
    #print(New_macroCharge)
    Dist_return = np.c_[Dist_return,New_macroCharge]
    Dist_return = np.c_[Dist_return,np.ones(Num)*a[1,8]]
    Dist_return = np.c_[Dist_return,np.ones(Num)*a[1,9]]
    #print(Dist_return[0])
    return Dist_return


#import time
import timeit
start = timeit.default_timer()

return_distribution = refill_distribution('Gun.0174.001',0.00025,20000)
with open('Gun_refill.0174.001','w') as final_dist:
    np.savetxt(final_dist,return_distribution)
    final_dist.close()


stop = timeit.default_timer()

print('Time: ', stop - start)  

