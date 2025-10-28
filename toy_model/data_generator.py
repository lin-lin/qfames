import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt

def generate_Hadamard_test_data(spectrum,coefficient,t_list,N_list):
    """ -Input:

    spectrum: np.array of eigenvalues
    coefficient: np.array of coefficients
    t_list: np.array of time points
    N_list: np.array of numbers of samples

    -Ouput:

    Z_Had: np.array of the output of Hadamard test (row)
    T_max: maximal Hamiltonian simulation time
    T_total: total Hamiltonian simulation time

    """
    if len(t_list)!=len(N_list):
       print('list error')
    t_list=np.array(t_list)
    N_list=np.array(N_list)
    N_list=N_list.flatten()
    N=len(t_list)
    Nsample=int(max(N_list))
    #generate true expectation
    z=coefficient.dot(np.exp(-1j*np.outer(spectrum,t_list)))
    Re_true=(1+np.real(z))/2
    Im_true=(1+np.imag(z))/2
    #construct check matrix for different Nsample
    N_check=np.arange(Nsample).reshape([Nsample, 1])
    N_check=N_check*np.ones((1,N))
    Sign_check=np.ones((Nsample, 1))*(N_list-0.5)
    Re_check=(np.sign(N_check-Sign_check)-1)/(-2)
    Im_check=(np.sign(N_check-Sign_check)-1)/(-2)
    Re_true=np.multiply(Re_check,np.ones((Nsample, 1)) * Re_true)
    Im_true=np.multiply(Im_check,np.ones((Nsample, 1)) * Im_true)
    #simulate Hadamard test
    Re_random=np.random.uniform(0,1,(Nsample,N))
    Im_random=np.random.uniform(0,1,(Nsample,N))
    Re=np.sum(Re_random<Re_true,axis=0)/N_list
    Im=np.sum(Im_random<Im_true,axis=0)/N_list
    Z_Had = (2*Re-1)+1j*(2*Im-1)
    T_max = max(np.abs(t_list))
    T_total = sum(np.multiply(np.abs(t_list),N_list))
    return Z_Had, T_max, T_total
