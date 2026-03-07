#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:07:07 2019

@author: lheller
"""
import numpy as np
import math
def orilistMult(Mats,Dr):
    #list of matrices (N,3,3).dot(vector Dr)
    Mr = np.reshape(Mats, (Mats.shape[0]*Mats.shape[1],Mats.shape[2]))
    Dr=np.array(Dr)/np.linalg.norm(Dr)
    data = Mr.dot(Dr)
    data = np.reshape(data,(int(data.shape[0]/3),3)).T
    return data

def eu2quat(phi1,Phi,phi2): #Euler angles to quaternions
    q = np.zeros((4))
    q[0]= np.cos(Phi/2)*np.cos((phi1+phi2)/2)
    q[1]= -np.sin(Phi/2)*np.cos((phi1-phi2)/2)
    q[2]= -np.sin(Phi/2)*np.sin((phi1-phi2)/2)
    q[3]= -np.cos(Phi/2)*np.sin((phi1+phi2)/2)
    
    if q[0]<0: q= -q
    
    return(q)
def np_euler_matrix(ai, aj, ak): 
    
    g=np.eye(3)
    s1, s2, s3 = np.sin(ai), np.sin(aj), np.sin(ak)
    c1, c2, c3 = np.cos(ai), np.cos(aj), np.cos(ak)
    
    g[0,0] = c1*c3-s1*s3*c2
    g[0,1] = s1*c3+c1*s3*c2
    g[0,2] = s3*s2    
    g[1,0] = -c1*s3-s1*c3*c2 
    g[1,1] = -s1*s3+c1*c3*c2
    g[1,2] = c3*s2 
    g[2,0] = s1*s2 
    g[2,1] = -c1*s2
    g[2,2] = c2   
    return g
def np_eulers_matrices(data,deg=False): 
    #data[0,:] three eulerangles
    if deg:
        ai=data[:,0]*np.pi/180
        aj=data[:,1]*np.pi/180
        ak=data[:,2]*np.pi/180
    else:
        ai=data[:,0]
        aj=data[:,1]
        ak=data[:,2]
        
    
    g=np.zeros((data.shape[0],3,3))
    s1, s2, s3 = np.sin(ai), np.sin(aj), np.sin(ak)
    c1, c2, c3 = np.cos(ai), np.cos(aj), np.cos(ak)
    
    g[:,0,0] = c1*c3-s1*s3*c2
    g[:,0,1] = s1*c3+c1*s3*c2
    g[:,0,2] = s3*s2    
    g[:,1,0] = -c1*s3-s1*c3*c2 
    g[:,1,1] = -s1*s3+c1*c3*c2
    g[:,1,2] = c3*s2 
    g[:,2,0] = s1*s2 
    g[:,2,1] = -c1*s2
    g[:,2,2] = c2       
    return g

def np_inverse_euler_matrix(ai, aj, ak): 
    U=np.eye(3)

    s1, s2, s3 = np.sin(ai), np.sin(aj), np.sin(ak)
    c1, c2, c3 = np.cos(ai), np.cos(aj), np.cos(ak)

    U[0,0] = c1*c3-s1*s3*c2
    U[0,1] = -c1*s3-s1*c3*c2
    U[0,2] = s1*s2    
    U[1,0] = s1*c3+c1*s3*c2 
    U[1,1] = -s1*s3+c1*c3*c2
    U[1,2] = -c1*s2 
    U[2,0] = s3*s2 
    U[2,1] = c3*s2
    U[2,2] = c2   
    return U
def ol_g_rtheta_rad(g):
    eps = 1.e-6;
    
    ptheta = np.arccos((g[0][0] + g[1][1] + g[2][2] - 1) / 2);
    r=[0.,0.,0.];
    if ((ptheta) < eps):
        r[0] = 1;
        r[1] = 0;
        r[2] = 0;
    elif ((ptheta) < (1 - eps)*np.pi):
        r[0] = (g[1][2] - g[2][1]) / (2 * np.sin(ptheta));
        r[1] = (g[2][0] - g[0][2]) / (2 * np.sin(ptheta));
        r[2] = (g[0][1] - g[1][0]) / (2 * np.sin(ptheta));
    else:
        r[0] = np.sqrt((g[0][0] + 1) / 2)
        r[1] = np.sqrt((g[1][1] + 1) / 2);
        r[2] = np.sqrt((g[2][2] + 1) / 2);
    m = r.index(max(r))
    for i in range(0,3):
        if not r==m:
            if g[i][m]<0:
                r[i] *= 1;
    return r,ptheta            
def np_ol_g_rtheta_rad(g):
    eps = 1.e-6;
    
    ptheta = np.arccos((np.trace(g) - 1) / 2);
    r=np.array([0.,0.,0.]);
    if ((ptheta) < eps):
        r[0] = 1;
        r[1] = 0;
        r[2] = 0;
    elif ((ptheta) < (1 - eps)*np.pi):
        r[0] = (g[1,2] - g[2,1]) / (2 * np.sin(ptheta));
        r[1] = (g[2,0] - g[0,2]) / (2 * np.sin(ptheta));
        r[2] = (g[0,1] - g[1,0]) / (2 * np.sin(ptheta));
    else:
        r[0] = np.sqrt((g[0,0] + 1) / 2)
        r[1] = np.sqrt((g[1,1] + 1) / 2);
        r[2] = np.sqrt((g[2,2] + 1) / 2);
    m = np.where(r==max(r))[0][0]
    for i in range(0,3):
        if not i==m:
            if g[i,m]<0:
                r[i] *= 1;
    return r,ptheta            

def ol_rtheta_g_rad(r, theta):

    g = [[0,0,0] for i in range(0,3)]

    g[0][0] = r[0] * r[0] * (1 - np.cos (theta)) + np.cos (theta);
    g[0][1] = r[0] * r[1] * (1 - np.cos (theta)) + r[2] * np.sin (theta);
    g[0][2] = r[0] * r[2] * (1 - np.cos (theta)) - r[1] * np.sin (theta);
    
    g[1][0] = r[1] * r[0] * (1 - np.cos (theta)) - r[2] * np.sin (theta);
    g[1][1] = r[1] * r[1] * (1 - np.cos (theta)) + np.cos (theta);
    g[1][2] = r[1] * r[2] * (1 - np.cos (theta)) + r[0] * np.sin (theta);
    
    g[2][0] = r[2] * r[0] * (1 - np.cos (theta)) + r[1] * np.sin (theta);
    g[2][1] = r[2] * r[1] * (1 - np.cos (theta)) - r[0] * np.sin (theta);
    g[2][2] = r[2] * r[2] * (1 - np.cos (theta)) + np.cos (theta);

    return g

def np_ol_rtheta_g_rad(r, theta):

    g=np.eye(3)

    g[0,0] = r[0] * r[0] * (1 - np.cos (theta)) + np.cos (theta);
    g[0,1] = r[0] * r[1] * (1 - np.cos (theta)) + r[2] * np.sin (theta);
    g[0,2] = r[0] * r[2] * (1 - np.cos (theta)) - r[1] * np.sin (theta);
    
    g[1,0] = r[1] * r[0] * (1 - np.cos (theta)) - r[2] * np.sin (theta);
    g[1,1] = r[1] * r[1] * (1 - np.cos (theta)) + np.cos (theta);
    g[1,2] = r[1] * r[2] * (1 - np.cos (theta)) + r[0] * np.sin (theta);
    
    g[2,0] = r[2] * r[0] * (1 - np.cos (theta)) + r[1] * np.sin (theta);
    g[2,1] = r[2] * r[1] * (1 - np.cos (theta)) - r[0] * np.sin (theta);
    g[2,2] = r[2] * r[2] * (1 - np.cos (theta)) + np.cos (theta);

    return g


def ol_g_R(g):
    #Quey
    r,theta = ol_g_rtheta_rad (g)
    R=[0.,0.,0.]
    for i in range(0,3):
        R[i]=r[i]*np.tan(theta/2)
    return R

def np_ol_g_R(g):
    #Quey
    r,theta = np_ol_g_rtheta_rad (g)
    R=r*np.tan(theta/2)

    return R

def ol_R_g (R):

  norm = np.sqrt(sum([ri*ri for ri in R]))
  r = [ri/norm for ri in R]
  theta = 2*np.arctan(norm)
  

  g=ol_rtheta_g_rad(r, theta);


  return g 

def np_ol_R_g (R):

  norm = np.sqrt(R.dot(R))
  r=R/norm
  theta = 2*np.arctan(norm)
  

  g=np_ol_rtheta_g_rad(r, theta);


  return g 


        
def ol_g_R2(g):
    #Poulsen
    gmm = g[0][0]+g[1][1]+g[2][2]
    R=[0.,0.,0.]
    epsilon = permut_tensor3()
    delta = kronecker()
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                R[i]=R[i]+(epsilon[i][j][k]*g[j][k])/(1+gmm)
                
    return R

def np_ol_g_R2(g,epsilon, delta):
    #Poulsen
    gmm = np.trace(g)
    R = np.einsum('ijk,jk',epsilon,g)/(1+gmm)    
    return R

def ol_R_g2(R):
    #Poulsen
    r2=sum([ri*ri for ri in R])
    
    epsilon = permut_tensor3()
    delta = kronecker()
    g=[]
    for i in range(0,3):
        gj=[]
        for j in range(0,3):
            er=0.
            for k in range(0,3):
                er=er+2*epsilon[i][j][k]*R[k]
            gj.append(1./(1+r2)*((1-r2)*delta[i][j]+2*R[i]*R[j]+er))
        g.append(gj)
            
                    
    return g

def np_ol_R_g2(R,epsilon, delta):
    #Poulsen
    r2=R.dot(R)
    g= 1./(1+r2)*((1-r2)*delta+2*np.einsum('i,j',R,R)+np.einsum('ijk,k',2*epsilon,R))
    return g

def np_ol_R_q2(R):
    #Poulsen
    q=np.empty(4)
    r2 = R.dot(R)
    q[0]=1./np.sqrt(1+r2);
    q[1:]=R/np.sqrt(1+r2)

    return q

def np_ol_g_q2(g):
    #Poulsen
    eps = 1e-6;
    q=np.empty(4)
    q[0] = 0.5*np.sqrt(np.trace(g)+1)
    
    if abs(q[0]) > eps:
        q[1]=1./4./q[0]*(g[2,1]-g[1,2])
        q[2]=1./4./q[0]*(g[2,0]-g[0,2])
        q[3]=1./4./q[0]*(g[0,1]-g[1,0])
    else:
        for i in range(0,2):
            q[i+1]=np.sqrt((g[i,i]+1)/2)
        
        m = 1+np.where(q[1:]==max(q[1:]))[0][0]
        for i in range(0,3):
            q[i]*=np.sign(g[i - 1][m - 1])
            
            
    
    return q

def np_ol_q_g(q):
    #Poulsen
    g=np.empty((3,3))
    
    g[0,0]=q[0]**2+q[1]**2-q[2]**2-q[3]**2
    g[1,1]=q[0]**2-q[1]**2+q[2]**2-q[3]**2
    g[2,2]=q[0]**2-q[1]**2-q[2]**2+q[3]**2
    
    for i in range(0,2):
        g[i,i]=2*(q[0]**2+q[i+1]**2)-1
    
    g[1,0] = 2*(q[1]*q[2]-q[0]*q[3])
    g[0,1] = 2*(q[1]*q[2]+q[0]*q[3])

    g[2,0] = 2*(q[1]*q[3]+q[0]*q[2])
    g[0,2] = 2*(q[1]*q[3]-q[0]*q[2])
    
    g[2,1] = 2*(q[2]*q[3]-q[0]*q[1])
    g[1,2] = 2*(q[2]*q[3]+q[0]*q[1])

    return g

def orilistMult(Mats,Dr):
    #list of matrices (N,3,3).dot(vector Dr)
    Mr = np.reshape(Mats, (Mats.shape[0]*Mats.shape[1],Mats.shape[2]))
    Dr=np.array(Dr)/np.linalg.norm(Dr)
    data = Mr.dot(Dr)
    data = np.reshape(data,(int(data.shape[0]/3),3)).T
    return data
def symposMult(sympos,Mats):
    # The product of two 2D matrices (numpy ndarray shape(N,N)) can be calculated
    # using the function 'numpy.dot'. In order to compute the matrix product of
    # higher dimensions arrays, numpy.dot can also be used, but paying careful
    # attention to the indices of the resulting matrix. Examples:
    #     - A is ndarray shape(N,M,3,3) and B is ndarray shape(3,3):
    #     np.dot(A,B)[i,j,k,m] = np.sum(A[i,j,:,k]*B[m,:])
    #     np.dot(A,B) is ndarray shape(N,M,3,3)

    #     - A is ndarray shape(N,3,3) and B is ndarray shape(M,3,3):
    #     np.dot(A,B)[i,j,k,m] = np.sum(A[i,:,j]*B[k,m,:])
    #
    #     The result np.dot(A,B) is ndarray shape(N,3,M,3). It's more convenient to
    #     express the result as ndarray shape(N,M,3,3). In order to obtain the
    #     desired result, the 'transpose' function should be used. i.e.,
    #     np.dot(A,B).transpose([0,2,1,3]) results in ndarray shape(N,M,3,3)

    #     - A is ndarray shape(3,3) and B is ndarray shape(N,M,3,3):
    #     np.dot(A,B)[i,j,k,m] = np.sum(A[:,i]*B[j,k,m,:])
    #
    #     np.dot(A,B) is ndarray shape(3,N,M,3). np.dot(A,B).transpose([1,2,0,3])
    #     results in ndarray shape(N,M,3,3)

    # 'numpy.dot' is a particular case of 'numpy.tensordot':
    # np.dot(A,B) == np.tensordot(A, B, axes=[[-1],[-2]])

    # numpy.tensordot is two times faster than numpy.dot

    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
    #list of symmetry matrices sympos (M,3,3) x list of orientation matrices Mats (N,3,3)
    if type(sympos) == list:
        sympos=np.array(sympos)
        
    MT=np.dot(sympos,Mats).transpose([0,2,1,3])#shape=(M.N,3,3)
    MT=MT.reshape((MT.shape[0]*MT.shape[1],3,3))#Contraction to (N*M,3,3)
    return MT
    
def active_rotation(an, aboutaxis, deg=False):
    if deg:
        an=an*np.pi/180.        
    if aboutaxis.lower()=='z':
        R = np.array([[np.cos(an),-np.sin(an),0],[np.sin(an),np.cos(an),0],[0,0,1]]);
    elif aboutaxis.lower()=='x':
        R = np.array([[1,0,0],[0,np.cos(an),-np.sin(an)],[0,np.sin(an),np.cos(an)]]);
    elif aboutaxis.lower()=='y':
        R = np.array([[np.cos(an),0,np.sin(an)],[0,1,0],[-np.sin(an),0,np.cos(an)]]);
    
    return R
    

def passive_rotation(an, aboutaxis, deg=False):
    R=np.transpose(active_rotation(an, aboutaxis, deg=deg));    
    
    return R
    
def symmetry_elements(lattice):
    U=[]
    #identity
    U.append(np.eye(3))
    #U.append(-1*np.eye(3))
    if lattice.lower()=='cubic':
        #3xpi/2 Rotations about 100,010,001=>9 operations
        U.append(np.array([[1,0,0],[0,0,-1],[0,1,0]]).T)
        U.append(np.array([[1,0,0],[0,-1,0],[0,0,-1]]).T)
        U.append(np.array([[1,0,0],[0,0,1],[0,-1,0]]).T)
        
        
        U.append(np.array([[0,0,1],[0,1,0],[-1,0,0]]).T)
        U.append(np.array([[-1,0,0],[0,1,0],[0,0,-1]]).T)
        U.append(np.array([[0,0,-1],[0,1,0],[1,0,0]]).T)


        U.append(np.array([[0,-1,0],[1,0,0],[0,0,1]]).T)
        U.append(np.array([[-1,0,0],[0,-1,0],[0,0,1]]).T)
        U.append(np.array([[0,1,0],[-1,0,0],[0,0,1]]).T)


        #1xpi Rotation about [110][-110][011][0-11][101][10-1]
        U.append(np.array([[0,1,0],[1,0,0],[0,0,-1]]).T)
        U.append(np.array([[-1,0,0],[0,0,1],[0,1,0]]).T)
        U.append(np.array([[0,0,1],[0,-1,0],[1,0,0]]).T)
        U.append(np.array([[0,-1,0],[-1,0,0],[0,0,-1]]).T)
        U.append(np.array([[-1,0,0],[0,0,-1],[0,-1,0]]).T)
        U.append(np.array([[0,0,-1],[0,-1,0],[-1,0,0]]).T)
        
        #2xpi/3 rotations about [111][11-1][-111][-11-1]
        U.append(np.array([[0,0,1],[1,0,0],[0,1,0]]).T)
        U.append(np.array([[0,1,0],[0,0,1],[1,0,0]]).T)
        U.append(np.array([[0,-1,0],[0,0,1],[-1,0,0]]).T)
        U.append(np.array([[0,0,-1],[-1,0,0],[0,1,0]]).T)
        U.append(np.array([[0,-1,0],[0,0,-1],[1,0,0]]).T)
        U.append(np.array([[0,0,1],[-1,0,0],[0,-1,0]]).T)
        U.append(np.array([[0,1,0],[0,0,-1],[-1,0,0]]).T)
        U.append(np.array([[0,0,-1],[1,0,0],[0,-1,0]]).T)
    Un=[]
    for u in U:
        Un.append(u)
        Un.append(-1*u)
        
    
    
    #print(len(Un))

        
#        for i in range(0,len(U)):
#            for j in range(0,len(U)):
#                if (U[i]==U[j]).all() and i<>j:
#                    print('spatne')
#        
    return Un

def euler_angles_reduction(Phi1,PHI,Phi2):

    if not type(Phi1)==list:
        Phi1 = [Phi1]
        PHI = [PHI]
        Phi2 = [Phi2]
    Phi1_red=[]
    Phi2_red=[]
    PHI_red=[]
    
    for phi1,phi,phi2 in zip(Phi1,PHI,Phi2):
    #converting PHI to 0-2*pi
        phi = phi-round(phi/(2*np.pi))
        if phi<0:
            phi=phi+2*np.pi

        #if phi>PI, applying reflection */
        #phi becomes within [0,PI] */

        if phi>np.pi:
            phi=2*np.pi-phi
            phi1 = phi1 + np.pi
            phi2 = phi2 + np.pi

        #treating the std case where phi != 0 */
        if (abs(phi) > 1e-6 and abs(phi-np.pi)> 1e-6):
            # ranging phi2 within [0,(2*np.pi)] */
            phi2 = phi2-round(phi2/(2*np.pi))
            if phi2<0:
                phi2 =phi2 + 2*np.pi
        
        
        # treating degeneracy: phi = 0: phi1 += phi2 and phi2 = 0. */
        elif (abs(phi) > 1e-6):
            phi1=phi1+phi2
            phi2 = 0
        else: # the same at phi = 180 */
            phi1=phi1-phi2
            phi2 = 0
        
        # ranging phi1 within [0,(2*PI)] */
        phi1 = phi1-round(phi1/(2*np.pi))
        if phi1<0:
            phi1 = phi1 + 2*np.pi

        Phi1_red.append(phi1)
        Phi2_red.append(phi2)
        PHI_red.append(phi)
    if len(Phi1_red)==1:
        return Phi1_red[0],PHI_red[0],Phi2_red[0]
    else:
        return Phi1_red,PHI_red,Phi2_red


def euler_angles_from_matrix(Rl):

    if not type(Rl)==list:
        Rl=[Rl]
        
    Phi1=[]
    Phi2=[]
    PHI=[]
    for R in Rl:
        PHI.append(np.arccos(R[2,2]))
        if PHI[-1]==0.0:
           Phi1.append(np.arctan2(-R[1,0],R[0,0]))
           Phi2.append(0.0)
        elif PHI[-1]==np.pi:
           Phi1.append(np.arctan2(R[1,0],R[0,0]))
           Phi2.append(0.0)
        else:
           Phi1.append(np.arctan2(R[2,0],-R[2,1]))
           Phi2.append(np.arctan2(R[0,2],R[1,2]))

    if len(Phi1)==1:
        return Phi1[0],PHI[0],Phi2[0]
    else:
        return Phi1,PHI,Phi2

def misorimat_ini(umatsa):
    Q=Mat2Quat(umatsa)
    #print(Q)
    Qinv = Q.copy()
    Qinv[1:4,:]=-1*Qinv[1:4,:]
    #QP=Qproduct(Qinv,Q)
    #print(Qinv)
    d=np.linalg.norm(Qlog(Qproduct(Qinv,Q)),axis=0)*180.0/np.pi*2
    d2=np.linalg.norm(Qlog(Qproduct(Qinv,-1*Q)),axis=0)*180.0/np.pi*2
    d[np.where(d2<d)]= d2[np.where(d2<d)]
    return d
def misorimat(umatsa):
    Q=Mat2Quat(umatsa)
    #print(Q)
    Qinv = Q.copy()
    Qinv[1:4,:]=-1*Qinv[1:4,:]
    #QP=Qproduct(Qinv,Q)
    #print(Qinv)
    #d=np.linalg.norm(Qlog(Qproduct(Qinv,Q)),axis=0)*180.0/np.pi*2
    d=np.linalg.norm(Qlog(Qproduct(Q,Qinv)),axis=0)*180.0/np.pi*2
    #d[np.where(d2<d)]= d2[np.where(d2<d)]
    return d


def disorimat_test02(umatsa,symops):
    Q=Mat2Quat(umatsa)
    #Qinv = Q.copy()
    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    symq=Mat2Quat(symops)
    #SQ=Qproduct(symq,Q)
    #SQ[1:4,:,:]=-1*SQ[1:4,:,:]

    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    #symq=mat2quat(symops)
    #SQ=Qproduct(symq,Qinv)
    #print(SQ)
    ds=[]
    for sym in symq.T:
        #print("Symmetry operation {} of {}".format(str(i),str(SQ.shape[1])))
        #print(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Q)),axis=0)[0,1]*180.0/np.pi*2)
        #print(SQ[:,i,:])
        #SQ=Q.copy()
        SQ=QMatproduct(sym,Q)
        SQ[1:4,:]=-1*SQ[1:4,:]
        QA=np.hstack((Q,SQ))
        #print(QA)
        QAinv=QA.copy()
        QAinv[1:4,:]=-1*QAinv[1:4,:]
        d=np.linalg.norm(Qlog(Qproduct(QAinv,QA)),axis=0)*180.0/np.pi*2
        d2=np.linalg.norm(Qlog(Qproduct(QAinv,-1*QA)),axis=0)*180.0/np.pi*2
        d[np.where(d2<d)]= d2[np.where(d2<d)]
        ds.append(d)
        print(d)
        print(d2)
    DS=np.amin(abs(np.array(ds)),axis=0)  
    #print(DS)     
    #print(abs(np.array(ds))*180.0/np.pi*2 )
    #ds2=[]
    #for i in range(0,SQ.shape[1]):
        #print("Symmetry operation {} of {}".format(str(i),str(SQ.shape[1])))
        #print(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],-1*Q)),axis=0)[0,1]*180.0/np.pi*2 )
    #    ds2.append(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],-1*Q)),axis=0))
        #print(ds2[-1]*180/np.pi)
    #    print(ds2[-1]*180.0/np.pi*2)
    #print((abs(np.array(ds2))*180.0/np.pi*2).shape )
    #DS2=np.amin(abs(np.array(ds2)),axis=0)*180.0/np.pi*2    
    
    #idxs=np.where(DS2<DS)
    #DS[idxs]= DS2[idxs]     
    return DS,ds,ds,SQ


def disorimat_test01(umatsa,symops):
    
    Q=Mat2Quat(umatsa)
    #Qinv = Q.copy()
    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    symq=Mat2Quat(symops)
    #SQ=Qproduct(symq,Qinv)
    #SQ[1:4,:,:]=-1*SQ[1:4,:,:]

    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    #symq=mat2quat(symops)
    #SQ=Qproduct(symq,Qinv)
    #print(SQ)
    ds=[]
    for sym in symq.T:
        #print("Symmetry operation {} of {}".format(str(i),str(SQ.shape[1])))
        #print(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Q)),axis=0)[0,1]*180.0/np.pi*2)
        SQi=QMatproduct(sym,Q)
        SQ=SQi.copy()
        SQ[0,:]=-SQ[-1,:]
        SQ[1:4,:]=-1*SQ[1:4,:]
        ds.append(np.linalg.norm(Qlog(Qproduct(SQ,Q)),axis=0))
        print(ds[-1]*180.0/np.pi*2)
    DS=np.amin(abs(np.array(ds)),axis=0)*180.0/np.pi*2       
    #print(abs(np.array(ds))*180.0/np.pi*2 )
    ds2=[]
    for sym in symq.T:
        #print("Symmetry operation {} of {}".format(str(i),str(SQ.shape[1])))
        #print(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],-1*Q)),axis=0)[0,1]*180.0/np.pi*2 )
        SQ=QMatproduct(sym,Q)
        SQ[1:4,:]=-1*SQ[1:4,:]
        ds2.append(np.linalg.norm(Qlog(Qproduct(SQ,-1*Q)),axis=0))
        #print(ds2[-1]*180/np.pi)
        print(ds2[-1]*180.0/np.pi*2)
    #print((abs(np.array(ds2))*180.0/np.pi*2).shape )
    DS2=np.amin(abs(np.array(ds2)),axis=0)*180.0/np.pi*2    
    
    idxs=np.where(DS2<DS)
    DS[idxs]= DS2[idxs]     
    return DS,ds,ds2,SQ

def disorimat_ini(umatsa,symops):
    print('test')
    Q=Mat2Quat(umatsa)
    Qinv = Q.copy()
    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    symq=Mat2Quat(symops)
    SQ=Qproduct(symq,Qinv)
    SQ[1:4,:,:]=-1*SQ[1:4,:,:]

    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    #symq=mat2quat(symops)
    #SQ=Qproduct(symq,Qinv)
    #print(SQ)
    ds=[]
    for i in range(0,SQ.shape[1]):
        #print("Symmetry operation {} of {}".format(str(i),str(SQ.shape[1])))
        #print(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Q)),axis=0)[0,1]*180.0/np.pi*2)
        ds.append(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Q)),axis=0))
        print(ds[-1]*180.0/np.pi*2)
    DS=np.amin(abs(np.array(ds)),axis=0)*180.0/np.pi*2       
    #print(abs(np.array(ds))*180.0/np.pi*2 )
    ds2=[]
    for i in range(0,SQ.shape[1]):
        #print("Symmetry operation {} of {}".format(str(i),str(SQ.shape[1])))
        #print(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],-1*Q)),axis=0)[0,1]*180.0/np.pi*2 )
        ds2.append(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],-1*Q)),axis=0))
        #print(ds2[-1]*180/np.pi)
        print(ds2[-1]*180.0/np.pi*2)
    #print((abs(np.array(ds2))*180.0/np.pi*2).shape )
    DS2=np.amin(abs(np.array(ds2)),axis=0)*180.0/np.pi*2    
    
    idxs=np.where(DS2<DS)
    DS[idxs]= DS2[idxs]     
    return DS,ds,ds2,SQ

def disorimat_ini(umatsa,symops):
    #print('test4')
    Q=Mat2Quat(umatsa)
    symq=Mat2Quat(symops)
    SQ=Qproduct(symq,Q)
    SQ[1:4,:,:]=-1*SQ[1:4,:,:]

    #Qinv=Q.copy()
    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    #symq=mat2quat(symops)
    #SQ=Qproduct(symq,Qinv)
    #print(SQ)
    ds=[]
    for i in range(0,SQ.shape[1]):
        print("Symmetry operation {} of {}".format(str(i),str(SQ.shape[1])))
        #print(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Q)),axis=0)[0,1]*180.0/np.pi*2)
        d=np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Q)),axis=0)*180.0/np.pi*2
        d2=np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],-1*Q)),axis=0)*180.0/np.pi*2
        d[np.where(d2<d)]= d2[np.where(d2<d)]
        ds.append(d)
#        d=np.linalg.norm(Qlog(Qproduct(Qinv,SQ[:,i,:])),axis=0)*180.0/np.pi*2
#        d2=np.linalg.norm(Qlog(Qproduct(-1*Qinv,SQ[:,i,:])),axis=0)*180.0/np.pi*2
#        d[np.where(d2<d)]= d2[np.where(d2<d)]
#        ds.append(d)
    DS=np.amin(abs(np.array(ds)),axis=0)     
    return DS
def disorimat(umatsa,symops,prnt=False,withfirst=False,eqmats=False):
    #print('test4')
    Q=Mat2Quat(umatsa)
    symq=Mat2Quat(symops)
    SQ=Qproduct(symq,Q)
    SQ[1:4,:,:]=SQ[1:4,:,:]

    Qinv=Q.copy()
    if withfirst:
        Qinv=Qinv[:,0:1]
    Qinv[1:4,:]=-1*Qinv[1:4,:]
    #Qinv[1:4,:]=-1*Qinv[1:4,:]
    #symq=mat2quat(symops)
    #SQ=Qproduct(symq,Qinv)
    #print(SQ)
    ds=[]
    for i in range(0,SQ.shape[1]):
        if prnt:
            print("Symmetry operation {} of {}".format(str(i),str(SQ.shape[1])))
        #print(np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Q)),axis=0)[0,1]*180.0/np.pi*2)
        #d=np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Q)),axis=0)*180.0/np.pi*2
        d=np.linalg.norm(Qlog(Qproduct(SQ[:,i,:],Qinv)),axis=0)*180.0/np.pi*2
        #d[np.where(d2<d)]= d2[np.where(d2<d)]
        ds.append(d)
#        d=np.linalg.norm(Qlog(Qproduct(Qinv,SQ[:,i,:])),axis=0)*180.0/np.pi*2
#        d2=np.linalg.norm(Qlog(Qproduct(-1*Qinv,SQ[:,i,:])),axis=0)*180.0/np.pi*2
#        d[np.where(d2<d)]= d2[np.where(d2<d)]
#        ds.append(d)
    DS=np.amin(abs(np.array(ds)),axis=0)  
    if eqmats:
        return DS,np.array(symops)[np.argmin(abs(np.array(ds)),axis=0)[:,0],:,:]
    else:
        return DS





def mat2quat02(matrix): #orientation matrix to quaternion
    q0 = np.sqrt(1 + matrix[0,0] + matrix[1,1] + matrix[2,2])
    q1 = np.sqrt(1 + matrix[0,0] - matrix[1,1] - matrix[2,2])
    q2 = np.sqrt(1 - matrix[0,0] + matrix[1,1] - matrix[2,2])
    q3 = np.sqrt(1 - matrix[0,0] - matrix[1,1] + matrix[2,2])
    
    if matrix[2,1]<matrix[1,2]: q1 = -q1
    if matrix[0,2]<matrix[2,0]: q2 = -q2
    if matrix[1,0]<matrix[0,1]: q3 = -q3
    
    q = 1./2* np.array([q0,q1,q2,q3])
    q /= np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    
    return q
def Mat2Quat_ini(umatsa): #orientation matrix to quaternion
    q0 = [np.sqrt(1 + matrix[0,0] + matrix[1,1] + matrix[2,2]) for matrix in umatsa]
    q1 = [np.sqrt(1 + matrix[0,0] - matrix[1,1] - matrix[2,2]) for matrix in umatsa]
    q2 = [np.sqrt(1 - matrix[0,0] + matrix[1,1] - matrix[2,2]) for matrix in umatsa]
    q3 = [np.sqrt(1 - matrix[0,0] - matrix[1,1] + matrix[2,2]) for matrix in umatsa]

    q1 = [-qi if matrix[2,1]<matrix[1,2] else qi for matrix,qi in zip(umatsa,q1)]
    q2 = [-qi if matrix[0,2]<matrix[2,0] else qi for matrix,qi in zip(umatsa,q2)]
    q3 = [-qi if matrix[1,0]<matrix[0,1] else qi for matrix,qi in zip(umatsa,q3)]
    
    
    #if matrix[2,1]<matrix[1,2]: q1 = -q1
    #if matrix[0,2]<matrix[2,0]: q2 = -q2
    #if matrix[1,0]<matrix[0,1]: q3 = -q3
    Q=0.5*np.stack((q0,q1,q2,q3));
    Q = Q / np.linalg.norm(Q, axis=0)
    Q[:,np.where(np.prod(Q[1:4,:]<0,axis=0)==1)[0]]=-1*Q[:,np.where(np.prod(Q[1:4,:]<0,axis=0)==1)[0]]
    return Q
def Mat2Quat(umatsa): #orientation matrix to quaternion
#assumes umatsa = array[:,3,3]
    """
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Altered to work with the column vector convention instead of row vectors
    """
    #m = matrix#.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
    
    #try:
    #    umatsa.shape
    #    umatsa=np.array([m.conj().transpose() for m in umatsa])
    #except:
    umatsa=np.array([m.conj().transpose() for m in umatsa])
        
    M22L0=np.where(umatsa[:,2,2]<0)[0]
    M22GE0=np.where(umatsa[:,2,2]>=0)[0]
    if M22L0.shape[0]>0:
        M00GM11=np.where(umatsa[M22L0,0,0]>umatsa[M22L0,1,1])[0]
        M00LEM11=np.where(umatsa[M22L0,0,0]<=umatsa[M22L0,1,1])[0]
    else:
        M00GM11=np.array([])
        M00LEM11=np.array([])
        
    if M22GE0.shape[0]>0:
        M00LnM11=np.where(umatsa[M22GE0,0,0]<-1*umatsa[M22GE0,1,1])[0]  
        M00GEnM11=np.where(umatsa[M22GE0,0,0]>=-1*umatsa[M22GE0,1,1])[0]
    else:
       M00LnM11=np.array([])
       M00GEnM11=np.array([])
    Q=np.empty((4,umatsa.shape[0]))
    T=np.empty((umatsa.shape[0]))
    try:
        T[M22L0[M00GM11]] = 1 + umatsa[M22L0[M00GM11],0,0] - umatsa[M22L0[M00GM11],1,1] - umatsa[M22L0[M00GM11],2,2]
        Q[:,M22L0[M00GM11]] = [umatsa[M22L0[M00GM11],1, 2]-umatsa[M22L0[M00GM11],2, 1],  T[M22L0[M00GM11]],  umatsa[M22L0[M00GM11],0, 1]+umatsa[M22L0[M00GM11],1, 0],  
                               umatsa[M22L0[M00GM11],2, 0]+umatsa[M22L0[M00GM11],0, 2]]
    except:
        pass
    
    try:
        T[M22L0[M00LEM11]]  = 1 - umatsa[M22L0[M00LEM11],0, 0] + umatsa[M22L0[M00LEM11],1, 1] - umatsa[M22L0[M00LEM11],2, 2]
        Q[:,M22L0[M00LEM11]] = [umatsa[M22L0[M00LEM11],2, 0]-umatsa[M22L0[M00LEM11],0, 2],  umatsa[M22L0[M00LEM11],0, 1]+umatsa[M22L0[M00LEM11],1, 0],
                                T[M22L0[M00LEM11]],  umatsa[M22L0[M00LEM11],1, 2]+umatsa[M22L0[M00LEM11],2, 1]]    
    except:
        pass    
    try:
        T[M22GE0[M00LnM11]] = 1 - umatsa[M22GE0[M00LnM11],0, 0] - umatsa[M22GE0[M00LnM11],1, 1] + umatsa[M22GE0[M00LnM11],2, 2]
        Q[:,M22GE0[M00LnM11]] = [umatsa[M22GE0[M00LnM11],0, 1]-umatsa[M22GE0[M00LnM11],1, 0],  umatsa[M22GE0[M00LnM11],2, 0]+umatsa[M22GE0[M00LnM11],0, 2],  
             umatsa[M22GE0[M00LnM11],1, 2]+umatsa[M22GE0[M00LnM11],2, 1],T[M22GE0[M00LnM11]]]
    except:
        pass
    try:
        T[M22GE0[M00GEnM11]] = 1 + umatsa[M22GE0[M00GEnM11],0, 0] + umatsa[M22GE0[M00GEnM11],1, 1] + umatsa[M22GE0[M00GEnM11],2, 2]
        Q[:,M22GE0[M00GEnM11]] =[T[M22GE0[M00GEnM11]],  umatsa[M22GE0[M00GEnM11],1, 2]-umatsa[M22GE0[M00GEnM11],2, 1],  umatsa[M22GE0[M00GEnM11],2, 0]-umatsa[M22GE0[M00GEnM11],0, 2],  
                                 umatsa[M22GE0[M00GEnM11],0, 1]-umatsa[M22GE0[M00GEnM11],1, 0]]
    except:
        pass

    Q[0,:] *= 0.5 / np.sqrt(T)
    Q[1,:] *= 0.5 / np.sqrt(T)
    Q[2,:] *= 0.5 / np.sqrt(T)
    Q[3,:] *= 0.5 / np.sqrt(T)
    return Q


def Qlog(QM): 
    qlog=QM.copy()
    qlog[0,:,:]=qlog[0,:,:]*0
    qlog[1:4,range(qlog.shape[1]),range(qlog.shape[2])]=1
    #np.tile(np.arccos(QM[0,:,:]/np.linalg.norm(QM[:,:,:], axis=0)),(3,1,1)).shape
    qlog[1:4,:,:] = (qlog[1:4,:,:] / np.linalg.norm(qlog[1:4,:,:], axis=0))*np.tile(np.arccos(QM[0,:,:]/np.linalg.norm(QM[:,:,:], axis=0)),(3,1,1))
    return qlog

def Qproduct(P,Q):
    Ones=np.ones(Q.shape)
    Ones[1:4,:]=-1*Ones[1:4,:]
    Q0=P.T.dot(Q*Ones)
    idxs=[[0,1],[1,0],[2,3],[3,2]]
    signs=[1,1,1,-1]
    Q1=np.zeros(Q0.shape)
    for idx,sgn in zip(idxs,signs):
        p1=np.zeros(P.shape)
        p1[0,:]=P[idx[0],:]    
        q1=np.zeros(Q.shape)
        q1[0,:]=sgn*Q[idx[1],:]
        Q1+=p1.T.dot(q1)

    idxs=[[0,2],[2,0],[1,3],[3,1]]
    signs=[1,1,-1,1]
    Q2=np.zeros(Q0.shape)
    for idx,sgn in zip(idxs,signs):
        p1=np.zeros(P.shape)
        p1[0,:]=P[idx[0],:]    
        q1=np.zeros(Q.shape)
        q1[0,:]=sgn*Q[idx[1],:]
        Q2+=p1.T.dot(q1)
    idxs=[[0,3],[3,0],[1,2],[2,1]]
    signs=[1,1,1,-1]
    Q3=np.zeros(Q0.shape)
    for idx,sgn in zip(idxs,signs):
        p1=np.zeros(P.shape)
        p1[0,:]=P[idx[0],:]    
        q1=np.zeros(Q.shape)
        q1[0,:]=sgn*Q[idx[1],:]
        Q3+=p1.T.dot(q1)
    return np.stack((Q0,Q1,Q2,Q3))
def QMatproduct(sym,Q):
    SQ=Q.copy()
    SQ[0,:]=sym[0]*Q[0,:]-sym[1]*Q[1,:]-sym[2]*Q[2,:]-sym[3]*Q[3,:]
    SQ[1,:]=sym[0]*Q[1,:]+sym[1]*Q[0,:]+sym[2]*Q[3,:]-sym[3]*Q[2,:]
    SQ[2,:]=sym[0]*Q[2,:]+sym[2]*Q[0,:]-sym[1]*Q[3,:]+sym[3]*Q[1,:]
    SQ[3,:]=sym[0]*Q[3,:]+sym[3]*Q[0,:]+sym[1]*Q[2,:]-sym[2]*Q[1,:]
    # SQ[0,:]=sym[0,:]*Q[0,:]-sym[1,:]*Q[1,:]-sym[2,:]*Q[2,:]-sym[3,:]*Q[3,:]
    # SQ[1,:]=sym[0,:]*Q[1,:]+sym[1,:]*Q[0,:]+sym[2,:]*Q[3,:]-sym[3,:]*Q[2,:]
    # SQ[2,:]=sym[0,:]*Q[2,:]+sym[2,:]*Q[0,:]-sym[1,:]*Q[3,:]+sym[3,:]*Q[1,:]
    # SQ[3,:]=sym[0,:]*Q[3,:]+sym[3,:]*Q[0,:]+sym[1,:]*Q[2,:]-sym[2,:]*Q[1,:]
    return SQ



def grid_s1(resol, grids=6):
    number_points = (2 ** resol) * grids

    interval = 2 * np.pi / number_points

    points = [interval / 2 + i * interval for i in range(number_points)]

    return points


def hopf2quat(Points):
    quats = []

    for i in range(len(Points)):
        x4 = math.sin(Points[i][0] / 2) * math.sin(Points[i][1] + Points[i][2] / 2)

        x1 = math.cos(Points[i][0] / 2) * math.cos(Points[i][2] / 2)

        x2 = math.cos(Points[i][0] / 2) * math.sin(Points[i][2] / 2)

        x3 = math.sin(Points[i][0] / 2) * math.cos(Points[i][1] + Points[i][2] / 2)

        quats.append([x1, x2, x3, x4])

    return quats


def nside2npix(nside):
    return 12 * nside * nside


def pix2ang_nest(nside, ipix, pix2x, pix2y):
    jrll = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

    jpll = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])

    if (nside < 1) or (nside > 8192):
        raise Exception('nside out of range:', nside)

    if (ipix < 0) or (ipix > 12 * nside * nside - 1):
        raise Exception('ipix out of range:', ipix)

    fn = 1. * nside

    fact1 = 1. / (3. * fn * fn)

    fact2 = 2. / (3. * fn)

    nl4 = 4 * nside

    npface = nside * nside

    face_num = int(ipix / npface)

    ipf = int(ipix % npface)

    ip_low = int(ipf % 1024)

    ip_trunc = ipf / 1024

    ip_med = int(ip_trunc % 1024)

    ip_hi = int(ip_trunc / 1024)

    ix = 1024 * pix2x[ip_hi] + 32 * pix2x[ip_med] + pix2x[ip_low]

    iy = 1024 * pix2y[ip_hi] + 32 * pix2y[ip_med] + pix2y[ip_low]

    jrt = ix + iy

    jpt = ix - iy

    jr = jrll[face_num] * nside - jrt - 1

    nr = nside

    z = (2 * nside - jr) * fact2

    kshift = int((jr - nside) % 2)

    if jr < nside:

        nr = jr

        z = 1. - nr * nr * fact1

        kshift = 0

    elif jr > 3 * nside:

        nr = nl4 - jr

        z = - 1. + nr * nr * fact1

        kshift = 0

    theta = np.arccos(z)

    jp = (jpll[face_num] * nr + jpt + 1 + kshift) / 2

    if jp > nl4:
        jp = jp - nl4

    if jp < 1:
        jp = jp + nl4

    phi = (jp - (kshift + 1) * 0.5) * (np.pi / 2 / nr)

    return theta, phi


def mk_pix2xy():
    pix2x = []

    pix2y = []

    for kpix in range(1024):

        jpix = kpix

        IX = 0

        IY = 0

        IP = 1

        while jpix != 0:
            ID = int(jpix % 2)

            jpix /= 2

            IX = ID * IP + IX

            ID = int(jpix % 2)

            jpix /= 2

            IY = ID * IP + IY

            IP = 2 * IP

        pix2x.append(IX)

        pix2y.append(IY)

    return pix2x, pix2y


def simple_grid(resol):
    Psi_Points = grid_s1(resol)

    Nside = 2 ** resol

    numpixels = nside2npix(Nside)

    pix2x, pix2y = mk_pix2xy()

    Healpix_Points = []

    for i in range(numpixels):
        theta, phi = pix2ang_nest(Nside, i, pix2x, pix2y)

        Healpix_Points.append([theta, phi])

    S3_Points = [[theta, phi, psi] for [theta, phi] in Healpix_Points for psi in Psi_Points]

    quats = hopf2quat(S3_Points)

    return quats

    
