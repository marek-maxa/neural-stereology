#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:06:11 2019

@author: lheller
"""
from numpy.linalg import inv
from scipy.linalg import sqrtm
from numpy.linalg import norm
import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import Wedge
import math
import sympy
from scipy.spatial import ConvexHull
from projlib import  *
def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
    
    
def find_gcd(x, y): 
#    l = [2, 4, 6, 8, 16] 
#      
#    num1 = l[0] 
#    num2 = l[1] 
#    gcd = find_gcd(num1, num2) 
#      
#    for i in range(2, len(l)): 
#        gcd = find_gcd(gcd, l[i]) 
#          
#    print(gcd)       


    while(y): 
        x, y = y, x % y 
      
    return x 

def perpendicular_vector(v):
    r""" Finds an arbitrary perpendicular vector to *v*."""
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        raise ValueError('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array([1, 0, 0])
    if v[1] == 0:
        return np.array([0, 1, 0])
    if v[2] == 0:
        return np.array([0, 0, 1])

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    v2=np.array([1, 1, -1.0 * (v[0] + v[1])/v[2]])
    return v2/np.sqrt(v2.dot(v2))


def vec2string(v, digits=2):
    # vstr = '[';
    # for vi in v[:-1]:
    #     if int(vi*10**(digits))==0:
    #         vi=0.
    #     exec("vstr+='%0."+str(int(digits))+"f' " + '% vi')
    #     vstr +=','
    #     print(vstr)
    # vi=v[-1]
    # if int(vi*10**(digits))==0:
    #     vi=0.
    # exec("vstr+='%0."+str(int(digits))+"f' " + '% vi')
    # vstr +=']'
    vstr='[{:0.{prec}f},{:0.{prec}f},{:0.{prec}f}]'.format(v[0],v[1],v[2],prec=digits)
    return vstr
def plane2string(v, digits=2):
    # vstr = '(';
    # for vi in v[:-1]:
    #     if int(vi*10**(digits))==0:
    #         vi=0.
    #     exec("vstr+='%0."+str(int(digits))+"f' " + '% vi')
    #     vstr +=','
    # vi=v[-1]
    # if int(vi*10**(digits))==0:
    #     vi=0.
    # exec("vstr+='%0."+str(int(digits))+"f' " + '% vi')
    # vstr +=')'
    vstr='({:0.{prec}f},{:0.{prec}f},{:0.{prec}f})'.format(v[0],v[1],v[2],prec=digits)
    return vstr
def dir2string(v, digits=2):
    # vstr = '[';
    # for vi in v[:-1]:
    #     if int(vi*10**(digits))==0:
    #         vi=0.
    #     exec("vstr+='%0."+str(int(digits))+"f' " + '% vi')
    #     vstr +=','
    # vi=v[-1]
    # if int(vi*10**(digits))==0:
    #     vi=0.
    # exec("vstr+='%0."+str(int(digits))+"f' " + '% vi')
    # vstr +=']'
    vstr='[{:0.{prec}f},{:0.{prec}f},{:0.{prec}f}]'.format(v[0],v[1],v[2],prec=digits)

    return vstr
        
        
def xyz2fractional(Txyz2uvw,V,frac=10,eps2=1e-2,decimals=5):
    eps=1.e-6
    uvw = Txyz2uvw.dot(V)
    uvw = miller2fractional(uvw,frac=frac,eps2=eps2,decimals=decimals)

#    idxs = np.where(abs(uvw)<eps)
#    uvw[idxs]=0.
#    uvwini=uvw;
#    idxs = np.where(abs(uvw)>eps)
#    uvw=uvw/min(abs(uvw[idxs]))
#    uvwini=uvw;
#    
#    if (abs(np.round(uvw*frac)/frac-uvw)<eps2).all():
#        #print(uvw)
#        uvw=np.float_(np.int_(uvw*frac))/frac
#        #print(uvw)
##    if (abs(np.round(uvw*eps)-uvw*eps)<1).all():
##        uvw=np.round(uvw*eps)/eps
#        uvwi=uvw
#        for k in range(1,frac+1):
#            if uvwi[0]%1==0 and uvwi[1]%1==0 and uvwi[2]%1==0:
#                uvw=uvwi
#                break
#            else:
#                uvwi=k*uvw
#        gcd = find_gcd(find_gcd(uvwi[0], uvwi[1]), uvwi[2])
#        uvw=uvwi/abs(gcd)
#        if (uvw>1.e10).any():
#            print(uvw)
#            print(uvwini)
#    else:
#        uvw=uvwini
#    idxs = np.where(abs(uvw)>eps)
#    if (np.sign(uvw[idxs])==-1).all():
#        uvw=-1*uvw
    return np.around(uvw,decimals=decimals)
def miller2fractional(uvw,frac=10,eps2=1e-2,decimals=5):
    eps=1.e-6
    idxs = np.where(abs(uvw)<eps)
    if len(idxs[0])>0:
        uvw[idxs]=0.
    uvwini=uvw;
    idxs = np.where(abs(uvw)>eps)
    if len(idxs[0])>0:
        uvw=uvw/min(abs(uvw[idxs]))
    uvwini=uvw;
    
    if (abs(np.round(uvw*frac)/frac-uvw)<eps2).all():
        #print(uvw)
        uvw=np.float_(np.int_(uvw*frac))/frac
        #print(uvw)
#    if (abs(np.round(uvw*eps)-uvw*eps)<1).all():
#        uvw=np.round(uvw*eps)/eps
        uvwi=uvw
        for k in range(1,frac+1):
            if uvwi[0]%1==0 and uvwi[1]%1==0 and uvwi[2]%1==0:
                uvw=uvwi
                break
            else:
                uvwi=k*uvw
        gcd = find_gcd(find_gcd(uvwi[0], uvwi[1]), uvwi[2])
        uvw=uvwi/abs(gcd)
    else:
        uvw=uvwini
    idxs = np.where(abs(uvw)>eps)
#    if (np.sign(uvw[idxs])==-1).all():
#        uvw=-1*uvw
    idxs = np.where(abs(uvw)<eps)
    #print(idxs)
    if len(idxs[0])>0:
        idxs = np.where(abs(uvw)>eps)
        if len(idxs)>0:
            print(idxs[0][0])
            uvw=uvw/uvw[idxs[0][0]]
        

    return np.around(uvw,decimals=decimals)


def xyz2fractional02(Txyz2uvw,V):
    eps=1.e-6
    uvw = Txyz2uvw.dot(V)
    idxs = np.where(abs(uvw)<eps)
    uvw[idxs]=0.
    uvwini=uvw;
    idxs = np.where(abs(uvw)>eps)
    uvw=uvw/min(abs(uvw[idxs]))
    
    return uvw
def normArrayColumns(arr):
    arr_norm = np.empty((3,3))
    for col in range(arr.shape[1]):
        arr_norm[:,col]=arr[:,col]/np.linalg.norm(arr[:,col])
    
    return arr_norm

    
def cubic_lattice_vec(a):
    V = a*np.eye(3)
    return V[:,0], V[:,1], V[:,2]

def monoclinic_lattice_vec(a,b,c,beta):
    V1 = np.array([a,0.,0])
    V2 = np.array([0,b,0])
    V3 = np.array([c*np.cos(beta),0,c*np.sin(beta)])
    
    return V1,V2,V3

def tetragonal_lattice_vec(a,b,c):
    V1 = np.array([a,0.,0])
    V2 = np.array([0,b,0])
    V3 = np.array([0,0,c])
    
    return V1,V2,V3
def uvtw2uvw(uvtw):
    if type(uvtw)==list:
        if len(np.array(uvtw).shape)==1:    
            uvtw = np.expand_dims(uvtw,axis=1)
        else:
            uvtw = np.array(uvtw).T
    else:
        if len(uvtw.shape)==1:    
            uvtw = np.expand_dims(uvtw,axis=1)
            
    uvw=np.array([uvtw[0,:]-uvtw[2,:],uvtw[1,:]-uvtw[2,:],uvtw[3,:]]) 
    return uvw
def uvw2uvtw(uvw):
    if type(uvw)==list:
        if len(np.array(uvw).shape)==1:    
            uvw = np.expand_dims(uvw,axis=1)
        else:
            uvw = np.array(uvw).T
    else:
        if len(uvw.shape)==1:    
            uvw = np.expand_dims(uvw,axis=1)
            
    uvtw=np.array([1/3*(2*uvw[0,:]-uvw[1,:]),1/3*(2*uvw[1,:]-uvw[0,:]),-1/3*(uvw[0,:]+uvw[1,:]),uvw[2,:]]) 
    return uvtw

def hkil2hkl(hkil):
    if type(hkil)==list:
        if len(np.array(hkil).shape)==1:    
            hkil = np.expand_dims(hkil,axis=1)
        else:
            hkil = np.array(hkil).T
    else:
        if len(hkil.shape)==1:    
            hkil = np.expand_dims(hkil,axis=1)
            
    hkl=hkil[[0,1,3],:] 
    return hkl
def hkl2hkil(hkl):
    if type(hkl)==list:
        if len(np.array(hkl).shape)==1:    
            hkl = np.expand_dims(hkl,axis=1)
        else:
            hkl = np.array(hkl).T
    else:
        if len(hkl.shape)==1:    
            hkl = np.expand_dims(hkl,axis=1)
            
    hkil=np.array([hkl[0,:],hkl[1,:],-1*(hkl[0,:]+hkl[1,:]),hkl[2,:]]) 
    return hkil


def gensystemsHexIni(eta1,K1,L, Lr,sm=None,eta2=None):
    from crystals import Crystal
    if sm is None:
        sm={}
        
        for title in 'System StrainNonSym StrainSym Rotation SchmidTensor'.split():
            sm[title]=[]
    sm['SystemFamily']={'hkl':eta1,'uvw':K1}
    Mg = Crystal.from_cif('/home/lheller/Documents/diffraction/cifs/Mg.cif')
    Mg_symops=[np.round(sym[0:3,0:3],decimals=4) for sym in Mg.symmetry_operations()]
    Mg_recsymops=[np.round(sym[0:3,0:3],decimals=4) for sym in Mg.reciprocal_symmetry_operations()]
    eta1s=[]
    K1s=[]
    for sym,recsym in zip(Mg_symops,Mg_recsymops):
        a1=L.dot(sym.dot(uvtw2uvw(eta1)))
        a1/=np.sqrt(np.sum(a1**2,axis=0))
        n1=Lr.dot(recsym.dot(hkil2hkl(K1)))
        n1/=np.sqrt(np.sum(n1**2,axis=0))
    
        isin=False
        for eta1si,K1si in zip(eta1s,K1s):
            if (hkl2hkil(recsym.dot(hkil2hkl(K1)[:,0]))[:,0]==K1si).all() or (hkl2hkil(recsym.dot(hkil2hkl(K1)[:,0]))[:,0]==-1*K1si).all():
                if (uvw2uvtw(sym.dot(uvtw2uvw(eta1)[:,0]))[:,0]==eta1si).all() or (uvw2uvtw(sym.dot(uvtw2uvw(eta1)[:,0]))[:,0]==-1*eta1si).all():
                    isin=True 
                    break
        
        if not isin:
            eta1s.append(uvw2uvtw(sym.dot(uvtw2uvw(eta1)[:,0]))[:,0])
            K1s.append(hkl2hkil(recsym.dot(hkil2hkl(K1)[:,0]))[:,0])
            #sm['SlipSystemFamily']={'hkl':eta1,'uvw':K1}
            sm['SchmidTensor'].append(np.outer(a1[:,0],n1[:,0]))
            sm['System'].append({'n':n1,'b':a1,'hkl':K1s[-1],'uvw':eta1s[-1],'hklstr':str(K1s[-1]).replace('[','(').replace(']',')').replace(',',''),'uvwstr':str(eta1s[-1]).replace(',','')})
            sm['StrainNonSym'].append(np.outer(a1,n1))
            sm['StrainSym'].append(0.5*(sm['StrainNonSym'][-1]+sm['StrainNonSym'][-1].T))
            sm['Rotation'].append(0.5*(sm['StrainNonSym'][-1]-sm['StrainNonSym'][-1].T))
    return sm

def gensystemsHex(eta1,K1,L, Lr,sm=None,eta2=None, K2=None):
    from crystals import Crystal
    if sm is None:
        sm={}
        
        for title in 'TwinningSystem System StrainNonSym StrainSym Rotation SchmidTensor SystemType DefGrad'.split():
            sm[title]=[]
        sm['SystemType']='slip'
        if eta2 is not None:
            sm['SystemType']='twinning'
            for title in 'MaxTensileStrain MaxCompressiveStrain rotangle skewmatrix uvw2xyz Rij shear_angle s StrainTensor Tension Compression TensileDir CompressionDir TwinDislocation'.split():
                sm[title]=[]
    G = np.matmul(L.T,L)
    Gr = inv(G)
    
    sm['SystemFamily']={'hkl':eta1,'uvw':K1}
    Mg = Crystal.from_cif('/home/lheller/Documents/diffraction/cifs/Mg.cif')
    Mg_symops=[np.round(sym[0:3,0:3],decimals=4) for sym in Mg.symmetry_operations()]
    Mg_recsymops=[np.round(sym[0:3,0:3],decimals=4) for sym in Mg.reciprocal_symmetry_operations()]
    eta1s=[]
    K1s=[]
    for sym,recsym in zip(Mg_symops,Mg_recsymops):
        a1=L.dot(sym.dot(uvtw2uvw(eta1)))
        a1/=np.sqrt(np.sum(a1**2,axis=0))
        n1=Lr.dot(recsym.dot(hkil2hkl(K1)))
        n1/=np.sqrt(np.sum(n1**2,axis=0))
        
        isin=False
        for eta1si,K1si in zip(eta1s,K1s):
            if (hkl2hkil(recsym.dot(hkil2hkl(K1)[:,0]))[:,0]==K1si).all() or (hkl2hkil(recsym.dot(hkil2hkl(K1)[:,0]))[:,0]==-1*K1si).all():
                if (uvw2uvtw(sym.dot(uvtw2uvw(eta1)[:,0]))[:,0]==eta1si).all() or (uvw2uvtw(sym.dot(uvtw2uvw(eta1)[:,0]))[:,0]==-1*eta1si).all():
                    isin=True 
                    break
        
        if not isin:
            eta1s.append(uvw2uvtw(sym.dot(uvtw2uvw(eta1)[:,0]))[:,0])
            K1s.append(hkl2hkil(recsym.dot(hkil2hkl(K1)[:,0]))[:,0])
            #sm['SlipSystemFamily']={'hkl':eta1,'uvw':K1}
            sm['SchmidTensor'].append(np.outer(a1[:,0],n1[:,0]))
            sm['System'].append({'n':n1,'b':a1,'hkl':K1s[-1],'uvw':eta1s[-1],'hklstr':str(K1s[-1]).replace('[','(').replace(']',')').replace(',',''),'uvwstr':str(eta1s[-1]).replace(',','')})
            if eta2 is not None:
                sm['System'].append({'n':n1,'b':a1,'hkl':K1s[-1],'uvw':eta1s[-1],
                                     'hklstr':str(K1s[-1]).replace('[','(').replace(']',')').replace(',',''),'uvwstr':str(eta1s[-1]).replace(',','')})
                eta2s=uvw2uvtw(sym.dot(uvtw2uvw(eta2)))[:,0]
                eta2str=str(eta2s).replace(',','')
                a2=L.dot(sym.dot(uvtw2uvw(eta2)))
                a2/=np.sqrt(np.sum(a2**2,axis=0))
                if K2 is not None:
                    K2s=hkl2hkil(recsym.dot(hkil2hkl(K2)[:,0]))[:,0]
                    n2=Lr.dot(recsym.dot(hkil2hkl(K2s)))
                    n2/=np.sqrt(np.sum(n2**2,axis=0))
                    K2str=str(K2s).replace('[','(').replace(']',')').replace(',','')
                else:
                    n2=None
                    K2s=None
                    K2str=None
                    
                
                sm['TwinningSystem'].append({'n1':n1,'a1':a1,'n2':n2,'a2':a2,'K1':K1s[-1],'eta1':eta1s[-1],'K2':K2s,'eta2':eta2s,
                                     'K1str':str(K1s[-1]).replace('[','(').replace(']',')').replace(',',''),'eta1str':str(eta1s[-1]).replace(',',''),
                                     'K2str':K2str,'eta2str':eta2str})

                #print(a1)
                sm['uvw2xyz'].append(np.matmul(2*np.outer(a1[:,0],a1[:,0])-np.eye(3),L))
                sm['Rij'].append(2*np.outer(a1[:,0],a1[:,0])-np.eye(3))
                sm['shear_angle'].append(2*abs(np.pi/2-np.arccos(a2[:,0].dot(a1[:,0]))))
                sm['s'].append(np.tan(sm['shear_angle'][-1]/2)*2)
            else:
                sm['System'].append({'n':n1,'b':a1,'hkl':K1s[-1],'uvw':eta1s[-1],'hklstr':str(K1s[-1]).replace('[','(').replace(']',')').replace(',',''),'uvwstr':str(eta1s[-1]).replace(',','')})
                sm['TwinningSystem'].append({})
            if eta2 is not None:
                
                sm['StrainNonSym'].append(sm['s'][-1]*np.outer(a1[:,0],n1[:,0]))
            else:
                sm['StrainNonSym'].append(np.outer(a1[:,0],n1[:,0]))
            sm['DefGrad'].append(np.eye(3)+sm['StrainNonSym'][-1])
            #sm['StrainSym'].append(0.5*(sm['StrainNonSym'][-1]+sm['StrainNonSym'][-1].T))
            sm['StrainSym'].append(0.5*(sm['DefGrad'][-1].T.dot(sm['DefGrad'][-1])-np.eye(3)))
            #sm['Rotation'].append(0.5*(sm['StrainNonSym'][-1]-sm['StrainNonSym'][-1].T))
            
            if eta2 is not None:
                W2=0.5*(sm['DefGrad'][-1]-sm['DefGrad'][-1].T)
                sm['skewmatrix'].append(W2)
                #Normalized skew matrix is np.cross(np.eye(3),rvn) formed with normilized elements of rv
                #https://www.brainm.com/software/pubs/math/Rotation_matrix.pdf
                #Rotation matrix=I*cos(omega)+sin(omega)*np.cross(rvn,np.eye(3)) + (1-cos(omega))*np.outer(rvn,rvn)
    
                rv=np.array([W2[2,1],-W2[2,0],W2[1,0]])
                #rotation angle
                omega=norm(rv)
                rvn=rv/norm(rv)
                sm['rotangle'].append(omega)
                sm['Rotation'].append(np.eye(3)*np.cos(omega)+np.sin(omega)*np.cross(rvn,np.eye(3)) + (1-np.cos(omega))*np.outer(rvn,rvn))

            if eta2 is not None:
                D,V = np.linalg.eig(sm['StrainSym'][-1])
                Idxs = np.argsort(D)
                Lambda=D[Idxs]
                V = V[:,Idxs]
                D = D[Idxs]
                sm['TensileDir'].append(V[:,2])
                sm['CompressionDir'].append(V[:,0])
                sm['MaxTensileStrain'].append(D[2])
                sm['MaxCompressiveStrain'].append(D[0])
                #print(hkil2hkl(K1s[-1]))
                sm['TwinDislocation'].append(get_twinning_dislocation(hkil2hkl(K1s[-1])[:,0],uvtw2uvw(eta1s[-1])[:,0],uvtw2uvw(eta2s)[:,0],L,G=G,Gr=Gr))

            
    return sm
def rotation_from_axis_angle(ax,an,deg=False):
    if deg:
        an=an*np.pi/180
    ax/=norm(ax)
    return np.eye(3)*np.cos(an)+np.sin(an)*np.cross(ax,np.eye(3)) + (1-np.cos(an))*np.outer(ax,ax)
def genallHexSys():
    #https://www.sciencedirect.com/science/article/pii/S1359645421001774#bib0025
    SSys={}
    etas1=[[1,-2,1,0], [1,-2,1,0],[1,1,-2,-3],[1,1,-2,-3],[1,0,-1,-1],[1,0,-1,-2],[1,1,-2,-6]]
    Ks1=[[0,0,0,1],[1,0,-1,0],[1,1,-2,2],[1,0,-1,1],[1,0,-1,2],[1,0,-1,1],[1,1,-2,1]]
    names=['basal slip','prismatic slip', 'pyramidal slip', 'pyramidal slip', 'twinning' , 'twinning' , 'twinning' ]

    for name, eta1, K1 in zip(names,etas1,Ks1):
        namei=name+' ' +str(K1).replace('[','{').replace(']','}').replace(',','')+str(eta1).replace('[','<').replace(']','>').replace(',','')
        SSys[namei]=gensystemsHex(eta1,K1)
    
def lattice_vec(lattice_param):
    if lattice_param['type'].lower()=='cubic':
        a= lattice_param['a'];
        V = a*np.eye(3)
    elif lattice_param['type'].lower()=='tetragonal':
        a= lattice_param['a'];
        b= lattice_param['b'];
        c= lattice_param['c'];
        V = np.zeros((3,3))
        V[:,0] = np.array([a,0.,0])
        V[:,1] = np.array([0,b,0])
        V[:,2] = np.array([0,0,c])    
    elif lattice_param['type'].lower()=='hexagonal':
       a= lattice_param['a'];
       c= lattice_param['c'];
       V = np.zeros((3,3))
       V[:,0] = np.array([a,0.,0])
       V[:,1] = np.array([-a/2,np.sqrt(3)/2*a,0])
       #V[:,2] = np.array([-a/2,-np.sqrt(3)/2*a,0])
       V[:,2] = np.array([0,0,c])
    elif lattice_param['type'].lower()=='monoclinic':
        a= lattice_param['a'];
        b= lattice_param['b'];
        c= lattice_param['c'];
        beta= lattice_param['beta'];
        V = np.zeros((3,3))
        V[:,0] = np.array([a,0.,0])
        V[:,1] = np.array([0,b,0])
        V[:,2] = np.array([c*np.cos(beta),0,c*np.sin(beta)])
    elif lattice_param['type'].lower()=='triclinic':
        a= lattice_param['a'];
        b= lattice_param['b'];
        c= lattice_param['c'];
        alpha= lattice_param['alpha'];
        beta= lattice_param['beta'];
        gamma= lattice_param['gamma'];
        V = np.zeros((3,3))
        V[:,0] = np.array([a,0.,0])
        V[:,1] = np.array([b*np.cos(gamma),b*np.sin(gamma),0])
        cx=c*np.cos(beta)
        cy=c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)
        cz=np.sqrt(c**2-cx**2-cy**2)
        V[:,2] = np.array([cx,cy,cz])
    elif lattice_param['type'].lower()=='trigonal':
        a= lattice_param['a'];
        c= lattice_param['c'];
        V = np.zeros((3,3))
        V[:,0] = np.array([1./2.*a,-np.sqrt(3)/2.*a,0])
        V[:,1] = np.array([1./2.*a,np.sqrt(3)/2.*a,0])
        V[:,2] = np.array([0,0,c])
        
        

        
    return V[:,0], V[:,1], V[:,2]

def reciprocal_basis(a1,a2,a3):
    
    b1 = np.cross(a2,a3)/np.dot(a1,np.cross(a2,a3))
    b2 = np.cross(a3,a1)/np.dot(a2,np.cross(a3,a1))
    b3 = np.cross(a1,a2)/np.dot(a3,np.cross(a1,a2))
    return b1,b2,b3

def np_euler_matrix(ai, aj, ak): 
#    ai=20.*np.pi/180.
#    aj=40.*np.pi/180.
#    ak=80.*np.pi/180.
#    np_euler_matrix(ai, aj, ak)
#    np.matmul(passive_rotation(ak,'z'),np.matmul(passive_rotation(aj,'x'),passive_rotation(ai,'z')))
    
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
#def np_ol_q_U2(q):
#    #Poulsen
#    g=np.empty((3,3))
#    
#    for i in range(0,2):
#        g[i,i]=2*(q[0]**2+q[i+1]**2)-1
#    
#    g[1,0] = 2*(q[1]*q[2]+q[0]*q[3])
#    g[0,1] = 2*(q[1]*q[2]-q[0]*q[3])
#
#    g[2,0] = 2*(q[1]*q[3]-q[0]*q[2])
#    g[0,2] = 2*(q[1]*q[3]+q[0]*q[2])
#    
#    g[2,1] = 2*(q[2]*q[3]+q[0]*q[1])
#    g[1,2] = 2*(q[2]*q[3]-q[0]*q[1])
#
#    return g
#
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



        
def permut_tensor3():
    epsilon=[];
    for i in range(0,3):
        pj=[]
        for j in range(0,3):
            pk=[]
            for k in range(0,3):
                if i==j or j==k or k==i:
                    pk.append(0)
                elif (i,j,k)==(0,1,2) or (i,j,k)==(1,2,0) or (i,j,k)==(2,0,1):
                    pk.append(1)
                else:
                    pk.append(-1)
            pj.append(pk)
        epsilon.append(pj)
    return epsilon

def np_permut_tensor3():
    epsilon=np.empty((3,3,3));
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                if i==j or j==k or k==i:
                    val=0
                elif (i,j,k)==(0,1,2) or (i,j,k)==(1,2,0) or (i,j,k)==(2,0,1):
                    val=1
                else:
                    val=-1
                epsilon[i,j,k]=val
    return epsilon

def kronecker():
    delta=[];
    for i in range(0,3):
        pj=[]
        for j in range(0,3):
            if i==j:
                pj.append(1.)
            else:
                pj.append(0)
        delta.append(pj)
    return delta

def np_kronecker():
    delta=np.empty((3,3));
    for i in range(0,3):
        for j in range(0,3):
            if i==j:
                val=1
            else:
                val=0
            delta[i,j]=val
    return delta



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
    
def stereoprojection_directions(dirs):
    #dirs = [x1,x2,...,xn;y1,y2,...,yn;z1,z2,...,zn];
    #example: dirs = [0,1,2,3;1,2,3,0;0,3,2,1]
    
    #normalize and project
    if len(dirs.shape)==1:
        dirs = np.expand_dims(dirs,axis=1)

    dirs = dirs.astype(float)
    #normalizing dirs
    dirs /= np.sqrt((dirs ** 2).sum(0))
    #check
    
    proj_dirs = np.vstack((1./(np.sign(dirs[2,:])*dirs[2,:]+1)*dirs[0,:], 
               1./(np.sign(dirs[2,:])*dirs[2,:]+1)*dirs[1,:],
                np.zeros(dirs[2,:].shape)))
    #check
#    alpha = np.arccos(np.sign(dirs[2,:])*dirs[2,:])
#    eps=1e-6;
#    idxs = np.where(alpha<eps)
#    alpha[idxs]=1.
#    beta = np.arccos(dirs[0,:]/np.sin(alpha))
#    rx=np.tan(alpha/2)*np.cos(beta)
#    ry=np.tan(alpha/2)*np.sin(beta)
#    rx[idxs]=0.
#    ry[idxs]=0.
#    abs(proj_dirs[0,:]-rx)<eps
#    abs(proj_dirs[1,:]-ry)<eps
    return proj_dirs

# def stereoprojection_intotriangle(dirs):
#     eps=1.0e-2
#     normals = np.array([-1,0,1]);
#     arclength = 40.#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
#     proj_normals, points = stereoprojection_planes(normals,arclength=arclength,iniangle=90)
#     proj_tans = np.arctan(proj_normals[1,:]/proj_normals[0,:])

#     if len(dirs.shape)==1:
#         dirs = np.expand_dims(dirs,axis=1)

#     proj_dirs = np.zeros(dirs.shape)
#     inc=-1
#     for co,diri in enumerate(dirs.T):
#         #print('Direction {} from {}'.format(co+1,dirs.shape[1]))
# #        print('===================================================')
# #        print(diri)
# #        print('===================================================')
#         inc+=1
#         el=equivalent_elements(diri,'cubic')
#         #print(el)
#         for eli in el:
#             proj_eli = stereoprojection_directions(eli)
# #            print(eli)
# #            print(np.arctan(proj_eli[1,0]/proj_eli[0,0])-np.arccos(1./np.sqrt(3.)))
# #            print(np.arctan(proj_eli[1,0]/proj_eli[0,0]))#-np.pi/4)
# #            print((np.arccos(abs(eli[2])/np.sqrt(eli.dot(eli)))-np.arccos(1./np.sqrt(3.))))
# #            if (eli>=-eps).all() and (np.arccos(abs(eli[2])/np.sqrt(eli.dot(eli)))-np.arccos(1./np.sqrt(3.)))<eps:
# #            if (proj_eli[:,0]>=-eps).all() and (np.arccos(abs(eli[2])/np.sqrt(eli.dot(eli)))-np.arccos(1./np.sqrt(3.)))<eps:
#             if ((proj_eli[:,0])>=-eps).all():                #proj_eli = stereoprojection_directions(eli)
#                 atan=np.arctan2(proj_eli[1,0],proj_eli[0,0])
#                 if (atan-np.pi/4)<eps:
#                     idx=np.where(abs(proj_tans-atan)==min(abs(proj_tans-atan)))[0][0]
# #                    print(proj_eli[:,0].dot(proj_eli[:,0])) 
# #                    print(proj_normals[:,idx].dot(proj_normals[:,idx])) 
# #                    print((proj_eli[:,0].dot(proj_eli[:,0])-proj_normals[:,idx].dot(proj_normals[:,idx])))
# #                    print((proj_eli[:,0].dot(proj_eli[:,0])-proj_normals[:,idx].dot(proj_normals[:,idx]))<eps)
#                     if (proj_eli[:,0].dot(proj_eli[:,0])-proj_normals[:,idx].dot(proj_normals[:,idx]))<eps:
#                         proj_dirs[:,inc]=proj_eli[:,0]
# #                        print('OK')
# #                        print(proj_dirs[:,inc])
#                         break
#     return proj_dirs

#def coodinate_from_equalarea_proj(projdirs):
    #dirs = [x1,x2,...,xn;y1,y2,...,yn];
    #example: dirs = [0,1,2,3;1,2,3,0]
#    if len(projdirs.shape)==1:
#        projdirs = np.expand_dims(projdirs,axis=1)
#    z=-2+np.sqrt(8-projdirs[0,:]**2-projdirs[1,:]**2)

#    z=-2+np.sqrt()
    
    
    
def equalarea_directions(dirs):
    #dirs = [x1,x2,...,xn;y1,y2,...,yn;z1,z2,...,zn];
    #example: dirs = np.array([[0,1,2,3],[1,2,3,0],[0,3,2,1]])
    
    #normalize and project
       
    if len(dirs.shape)==1:
        dirs = np.expand_dims(dirs,axis=1)

    dirs = dirs.astype(float)
    #normalizing dirs
    dirs /= np.sqrt((dirs ** 2).sum(0))
    dirsxy = dirs[0:2,:];
    #print(dirsxy)
    eps=1e-6
    normdirsxy = np.sqrt((dirsxy ** 2).sum(0))
    idxs=np.where(normdirsxy<eps)
    normdirsxy[idxs]=1.
    dirsxy /= normdirsxy
    
    alpha=np.arccos(np.sign(dirs[2,:])*dirs[2,:]);
    proj_dirs = np.vstack((np.sin(alpha/2)*2.*dirsxy[0,:], 
               np.sin(alpha/2)*2.*dirsxy[1,:],
                np.zeros(dirs[2,:].shape)))
    proj_dirs[:,idxs]=0.
    #check
#    if False:
#        phi2=90*np.pi/180.
#        phi1=0*np.pi/180.
#        RotX = active_rotation(phi1, 'x') 
#        RotY = active_rotation(phi2, 'y')
#        Dc=[0.,0.,1.];
#        Ds = np.matmul(RotY,RotX).dot(Dc)
#        proj_Ds = equalarea_directions(Ds)  
#        #np.pi*proj_Ds[0][0]**2-(-np.cos(phi2)+np.cos(0))*2*np.pi
#        phi22=90*np.pi/180.
#        phi21=85*np.pi/180.
#        phi1=0*np.pi/180.
#        RotX = active_rotation(phi1, 'x') 
#        RotY = active_rotation(phi21, 'y')
#        RotY2 = active_rotation(phi22, 'y')
#        Dc=[0.,0.,1.];
#        Ds = np.matmul(RotY,RotX).dot(Dc)
#        proj_Ds = equalarea_directions(Ds)  
#        Ds = np.matmul(RotY2,RotX).dot(Dc)
#        proj_Ds2 = equalarea_directions(Ds)  
#        #np.pi*(proj_Ds2[0][0]**2-proj_Ds[0][0]**2)-(-np.cos(phi22)+np.cos(phi21))*2*np.pi
#        phi2=45*np.pi/180.
#        dphi2=5*np.pi/180.
#        dphi1=5.*np.pi/180.
#        RotY = active_rotation(phi2-dphi2, 'y')
#        RotY2 = active_rotation(phi2+dphi2, 'y')
#        Dc=[0.,0.,1.];
#        Ds = RotY.dot(Dc)
#        proj_Ds = equalarea_directions(Ds)  
#        Ds = RotY2.dot(Dc)
#        proj_Ds2 = equalarea_directions(Ds)  
#        #1./2.*dphi1*(proj_Ds2[0][0]**2-proj_Ds[0][0]**2)-(+np.cos(phi2-dphi2)-np.cos(phi2+dphi2))*dphi1
        
        
    return proj_dirs
# def equalarea_planes(normals,arclength=360.,iniangle=0.,hemisphere="both"):
#     #%normals = [x1,x2,...,xn;y1,y2,...,yn;z1,z2,...,zn];
#     #%varargin{1} arclength in deg
#     #normals = np.transpose(np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]))
#     #
#     if len(normals.shape)==1:
#         normals = np.expand_dims(normals,axis=1)

#     normals = normals.astype(float)
#     normals /= np.sqrt((normals ** 2).sum(0))

#     proj_normals = equalarea_directions(normals)

#     idxs = np.where(abs(normals[0,:])+abs(normals[1,:])==0)[0]
    
#     inplanedirs = np.vstack((-normals[1,:],normals[0,:],np.zeros(normals[0,:].shape)));
#     inplanedirs[:,idxs] = np.vstack((np.zeros(normals[0,idxs].shape), -normals[2,idxs],normals[1,idxs]));
    
#     inplanedirs /= np.sqrt((inplanedirs ** 2).sum(0))

#     thirdaxis=np.cross(normals,inplanedirs,axisa=0,axisb=0,axisc=0)
# #    thirdaxis = np.vstack((normals[1,:]*inplanedirs[2,:]-normals[2,:]*inplanedirs[1,:],
# #                           -1*(normals[0,:]*inplanedirs[2,:]-normals[2,:]*inplanedirs[0,:]),
# #                           normals[0,:]*inplanedirs[1,:]-normals[1,:]*inplanedirs[0,:]));
#     t=np.linspace(iniangle,iniangle+arclength,180*2+1)*np.pi/180;
#     basicarc = np.vstack((np.cos(t),np.sin(t),np.zeros(t.shape)));
    
#     proj_planes=[];
#     Zdir=[]
#     for i in range(0,normals.shape[1]):
#         Rot2Global = np.transpose(np.vstack((inplanedirs[:,i],thirdaxis[:,i],normals[:,i])));
#         Ccp = np.matmul(Rot2Global,basicarc);
#         Zdir=Ccp[2]
#         if hemisphere == "both":            
#             Ds = equalarea_directions(Ccp)
#         else:
#             if hemisphere == "upper":
#                 idxs = np.where(Ccp[2,:]>=0)[0]
#                 Ds = equalarea_directions(Ccp[:,idxs])
#             elif hemisphere == "lower":
#                 idxs = np.where(Ccp[2,:]<=0)[0]
#                 Ds = equalarea_directions(Ccp[:,idxs])
#         proj_planes.append(Ds)
#     if len(proj_planes)==1:
#         return proj_planes[0]
#     else:          
#         return proj_planes


# def stereoprojection_planes(normals,arclength=360.,iniangle=0.,hemisphere="both"):
#     #%normals = [x1,x2,...,xn;y1,y2,...,yn;z1,z2,...,zn];
#     #%varargin{1} arclength in deg
#     #normals = np.transpose(np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]))
#     #
#     if len(normals.shape)==1:
#         normals = np.expand_dims(normals,axis=1)

#     normals = normals.astype(float)
#     normals /= np.sqrt((normals ** 2).sum(0))

#     proj_normals = stereoprojection_directions(normals)

#     idxs = np.where(abs(normals[0,:])+abs(normals[1,:])==0)[0]
    
#     inplanedirs = np.vstack((-normals[1,:],normals[0,:],np.zeros(normals[0,:].shape)));
#     inplanedirs[:,idxs] = np.vstack((np.zeros(normals[0,idxs].shape), -normals[2,idxs],normals[1,idxs]));
    
#     inplanedirs /= np.sqrt((inplanedirs ** 2).sum(0))

#     thirdaxis=np.cross(normals,inplanedirs,axisa=0,axisb=0,axisc=0)
# #    thirdaxis = np.vstack((normals[1,:]*inplanedirs[2,:]-normals[2,:]*inplanedirs[1,:],
# #                           -1*(normals[0,:]*inplanedirs[2,:]-normals[2,:]*inplanedirs[0,:]),
# #                           normals[0,:]*inplanedirs[1,:]-normals[1,:]*inplanedirs[0,:]));
#     t=np.linspace(iniangle,iniangle+arclength,180*2+1)*np.pi/180;
#     basicarc = np.vstack((np.cos(t),np.sin(t),np.zeros(t.shape)));
    
#     proj_planes=[];
#     Zdir=[]
#     points=[]
#     for i in range(0,normals.shape[1]):
#         Rot2Global = np.transpose(np.vstack((inplanedirs[:,i],thirdaxis[:,i],normals[:,i])));
#         Ccp = np.matmul(Rot2Global,basicarc);
#         points.append(Ccp)
#         Zdir=Ccp[2]
#         if hemisphere == "both":            
#             Ds = stereoprojection_directions(Ccp)
#         else:
#             if hemisphere == "upper":
#                 idxs = np.where(Ccp[2,:]>=0)[0]
#                 Ds = stereoprojection_directions(Ccp[:,idxs])
#             elif hemisphere == "lower":
#                 idxs = np.where(Ccp[2,:]<=0)[0]
#                 Ds = stereoprojection_directions(Ccp[:,idxs])
#         proj_planes.append(Ds)
#     if len(proj_planes)==1:
#         return proj_planes[0],points[0]
#     else:          
#         return proj_planes, points

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
        
def symmetry_elements(lattice):
    U=[]
    #identity
    U.append(np.eye(3))
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

    if lattice.lower()=='tetragonal':
        #1xpi/2 Rotations about 001=>3 operations

        U.append(np.array([[0,-1,0],[1,0,0],[0,0,1]]).T)
        U.append(np.array([[-1,0,0],[0,-1,0],[0,0,1]]).T)
        U.append(np.array([[0,1,0],[-1,0,0],[0,0,1]]).T)

        #1xpi Rotations about 100,010=>2 operations

        U.append(np.array([[1,0,0],[0,-1,0],[0,0,-1]]).T)
        U.append(np.array([[-1,0,0],[0,1,0],[0,0,-1]]).T)
        


        
        #1xpi Rotation about [110][-110]=>2 operations
        U.append(np.array([[0,1,0],[1,0,0],[0,0,-1]]).T)
        U.append(np.array([[0,-1,0],[-1,0,0],[0,0,-1]]).T)
        
       
    if lattice.lower()=='monoclinic':        
        U.append(np.array([[-1,0,0],[0,1,0],[0,0,-1]]).T)

        
#        for i in range(0,len(U)):
#            for j in range(0,len(U)):
#                if (U[i]==U[j]).all() and i<>j:
#                    print('spatne')
#        
#for u in U:
#    print(u)
#    print('')
#    return U
#     
    return U
           
def equivalent_elements(element,lattice):
    eq_elements=[]
    R=symmetry_elements(lattice)
    eq_elements.append(R[0].dot(element))
    for u in R[1:]:
        el=u.dot(element)
        isin=False
        for els in eq_elements:
            if (els==el).all():
                isin=True
                break
        if not isin:
            eq_elements.append(el)
        #eq_elements.append(el)
         
    return eq_elements
    
  
    

# def wulffnet(ax=None,basedirs=False,baseplanes=False,basedirscol='k',baseplanescol='tab:gray', description=True,R2Proj=np.eye(3),eps=1e-5,eps2=-1e-5,rotategrid=True):
#     if ax==None:
#         fig, ax = plt.subplots()

#         fig.patch.set_alpha(0)
#     else:
#         fig=[]
#     if basedirs:
#         basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]);
#         #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
#         basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]);
#         #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
#         basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],\
#                                         [-1,0,0],[0,-1,0],\
#                                         [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],\
#                                         [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],\
#                                         [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],\
#                                         [-1,1,1],[1,-1,1],[1,1,-1],\
#                                         [1,1,1],[-1,-1,1],[1,-1,-1],[-1,1,-1],[-1,-1,-1]]);
#         basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],\
#                                         [-1,0,0],[0,-1,0],\
#                                         [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],\
#                                         [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],\
#                                         [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],\
#                                         [-1,1,1],[1,-1,1],[1,1,-1],\
#                                         [1,1,1],[-1,-1,1],[1,-1,-1],[-1,1,-1],[-1,-1,-1]]);

#         basicplanes = np.array([[1,0,0],[0,1,0],[0,0,1],\
#                                         [-1,0,0],[0,-1,0],\
#                                         [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],\
#                                         [0,1,1],[0,-1,1],\
#                                         [1,0,1],[-1,0,1]]);

#         # basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],\
#         #                                 [-1,0,0],[0,-1,0],\
#         #                                 [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],\
#         #                                 [0,1,1],[0,-1,1],\
#         #                                 [1,0,1],[-1,0,1],\
#         #                                 [-1,1,1],[1,-1,1],\
#         #                                 [1,1,1],[-1,-1,1]]);
#         # basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],\
#         #                                 [-1,0,0],[0,-1,0],\
#         #                                 [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],\
#         #                                 [0,1,1],[0,-1,1],\
#         #                                 [1,0,1],[-1,0,1],\
#         #                                 [-1,1,1],[1,-1,1],\
#         #                                 [1,1,1],[-1,-1,1]]);

#     #longitude lines
#     #fig, ax = plt.subplots()
#     ax.tick_params(
#         axis='both',
#         which='both',
#         bottom=False,
#         top=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False)
#     ax.plot(0, 0, 'k+')
#     circ = plt.Circle((0, 0), 1.0, facecolor=(210./255.,235./255.,255./255.), edgecolor='black')
#     ax.add_patch(circ)

#     ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
#   # equal aspect ratio
#     ax.axis('off')  # remove the box
#     #plt.show()
    
#     t=np.linspace(0,180,180*2+1)*np.pi/180;
#     xc = np.sin(t);
#     yc = np.cos(t);
#     AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
#     AltitudeAngle = np.linspace(0,360,73)*np.pi/180;
#     for an in AltitudeAngle:
#         RotY = passive_rotation(an, 'y')
#         if rotategrid:
#             Ccp = R2Proj.dot(np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape)))))
#         else:
#             Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))
#         Upper=np.where(Ccp[2,:]>eps)[0]
#         proj_dirs = stereoprojection_directions(Ccp)
#         idxs=np.where((np.diff(Upper)>1)==True)[0]
#         if idxs.shape[0]==0:
#             ax.plot(proj_dirs[0,Upper],proj_dirs[1,Upper],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
#         else:
#             idxini=0
#             for idx in idxs:
#                 ax.plot(proj_dirs[0,Upper[idxini:idx+1]],proj_dirs[1,Upper[idxini:idx+1]],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
#                 idxini=idx+1
#             ax.plot(proj_dirs[0,Upper[idxini:]],proj_dirs[1,Upper[idxini:]],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

#         #ax.plot(proj_dirs[0,Upper],proj_dirs[1,Upper],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

#     #Latitude Lines
#     LatitudeAngle = np.linspace(-90,90,37)*np.pi/180; 
#     LatitudeAngle = np.linspace(-180,180,73)*np.pi/180; 
#     #LatitudeAngle = np.linspace(125,135,2)*np.pi/180; 
    
#     #t=np.linspace(0,180,360*2+1)*np.pi/180;
#     zc = np.sin(t);
#     xc = np.cos(t);
#     for an in LatitudeAngle:#[0.]:#LatitudeAngle:
#         Rmeridian = np.cos(an);
#         px = Rmeridian*xc;
#         py = np.sin(an)*np.ones(t.shape);
#         pz = Rmeridian*zc;
#         if rotategrid:
#             Ccp = R2Proj.dot(np.vstack((px,py,pz)))
#         else:
#             Ccp = np.vstack((px,py,pz))
#         Upper=np.where(Ccp[2,:]>eps)[0]
#         proj_dirs = stereoprojection_directions(Ccp)
#         #print(Upper)
#         idxs=np.where((np.diff(Upper)>1)==True)[0]
#         if idxs.shape[0]==0:
#             ax.plot(proj_dirs[0,Upper],proj_dirs[1,Upper],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
#         else:
#             idxini=0
#             for idx in idxs:
#                 ax.plot(proj_dirs[0,Upper[idxini:idx+1]],proj_dirs[1,Upper[idxini:idx+1]],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
#                 idxini=idx+1
#             ax.plot(proj_dirs[0,Upper[idxini:]],proj_dirs[1,Upper[idxini:]],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
        
        
    
#     if basedirs:
#         an=45.*np.pi/180.;
#         an=0.*np.pi/180.;
#         Rotz = active_rotation(an, 'z')
#         an=np.arccos(1/np.sqrt(3));
#         an=0.*np.pi/180.;
#         Rotx = active_rotation(an, 'x')
#         dirs = R2Proj.dot(np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections)))
#         Upper=np.where(dirs[2,:]>eps2)[0]
#         proj_dirs = stereoprojection_directions(dirs)
#         ax.plot(proj_dirs[0,Upper],proj_dirs[1,Upper],color=basedirscol,marker='o',linestyle='',zorder=10010)
#         if description:
#             for diri,proj_diri in zip(basicdirectionstext.T[:,Upper].T,np.transpose(proj_dirs[:,Upper])):
#                 ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri),zorder=10010)

#     if baseplanes:
#         an=45.*np.pi/180.;
#         an=0.*np.pi/180.;
#         Rotz = active_rotation(an, 'z')
#         an=np.arccos(1/np.sqrt(3));
#         an=0.*np.pi/180.;
#         Rotx = active_rotation(an, 'x')
#         planes = R2Proj.dot(np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicplanes)))
#         proj_planes, points=stereoprojection_planes(planes)
#         for proj_plane,point in zip(proj_planes,points):
#             Upper=np.where(point[2,:]>eps)[0]
#             ax.plot(proj_plane[0,Upper],proj_plane[1,Upper],color=baseplanescol,linestyle='-',zorder=10000)

#     return fig,ax

# def schmidtnet(ax=None,basedirs=False,facecolor=(210./255.,235./255.,255./255.)):
#     if ax==None:
#         fig, ax = plt.subplots()

#         fig.patch.set_alpha(0)
#     else:
#         fig=[]
#     if basedirs:
#         basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]);
#         #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
#         basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]);
#         #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];



#     #longitude lines
#     #fig, ax = plt.subplots()
#     ax.tick_params(
#         axis='both',
#         which='both',
#         bottom=False,
#         top=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False)
#     ax.plot(0, 0, 'k+')
#     equaarea_factor = 2./np.sqrt(2)
#     circ = plt.Circle((0, 0), equaarea_factor*1.0, facecolor=facecolor, edgecolor='black')
#     ax.add_patch(circ)

#     ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
#   # equal aspect ratio
#     ax.axis('off')  # remove the box
#     #plt.show()
    
#     t=np.linspace(0,180,180*2+1)*np.pi/180;
#     xc = np.sin(t);
#     yc = np.cos(t);
#     AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
#     for an in AltitudeAngle:
#         RotY = passive_rotation(an, 'y')
#         Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

#         proj_dirs = equalarea_directions(Ccp)
#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

#     #Latitude Lines
#     LatitudeAngle = np.linspace(-90,90,37)*np.pi/180; 
#     #t=np.linspace(0,180,360*2+1)*np.pi/180;
#     zc = np.sin(t);
#     xc = np.cos(t);
#     for an in LatitudeAngle:#[0.]:#LatitudeAngle:
#         Rmeridian = np.cos(an);
#         px = Rmeridian*xc;
#         py = np.sin(an)*np.ones(t.shape);
#         pz = Rmeridian*zc;

#         proj_dirs = equalarea_directions(np.vstack((px,py,pz)))

#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
    
#     if basedirs:
#         an=45.*np.pi/180.;
#         an=0.*np.pi/180.;
#         Rotz = active_rotation(an, 'z')
#         an=np.arccos(1/np.sqrt(3));
#         an=0.*np.pi/180.;
#         Rotx = active_rotation(an, 'x')
#         dirs = np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections))
#         proj_dirs = equalarea_directions(dirs)
#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='b',marker='o',linestyle='')
#         for diri,proj_diri in zip(basicdirectionstext,np.transpose(proj_dirs)):
#             ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))

#     return fig,ax

# def stereotriangle02(ax=None,basedirs=False):
#     if ax==None:
#         fig, ax = plt.subplots()
#     else:
#         fig=[]
#     ax.tick_params(
#         axis='both',
#         which='both',
#         bottom=False,
#         top=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False)
#     ax.plot(0, 0, 'k+')
#     #circ = plt.Circle((0, 0), 1.0, facecolor=(210./255.,235./255.,255./255.), edgecolor='black')
#     #ax.add_patch(circ)

#     ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
#   # equal aspect ratio
#     ax.axis('off')  # remove the box
#     #plt.show()
    
#     normals = np.array([1,-1,0]);
#     arclength = 90#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
#     proj_normals, points = stereoprojection_planes(normals,arclength=arclength)
#     ax.plot(proj_normals[0,:], proj_normals[1,:], 'k')
    
#     t=np.linspace(0,45,180*2)*np.pi/180;
#     xc = np.cos(t);
#     yc = np.sin(t);
#     ax.plot(xc,yc, 'k')
#     ax.plot([0,1],[0,0], 'k')
    
#     dirs = np.column_stack([[0,0,1],[1,1,0],[1,0,0]]);
# #    dirs = np.column_stack([[0,0,1],[1,1,1],[0,1,1]]);
#     proj_dirs = stereoprojection_directions(dirs)
#     ax.plot(proj_dirs[0,:], proj_dirs[1,:], 'ro')
#     if basedirs:
#         for diri,proj_diri in zip(np.transpose(dirs),np.transpose(proj_dirs)):
#             if diri==[0,0,1]:
#                 ax.text(-0.03+proj_diri[0],0.03+proj_diri[1],str(diri))
#             else:
#                 ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))
    
#     return fig,ax

# def stereotriangle_ini(ax=None,basedirs=False):
#     #print('ok')
#     if ax==None:
#         fig, ax = plt.subplots()
#     else:
#         fig=[]
#     ax.tick_params(
#         axis='both',
#         which='both',
#         bottom=False,
#         top=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False)
#     ax.plot(0, 0, 'k+')
#     #circ = plt.Circle((0, 0), 1.0, facecolor=(210./255.,235./255.,255./255.), edgecolor='black')
#     #ax.add_patch(circ)

#     ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
#   # equal aspect ratio
#     ax.axis('off')  # remove the box
#     #plt.show()





    
#     normals = np.array([1,-1,0]);
# #    normals = np.array([1,1,1]);
#     proj_111=stereoprojection_directions(np.array([1,1,1]))
#     arclength = 55#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
#     proj_normals, points = stereoprojection_planes(normals,arclength=arclength,iniangle=35)
#     ax.plot(proj_normals[0,:], proj_normals[1,:], 'k')

#     normals = np.array([-1,0,1]);
#     arclength = 35#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
#     proj_normals, points = stereoprojection_planes(normals,arclength=arclength,iniangle=90)
#     ax.plot(proj_normals[0,:], proj_normals[1,:], 'k')
    
#     R = stereoprojection_directions(np.array([1,0,1]))[0]
#     ax.plot([0,R],[0,0], 'k')
# #    dirs = np.column_stack([[0,0,1],[1,1,0],[1,0,0]]);
#     dirs = np.column_stack([[0,0,1],[1,1,1],[1,0,1]]);
#     proj_dirs = stereoprojection_directions(dirs)
#     ax.plot(proj_dirs[0,:], proj_dirs[1,:], 'ro')
#     if basedirs:
#         for diri,proj_diri in zip(np.transpose(dirs),np.transpose(proj_dirs)):
#             ax.text(0.01+proj_diri[0],0.01+proj_diri[1],str(diri))
    
#     return fig,ax

# def stereotriangle(ax=None,basedirs=False,equalarea=False):
#     if ax==None:
#         fig, ax = plt.subplots()
#     else:
#         fig=[]
#     ax.tick_params(
#         axis='both',
#         which='both',
#         bottom=False,
#         top=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False)
#     ax.plot(0, 0, 'k+')
#     #circ = plt.Circle((0, 0), 1.0, facecolor=(210./255.,235./255.,255./255.), edgecolor='black')
#     #ax.add_patch(circ)

#     ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
  # equal aspect ratio
#     ax.axis('off')  # remove the box
#     #plt.show()

#     # t=np.linspace(0,180,180*2+1)*np.pi/180;
#     # xc = np.sin(t);
#     # yc = np.cos(t);
#     # AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
#     # for an in AltitudeAngle:
#     #     RotY = passive_rotation(an, 'y')
#     #     Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

#     #     proj_dirs = stereoprojection_directions(Ccp)
#     #     ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')




    
#     normals = np.array([1,-1,0]);
# #    normals = np.array([1,1,1]);
#     if equalarea:
#         proj_111=equalarea_directions(np.array([1,1,1]))
#     else:
#         proj_111=stereoprojection_directions(np.array([1,1,1]))
#     arclength = 55#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
    
#     if equalarea:
#         proj_normals = equalarea_planes(normals,arclength=arclength,iniangle=35)
#     else:
#         proj_normals,points = stereoprojection_planes(normals,arclength=arclength,iniangle=35)
        
#     ax.plot(proj_normals[0,:], proj_normals[1,:], 'k')

#     normals = np.array([-1,0,1]);
#     arclength = 35#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
#     if equalarea:
#         proj_normals = equalarea_planes(normals,arclength=arclength,iniangle=90)
#     else:
#         proj_normals,points = stereoprojection_planes(normals,arclength=arclength,iniangle=90)
#     ax.plot(proj_normals[0,:], proj_normals[1,:], 'k')
    
#     if equalarea:
#         R = equalarea_directions(np.array([1,0,1]))[0]
#     else:
#         R = stereoprojection_directions(np.array([1,0,1]))[0]
#     ax.plot([0,R],[0,0], 'k')
# #    dirs = np.column_stack([[0,0,1],[1,1,0],[1,0,0]]);

#     dirs = np.column_stack([[0,0,1],[1,1,1],[1,0,1]]);
    
    
#     if equalarea:
#         proj_dirs = equalarea_directions(dirs)
#     else:
#         proj_dirs = stereoprojection_directions(dirs)
#     ax.plot(proj_dirs[0,:], proj_dirs[1,:], 'ro')
#     if basedirs:
#         for diri,proj_diri in zip(np.transpose(dirs),np.transpose(proj_dirs)):
#             if diri==[0,0,1]:
#                 ax.text(-0.05+proj_diri[0],0.01+proj_diri[1],str(diri))
#             else:
#                 ax.text(0.01+proj_diri[0],0.01+proj_diri[1],str(diri))
#     # resolution=1
#     # grid_cub = get_beam_directions_grid("cubic", resolution, mesh="spherified_cube_edge")
#     # grid_stereo = Rotation.from_euler(np.deg2rad(grid_cub))*Vector3d.zvector()

#     # proj_Ds = stereoprojection_directions(grid_stereo.data.T)
#     # #Colors=stereotriangle_colors(proj_Ds)
#     # #fig,ax=stereotriangle(ax=None,basedirs=basedirs)
#     # ax.scatter(proj_Ds[0,:],proj_Ds[1,:],c='k',s=5)#,'.',color='r',markersize=1)
#     #plt.show()
    
#     return fig,ax

# def wulffnet_half(ax=None,basedirs=False,facecolor=(210./255.,235./255.,255./255.)):
#     if ax==None:
#         fig, ax = plt.subplots()
#     else:
#         fig=[]
#     if basedirs:
#         basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
#         #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
#         basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
#         #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];



#     #longitude lines
#     #fig, ax = plt.subplots()
#     ax.tick_params(
#         axis='both',
#         which='both',
#         bottom=False,
#         top=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False)
#     ax.plot(0, 0, 'k+')
    
    
    
    
#     w1 = Wedge((0,0), 1.0, 0, 180, fc=facecolor, edgecolor='black')
#     ax.add_artist(w1)
    
            
    
    
# #    circ = plt.Circle((0, 0), 1.0, facecolor=(210./255.,235./255.,255./255.), edgecolor='black')
# #    ax.add_patch(circ)

#     ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
#   # equal aspect ratio
#     ax.axis('off')  # remove the box
#     #plt.show()
    
#     t=np.linspace(0,90,180*2+1)*np.pi/180;
#     xc = np.sin(t);
#     yc = np.cos(t);
#     AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
#     for an in AltitudeAngle:
#         RotY = passive_rotation(an, 'y')
#         Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

#         proj_dirs = stereoprojection_directions(Ccp)
#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

#     #Latitude Lines
#     LatitudeAngle = np.linspace(0,90,37)*np.pi/90; 
#     #t=np.linspace(0,180,360*2+1)*np.pi/180;
#     zc = np.sin(t);
#     xc = np.cos(t);
#     for an in LatitudeAngle:#[0.]:#LatitudeAngle:
#         Rmeridian = np.cos(an);
#         px = Rmeridian*xc;
#         py = np.sin(an)*np.ones(t.shape);
#         pz = Rmeridian*zc;

#         proj_dirs = stereoprojection_directions(np.vstack((px,py,pz)))

#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
    
#     if basedirs:
#         an=45.*np.pi/180.;
#         an=0.*np.pi/180.;
#         Rotz = active_rotation(an, 'z')
#         an=np.arccos(1/np.sqrt(3));
#         an=0.*np.pi/180.;
#         Rotx = active_rotation(an, 'x')
#         dirs = np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections))
#         proj_dirs = stereoprojection_directions(dirs)
#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='b',marker='o',linestyle='')
#         for diri,proj_diri in zip(basicdirectionstext,np.transpose(proj_dirs)):
#             ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))

#     return fig,ax
# def schmidtnet_half(ax=None,basedirs=False,facecolor=(210./255.,235./255.,255./255.)):
#     if ax==None:
#         fig, ax = plt.subplots()
#     else:
#         fig=[]
#     if basedirs:
#         basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
#         #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
#         basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
#         #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];



#     #longitude lines
#     #fig, ax = plt.subplots()
#     ax.tick_params(
#         axis='both',
#         which='both',
#         bottom=False,
#         top=False,
#         left=False,
#         labelbottom=False,
#         labelleft=False)
#     ax.plot(0, 0, 'k+')
#     equaarea_factor = 2./np.sqrt(2)

#     w1 = Wedge((0,0), equaarea_factor*1.0, 0, 180, fc=facecolor, edgecolor='black')
#     ax.add_artist(w1)

#     #circ = plt.Circle((0, 0), equaarea_factor*1.0, facecolor=facecolor, edgecolor='black')
#     #ax.add_patch(circ)

#     ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
#   # equal aspect ratio
#     ax.axis('off')  # remove the box
#     #plt.show()
    
#     t=np.linspace(0,90,180*2+1)*np.pi/180;
#     xc = np.sin(t);
#     yc = np.cos(t);
#     AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
#     for an in AltitudeAngle:
#         RotY = passive_rotation(an, 'y')
#         Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

#         proj_dirs = equalarea_directions(Ccp)
#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

#     #Latitude Lines
#     LatitudeAngle = np.linspace(0,180,37)*np.pi/180; 
#     #t=np.linspace(0,180,360*2+1)*np.pi/180;
#     zc = np.sin(t);
#     xc = np.cos(t);
#     for an in LatitudeAngle:#[0.]:#LatitudeAngle:
#         Rmeridian = np.cos(an);
#         px = Rmeridian*xc;
#         py = np.sin(an)*np.ones(t.shape);
#         pz = Rmeridian*zc;

#         proj_dirs = equalarea_directions(np.vstack((px,py,pz)))

#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
    
#     if basedirs:
#         an=45.*np.pi/180.;
#         an=0.*np.pi/180.;
#         Rotz = active_rotation(an, 'z')
#         an=np.arccos(1/np.sqrt(3));
#         an=0.*np.pi/180.;
#         Rotx = active_rotation(an, 'x')
#         dirs = np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections))
#         proj_dirs = equalarea_directions(dirs)
#         ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='b',marker='o',linestyle='')
#         for diri,proj_diri in zip(basicdirectionstext,np.transpose(proj_dirs)):
#             ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))

#     return fig,ax

# def wulffnet_regular_grid_ini(ax,dangle,markersize=1):
#     #dphi=10.deg
#     #dtheta=10.deg
#     #dangle = 10.
    
#     Phi1=np.linspace(0.,360.-dangle,int(360./dangle))
#     Phi2=np.linspace(0.,180.-dangle,int(180./dangle))
#     GridX=[];
#     GridY=[];
#     Dc=[1.,0.,0.];
#     for phi1 in Phi1:
#         for phi2 in Phi2:        
#             RotZ = active_rotation(phi1, 'z', deg=True) 
#             RotY = active_rotation(phi2, 'y', deg=True)
#             Ds = np.matmul(RotY,RotZ).dot(Dc)
#             proj_Ds = stereoprojection_directions(Ds)
#             GridX.append(proj_Ds[0,0])
#             GridY.append(proj_Ds[1,0])
                
    

#     #fig,ax = wulffnet()
#     ax.plot(GridX,GridY,'.',color='r',markersize=markersize)
    
#     return GridX,GridY
# def wulffnet_regular_grid(ax,dangle,dirout=False, plot=True):
#     #dphi=10.deg
#     #dtheta=10.deg
#     #dangle = 10.
    
#     Phi1=np.linspace(0.,360.-dangle,int(360./dangle))
#     Phi2=np.linspace(0.,180.-dangle,int(180./dangle))
#     GridX=[];
#     GridY=[];
#     Dc=[1.,0.,0.];
#     dirs=[]
#     for phi1 in Phi1:
#         for phi2 in Phi2:        
#             RotZ = active_rotation(phi1, 'z', deg=True) 
#             RotY = active_rotation(phi2, 'y', deg=True)
#             Ds = np.matmul(RotY,RotZ).dot(Dc)
#             dirs.append(Ds)
#             proj_Ds = stereoprojection_directions(Ds)
#             GridX.append(proj_Ds[0,0])
#             GridY.append(proj_Ds[1,0])
                
    

#     #fig,ax = wulffnet()
#     if plot:
#         ax.plot(GridX,GridY,'.',color='r',markersize=1)
#     if dirout:
#         return GridX,GridY,dirs
#     else:
#         return GridX,GridY

# def schmidt_regular_grid_ini(ax,dangle):
#     #dphi=10.deg
#     #dtheta=10.deg
#     #dangle = 10.
    
#     Phi1=np.linspace(0.,360.-dangle,int(360./dangle))
#     Phi2=np.linspace(0.,180.-dangle,int(180./dangle))
#     GridX=[0.];
#     GridY=[0.];
#     Dc=[1.,0.,0.];
#     for phi1 in Phi1:
#         for phi2 in Phi2:        
#             RotZ = active_rotation(phi1, 'z', deg=True) 
#             RotY = active_rotation(phi2, 'y', deg=True)
#             Ds = np.matmul(RotY,RotZ).dot(Dc)
#             proj_Ds = equalarea_directions(Ds)
#             GridX.append(proj_Ds[0,0])
#             GridY.append(proj_Ds[1,0])
                
    

#     #fig,ax = wulffnet()
#     ax.plot(GridX,GridY,'.',color='r',markersize=1)
    
#     return GridX,GridY

# def schmidt_regular_grid(ax,Na=72,Nr=20,plot=True):
#     dphi1=360/Na
#     phi1=np.linspace(0,360-dphi1,Na)
#     R=equalarea_directions(np.array([1,0,0]))[0,0]
#     TotalArea=np.pi*R**2
#     dr=R/(Nr+0.5)
#     r=np.linspace(0,R-dr/2,Nr+1)
# #    GridX=[0.];
# #    GridY=[0.];
# #    Weight= np.pi*(r[1]/2)**2/TotalArea
#     AreaRatio=[]
#     for ri in r:
#         Nari=int(Na*(ri/r[-1]))
        
#         if Nari<8:
#             Nari=8
#         else:
#             Nari=Nari-Nari%8    
#         #print(Nari)
#         #Nari=8
#         dphi1=360./(Nari)
#         phi1=np.linspace(0,360-dphi1,Nari)
#         phi1=np.linspace(dphi1/2,360-dphi1/2,Nari)
#         #phi1=phi1[0:-1]
#         #print(2*np.pi*((ri+dr/2)**2/2-(ri-dr/2)**2/2))
#         #tot=0
#         for phi1i in phi1:        
#             if ri==0:
#                 GridX=[0.];
#                 GridPhi=[0.]
#                 GridY=[0.];
#                 GridR=[r[1]-dr/2]
#                 GridR=[ri]
#                 AreaRatio= [np.pi*(r[1]-dr/2.)**2/TotalArea]
#             else:
#                 GridX.append(ri*np.cos(phi1i*np.pi/180.))
#                 GridY.append(ri*np.sin(phi1i*np.pi/180.))
#                 AreaRatio.append(dphi1*np.pi/180.*((ri+dr/2)**2/2.-(ri-dr/2)**2/2.)/TotalArea)
#                 GridR.append(ri)
#                 phi=np.arctan2(GridY[-1],GridX[-1])*180./np.pi
#                 GridPhi.append(phi)
#                 #tot+=dphi1*np.pi/180.*((ri+dr/2)**2/2-(ri-dr/2)**2/2)
#         #print(tot)
#     #sum(AreaRatio)
#     if plot:
#         ax.plot(GridX,GridY,'.',color='r',markersize=1)
    
#     return GridX,GridY,GridR,GridPhi,AreaRatio

# def schmidt_triangle_regular_grid(ax,Na=72,Nr=20,plot=True):
#     amax =45
#     dphi1=amax/Na
#     phi1=np.linspace(0,amax-dphi1,Na)
#     R=np.linalg.norm(equalarea_directions(np.array([1,1,1]))[:,0])
#     Rw=equalarea_directions(np.array([1,0,0]))[0,0]
#     TotalArea=np.pi*Rw**2
#     dr=R/(Nr+0.5)
#     r=np.linspace(0,R-dr/2,Nr+1)
#     normals = np.array([-1,0,1]);
#     arclength = 35#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
#     proj_normals = equalarea_planes(normals,arclength=arclength,iniangle=90)
#     #ax.plot(proj_normals[0,:], proj_normals[1,:], 'k')
#     RN=np.sqrt(proj_normals[0,:]**2+proj_normals[1,:]**2)
#     AN=np.arctan(proj_normals[1,:]/proj_normals[0,:])*180/np.pi
# #    GridX=[0.];
# #    GridY=[0.];
# #    Weight= np.pi*(r[1]/2)**2/TotalArea
#     AreaRatio=[]
#     for ri in r:
#         Nari=int(Na*(ri/r[-1]))
        
#         if Nari<8:
#             Nari=8
#         else:
#             Nari=Nari-Nari%8    
#         #print(Nari)
#         #Nari=8
#         dphi1=amax/(Nari)
#         phi1=np.linspace(0,amax-dphi1,Nari)
#         phi1=np.linspace(dphi1/2,amax-dphi1/2,Nari)
#         #phi1=phi1[0:-1]
#         #print(2*np.pi*((ri+dr/2)**2/2-(ri-dr/2)**2/2))
#         #tot=0
#         for phi1i in phi1: 
            
#             rmax=RN[np.where(abs(AN-phi1i)==min(abs(AN-phi1i)))[0]]
            
#             #print(rmax)
#             if ri==0:
#                 GridX=[0.];
#                 GridPhi=[0.]
#                 GridY=[0.];
#                 GridR=[r[1]-dr/2]
#                 GridR=[ri]
#                 AreaRatio= [np.pi*(r[1]-dr/2.)**2/TotalArea]
#             elif ri<=rmax:
#                 GridX.append(ri*np.cos(phi1i*np.pi/180.))
#                 GridY.append(ri*np.sin(phi1i*np.pi/180.))
#                 AreaRatio.append(dphi1*np.pi/180.*((ri+dr/2)**2/2.-(ri-dr/2)**2/2.)/TotalArea)
#                 GridR.append(ri)
#                 phi=np.arctan2(GridY[-1],GridX[-1])*180./np.pi
#                 GridPhi.append(phi)
#                 #tot+=dphi1*np.pi/180.*((ri+dr/2)**2/2-(ri-dr/2)**2/2)
#         #print(tot)
#     #sum(AreaRatio)
#     if plot:
#         ax.plot(GridX,GridY,'.',color='r',markersize=1)
    
#     return GridX,GridY,GridR,GridPhi,AreaRatio

# def pf(gPhi1,gPHI,gPhi2,Dc,lattice,Na=72,Nr=20):
#     #fig,ax = schmidtnet()
#     GridX,GridY,GridR,GridPhi,AreaRatio=schmidt_regular_grid([],Na=Na,Nr=Nr,plot=False)
    
#     Intensity = np.array(GridX)*0
#     inc=0
#     #Dc=[1,0,0]
#     Syms = symmetry_elements(lattice)
    
#     for Phi1,PHI,Phi2 in zip(gPhi1,gPHI,gPhi2):
#         inc+=1
#         print(str(inc)+'/'+str(len(gPhi1)))
#         R = np.array(np_euler_matrix(Phi1, PHI,Phi2))
#         for Sym in Syms:
#             Ri=np.matmul(Sym,R)
#             Phi1,PHI,Phi2=euler_angles_from_matrix(Ri)
#             Phi1,PHI,Phi2 = euler_angles_reduction(Phi1,PHI,Phi2)
#             U = np_inverse_euler_matrix(Phi1, PHI,Phi2)
#             Ds = np.array(U).dot(Dc)
#             proj_Ds = equalarea_directions(Ds)
#             phi = np.arctan2(proj_Ds[1],proj_Ds[0])[0]*180./np.pi
#             r=np.sqrt(proj_Ds[:,0].dot(proj_Ds[:,0]))
    
#             dr=np.array(GridR)-r
#             idxmin=np.where(abs(dr)==min(abs(dr)))
#             ri=GridR[idxmin[0][0]]
    
#             idxr=np.where(np.array(GridR)==ri)[0]
#             dtan = np.array(GridPhi)[idxr]-phi
#             idxtan = np.where(abs(dtan)==min(abs(dtan)))
#             idxmin3=idxr[idxtan[0]][0]
            
#     #        dx = np.array(GridX)-proj_Ds[0];
#     #        dy = np.array(GridY)-proj_Ds[1];
#     #        dr = dx**2+dy**2
#     #        idxmin = np.where(dr==min(dr))[0][0]
#             Intensity[idxmin3]+=1/AreaRatio[idxmin3]
#     #        GridX[idxmin]
#     #        GridY[idxmin]
    
    
    
        
#     #fig, ax = plt.subplots()
#     fig,ax = schmidtnet()
#     GridX,GridY,GridR,GridPhi,AreaRatio=schmidt_regular_grid(ax,Na=Na,Nr=Nr,plot=True)

#     plt.scatter(GridX,GridY, c=Intensity, s=50, edgecolor='',zorder=10,cmap='jet')
#     cb=plt.colorbar()
#     #plt.show()
def B19p_B2_lattice_correspondence(notation='Miyazaki'):
    #check
    #Variant = 8
    #B19puvw_2_B2uvw_all[:,:,Variant]
    #testB19v=[1,2,3]
    #testB2v = B19puvw_2_B2uvw_all[:,:,Variant].dot(testB2v)            
    #testB19v-np.linalg.inv(B19puvw_2_B2uvw_all[:,:,Variant]).dot(testB2v)
    #testB19p=[1,2,1]
    #testB2p = B19phkl_2_B2hkl_all[:,:,Variant].dot(testB19p)            
    #testB19p-np.linalg.inv(B19phkl_2_B2hkl_all[:,:,Variant]).dot(testB2p)
    #testB2p=[1,-3,1]

    #correspondance matrix betwen uvw of B19p and that of B2 /B19[uvw]->B2[uvw]/
    B19puvw_2_B2uvw_all = np.empty((3,3,12))
    B19phkl_2_B2hkl_all = np.empty((3,3,12))
    B2uvw_2_B19puvw_all = np.empty((3,3,12))
    B2hkl_2_B19phkl_all = np.empty((3,3,12))
    
    for Variant in range(B19puvw_2_B2uvw_all.shape[2]):
        
        if (Variant+1)==1:
            
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,1,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
            elif notation=='Waitz':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,1,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,1,1];

            
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;          
                
    #            B19puvw_2_B2uvw_all[:,0,Variant] = [-1, 0, 0];
    #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, 1,-1];
    #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1];
                
    #            B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0];
    #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, 1,1];
    #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
    #            
    
        if (Variant+1)==2:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ -1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,-1,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
            elif notation=='Waitz':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,-1,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1];
                
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
                
    #            B19puvw_2_B2uvw_all[:,0,Variant] = [-1, 0, 0];
    #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, -1,-1];
    #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
    #
        if (Variant+1)==3:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, -1,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1];
            elif notation=='Waitz':
                B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, 1,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
    
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
                
    #            B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0];
    #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, -1,1];
    #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1];
    
        if (Variant+1)==4:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [-1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,1,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1];
            elif notation=='Waitz':
                B19puvw_2_B2uvw_all[:,0,Variant] = [1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,-1,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,1,-1];
            
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==5:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [1,0,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,0,-1];
            elif notation=='Waitz':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,-1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [-1,0,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,0,-1];
                
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==6:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,-1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ -1,0,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,0,-1];
            elif notation=='Waitz':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,-1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [1,0,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,0,1];
        
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==7:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 1,0,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ -1,0,-1];
            elif notation=='Waitz':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 1,0,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 1,0,-1];
        
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==8:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,-1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [-1,0,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ -1,0,-1];
            elif notation=='Waitz':        
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [-1,0,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ -1,0,1];
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
                
        if (Variant+1)==9:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 1,1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,1,0];
            elif notation=='Waitz':        
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,-1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 1,-1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,-1,0];
        
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==10:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,-1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ -1,-1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,1,0];
            elif notation=='Waitz':        
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,-1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ -1,1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,1,0];
        
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==11:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ -1,1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,-1,0];
            elif notation=='Waitz':        
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ -1,-1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,-1,0];
    
        
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==12:
            if notation=='Miyazaki':
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,-1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [1,-1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,-1,0];
            elif notation=='Waitz':        
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 1,1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,1,0];
          
            B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        
        B2uvw_2_B19puvw_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]);
        B2hkl_2_B19phkl_all[:,:,Variant] = inv(B19phkl_2_B2hkl_all[:,:,Variant]);         

    return B19puvw_2_B2uvw_all,B2uvw_2_B19puvw_all,B19phkl_2_B2hkl_all,B2hkl_2_B19phkl_all

def lattice_correspondence(LatCorr,parent_symops,product_symops):
    #LatCorr=np.empty((3,3,1))
    #Variant=0
    #LatCorr[:,0,Variant] = [ 1,0,0];
    #LatCorr[:,1,Variant] = [ 0,1,-1];
    #LatCorr[:,2,Variant] = [ 0,1,1];
    count=-1
    #print(LatCorr[:,:,0])
    for syms in parent_symops:
        count+=1
        isin=False
        for var in range(LatCorr.shape[2]):#check if not yet included
            for symsB19p in product_symops:#take into account symmetry of product lattice
                if (abs(LatCorr[:,:,var]-syms.dot(LatCorr[:,:,0].dot(symsB19p)))<1e-10).all():# or (abs(-1*LatCorr[:,:,var]-syms.dot(LatCorr[:,:,0]))<1e-10).all():
                    isin=True
    
        if not isin and (syms.dot(LatCorr[:,:,0])[:,0]>=0).all() and np.linalg.det(syms.dot(LatCorr[:,:,0]))>0:#select correspondence that preserves right-handed system between [100]_a, [010]_a [001]_a
            #Variant+=1
            #LatCorr[:,:,Variant]=syms.dot(LatCorr[:,:,0])
            LatCorr=np.concatenate((LatCorr,np.expand_dims(syms.dot(LatCorr[:,:,0]),axis=(2))),axis=2)
            #print(f'count={count}')
            #print(LatCorr[:,:,-1])
            #print

def B19p_B2_lattice_correspondence_ini():
    #check
    #Variant = 8
    #B19puvw_2_B2uvw_all[:,:,Variant]
    #testB19v=[1,2,3]
    #testB2v = B19puvw_2_B2uvw_all[:,:,Variant].dot(testB2v)            
    #testB19v-np.linalg.inv(B19puvw_2_B2uvw_all[:,:,Variant]).dot(testB2v)
    #testB19p=[1,2,1]
    #testB2p = B19phkl_2_B2hkl_all[:,:,Variant].dot(testB19p)            
    #testB19p-np.linalg.inv(B19phkl_2_B2hkl_all[:,:,Variant]).dot(testB2p)
    #testB2p=[1,-3,1]

    #correspondance matrix betwen uvw of B19p and that of B2 /B19[uvw]->B2[uvw]/
    B19puvw_2_B2uvw_all = np.empty((3,3,12))
    B19phkl_2_B2hkl_all = np.empty((3,3,12))
    B2uvw_2_B19puvw_all = np.empty((3,3,12))
    B2hkl_2_B19phkl_all = np.empty((3,3,12))
    
    for Variant in range(B19puvw_2_B2uvw_all.shape[2]):
        
        if (Variant+1)==1:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,-1,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,1,1];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;          
                
    #            B19puvw_2_B2uvw_all[:,0,Variant] = [-1, 0, 0];
    #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, 1,-1];
    #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1];
                
    #            B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0];
    #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, 1,1];
    #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
    #            
    
        if (Variant+1)==2:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ -1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,1,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,1,1];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
                
    #            B19puvw_2_B2uvw_all[:,0,Variant] = [-1, 0, 0];
    #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, -1,-1];
    #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
    #
        if (Variant+1)==3:
                B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, -1,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
    
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
                
    #            B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0];
    #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, -1,1];
    #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1];
    
        if (Variant+1)==4:
                B19puvw_2_B2uvw_all[:,0,Variant] = [-1,0,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 0,1,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==5:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [1,0,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,0,1];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==6:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,-1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ -1,0,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,0,1];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==7:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,-1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 1,0,1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 1,0,-1];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==8:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,1,0];
                B19puvw_2_B2uvw_all[:,1,Variant] = [-1,0,-1];
                B19puvw_2_B2uvw_all[:,2,Variant] = [ 1,0,-1];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
                
        if (Variant+1)==9:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ -1,1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,1,0];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==10:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,-1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 1,-1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [1,1,0];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==11:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,-1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [ 1,1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,1,0];
    
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        if (Variant+1)==12:
                B19puvw_2_B2uvw_all[:,0,Variant] = [ 0,0,1];
                B19puvw_2_B2uvw_all[:,1,Variant] = [-1,-1,0];
                B19puvw_2_B2uvw_all[:,2,Variant] = [-1,1,0];
                B19phkl_2_B2hkl_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]).T;
        
        B2uvw_2_B19puvw_all[:,:,Variant] = inv(B19puvw_2_B2uvw_all[:,:,Variant]);
        B2hkl_2_B19phkl_all[:,:,Variant] = inv(B19phkl_2_B2hkl_all[:,:,Variant]);         

    return B19puvw_2_B2uvw_all,B2uvw_2_B19puvw_all,B19phkl_2_B2hkl_all,B2hkl_2_B19phkl_all
#nn=0
#for ii in range(0,12):
#    Cd=B19puvw_2_B2uvw_all[:,:,ii];
#    for i in range(0,2):
#        for j in range(0,2):
#            if Cd[i,j]==round(Cd[i,j]):
#                Cd[i,j]=round(Cd[i,j])
#    nn+=1
#    #print("\\begin{{pmatrix}}{} & {} &{}\\\\{} &{} &{}\\\\{} &{} &{}\\end{{pmatrix}}".format(Cd[0,0],Cd[0,1],Cd[0,2],Cd[1,0],Cd[1,1],Cd[1,2],Cd[2,0],Cd[2,1],Cd[2,2]))
#    print("\\Bigl(\\begin{{smallmatrix}}{} & {} &{}\\\\{} &{} &{}\\\\{} &{} &{}\\end{{smallmatrix}}\\Bigr)".format(Cd[0,0],Cd[0,1],Cd[0,2],Cd[1,0],Cd[1,1],Cd[1,2],Cd[2,0],Cd[2,1],Cd[2,2]))
#    print('&')
#    Cd=B19phkl_2_B2hkl_all[:,:,ii]
#    #print("\\begin{{pmatrix}}{} & {} &{}\\\\{} &{} &{}\\\\{} &{} &{}\\end{{pmatrix}}".format(Cd[0,0],Cd[0,1],Cd[0,2],Cd[1,0],Cd[1,1],Cd[1,2],Cd[2,0],Cd[2,1],Cd[2,2]))
#    print("\\Bigl(\\begin{{smallmatrix}}{} & {} &{}\\\\{} &{} &{}\\\\{} &{} &{}\\end{{smallmatrix}}\\Bigr)".format(Cd[0,0],Cd[0,1],Cd[0,2],Cd[1,0],Cd[1,1],Cd[1,2],Cd[2,0],Cd[2,1],Cd[2,2]))
#    if nn<4:
#        print('&')
#    else:
#        print('\\\\')
#        print('\hline')
#        nn=0
##    print(B19puvw_2_B2uvw_all[:,:,ii])
#    print(B19phkl_2_B2hkl_all[:,:,ii])
#    print('============================================')
    
    
def cubic2tetragonal_lattice_correspondence():
    #check
    #Variant = 8
    #B19puvw_2_B2uvw_all[:,:,Variant]
    #testB19v=[1,2,3]
    #testB2v = B19puvw_2_B2uvw_all[:,:,Variant].dot(testB2v)            
    #testB19v-np.linalg.inv(B19puvw_2_B2uvw_all[:,:,Variant]).dot(testB2v)
    #testB19p=[1,2,1]
    #testB2p = B19phkl_2_B2hkl_all[:,:,Variant].dot(testB19p)            
    #testB19p-np.linalg.inv(B19phkl_2_B2hkl_all[:,:,Variant]).dot(testB2p)
    #testB2p=[1,-3,1]

    #correspondance matrix betwen uvw of B19p and that of B2 /B19[uvw]->B2[uvw]/
    Productuvw_2_Parentuvw_all = np.empty((3,3,3))
    Producthkl_2_Parenthkl_all = np.empty((3,3,3))
    Parentuvw_2_Productuvw_all = np.empty((3,3,3))
    Parenthkl_2_Producthkl_all = np.empty((3,3,3))
    
    for Variant in range(Productuvw_2_Parentuvw_all.shape[2]):
        
        if (Variant+1)==1:
                Productuvw_2_Parentuvw_all[:,0,Variant] = [ 0,1,1];
                Productuvw_2_Parentuvw_all[:,1,Variant] = [ 0,-1,1];
                Productuvw_2_Parentuvw_all[:,2,Variant] = [ 1,0,0];
                Producthkl_2_Parenthkl_all[:,:,Variant] = inv(Productuvw_2_Parentuvw_all[:,:,Variant]).T;          
                
        if (Variant+1)==2:
                Productuvw_2_Parentuvw_all[:,0,Variant] = [ 1,0,-1];
                Productuvw_2_Parentuvw_all[:,1,Variant] = [ 1,0,1];
                Productuvw_2_Parentuvw_all[:,2,Variant] = [ 0,1,0];
                Producthkl_2_Parenthkl_all[:,:,Variant] = inv(Productuvw_2_Parentuvw_all[:,:,Variant]).T;          

        if (Variant+1)==3:
                Productuvw_2_Parentuvw_all[:,0,Variant] = [ 1,1,0];
                Productuvw_2_Parentuvw_all[:,1,Variant] = [ -1,1,0];
                Productuvw_2_Parentuvw_all[:,2,Variant] = [ 0,0,1];
                Producthkl_2_Parenthkl_all[:,:,Variant] = inv(Productuvw_2_Parentuvw_all[:,:,Variant]).T;          


        
        Parentuvw_2_Productuvw_all[:,:,Variant] = inv(Productuvw_2_Parentuvw_all[:,:,Variant]);
        Parenthkl_2_Producthkl_all[:,:,Variant] = inv(Producthkl_2_Parenthkl_all[:,:,Variant]);         

    return Parentuvw_2_Productuvw_all,Productuvw_2_Parentuvw_all,Parenthkl_2_Producthkl_all,Producthkl_2_Parenthkl_all
def Rp_B2_lattice_correspondence():
    #correspondance matrix betwen uvw of Rp and that of B2 /B19[uvw]->B2[uvw]/
    Rpuvw_2_B2uvw_all = np.empty((3,3,4))
    Rphkl_2_B2hkl_all = np.empty((3,3,4))
    B2uvw_2_Rpuvw_all = np.empty((3,3,4))
    B2hkl_2_Rphkl_all = np.empty((3,3,4))
    
    for Variant in range(Rpuvw_2_B2uvw_all.shape[2]):
        
        if (Variant+1)==1:
                Rpuvw_2_B2uvw_all[:,0,Variant] = [ 1,-2,1];
                Rpuvw_2_B2uvw_all[:,1,Variant] = [ 1,1,-2];
                Rpuvw_2_B2uvw_all[:,2,Variant] = [ 1,1,1];
                Rphkl_2_B2hkl_all[:,:,Variant] = inv(Rpuvw_2_B2uvw_all[:,:,Variant]).T;          
                
    #            
    
        if (Variant+1)==2:
                Rpuvw_2_B2uvw_all[:,0,Variant] = [ 2,1,1];
                Rpuvw_2_B2uvw_all[:,1,Variant] = [ -1,1,-2];
                Rpuvw_2_B2uvw_all[:,2,Variant] = [ -1,1,1];
                Rphkl_2_B2hkl_all[:,:,Variant] = inv(Rpuvw_2_B2uvw_all[:,:,Variant]).T;
                
    #
        if (Variant+1)==3:
                Rpuvw_2_B2uvw_all[:,0,Variant] = [1,2,1];
                Rpuvw_2_B2uvw_all[:,0,Variant] = [1,-2,-1];
                Rpuvw_2_B2uvw_all[:,1,Variant] = [1,1,2];
                #Rpuvw_2_B2uvw_all[:,1,Variant] = [1,-1,-2];
                Rpuvw_2_B2uvw_all[:,2,Variant] = [ -1,-1,1];
    
                Rphkl_2_B2hkl_all[:,:,Variant] = inv(Rpuvw_2_B2uvw_all[:,:,Variant]).T;
                
    
        if (Variant+1)==4:
                Rpuvw_2_B2uvw_all[:,0,Variant] = [2,1,1];
                Rpuvw_2_B2uvw_all[:,0,Variant] = [2,1,-1];
                Rpuvw_2_B2uvw_all[:,1,Variant] = [ 1,1,-2];
                Rpuvw_2_B2uvw_all[:,1,Variant] = [ -1,1,2];
                Rpuvw_2_B2uvw_all[:,2,Variant] = [ 1,-1,1];
                Rphkl_2_B2hkl_all[:,:,Variant] = inv(Rpuvw_2_B2uvw_all[:,:,Variant]).T;

        
        B2uvw_2_Rpuvw_all[:,:,Variant] = inv(Rpuvw_2_B2uvw_all[:,:,Variant]);
        B2hkl_2_Rphkl_all[:,:,Variant] = inv(Rphkl_2_B2hkl_all[:,:,Variant]);         

    return Rpuvw_2_B2uvw_all,B2uvw_2_Rpuvw_all,Rphkl_2_B2hkl_all,B2hkl_2_Rphkl_all

def print_correspondence(Mcorr,VecA,latticeA, latticeB,planes=False,returnB=False):
    if type(VecA[0])==list:
        VecA=np.array(VecA)
    if not VecA.ndim==2:
        VecA=np.expand_dims(VecA, axis=0)    
    if not VecA.shape[0]==3:
        VecA=VecA.T
    #print basal directions
    if planes:
        strfunction='plane2string'
    else:
        strfunction='dir2string'
        
    BasalDirs = np.empty((3,3,Mcorr.shape[2]))
    VecB = np.empty((3,VecA.shape[1],Mcorr.shape[2]))
    print('================================================================================')
    print('Basal directions')
    print('================================================================================')
    for var in range(0,Mcorr.shape[2]):
        print("\t"+'--------------------------------------------------------------------------------')
        print("\t"+'Variant '+str(var+1))
        print("\t"+'--------------------------------------------------------------------------------')
        for ei,i in zip(np.eye(3),range(0,3)):
            strvari=eval(strfunction+'(ei)')+"_"+latticeA+"=="+eval(strfunction+'(Mcorr[:,i,var])')+"_"+latticeB;
            BasalDirs[:,i,var]=Mcorr[:,i,var]
            print("\t"+strvari)

    print('================================================================================')
    print('Input directions')
    print('================================================================================')
    #print(VecA.shape)
    for var in range(0,Mcorr.shape[2]):
        print("\t"+'--------------------------------------------------------------------------------')
        print("\t"+'Variant '+str(var+1))
        print("\t"+'--------------------------------------------------------------------------------')
        inc=-1
        for va in VecA.T:
            inc+=1
#            print(va)
#            print('================================================================================')
#            print('Direction '+dir2string(va)+"_"+latticeA)
#            print('================================================================================')
#            print(va)
            vb = Mcorr[:,:,var].dot(va)
            if not planes:
                vb=miller2fractional(vb)
            str2print=eval(strfunction+'(va)')+"_"+latticeA+"=="+eval(strfunction+'(vb)')+"_"+latticeB;
            print("\t"+str2print)
            #print(vb)
            VecB[:,inc,var]=vb
 
    if returnB:
        return VecB
            #    var2=dir2string([0,1,0])+"_B19'=="+dir2string(B19puvw_2_B2uvw_all[:,1,var])+'_B2'
#    var3=dir2string([0,0,1])+"_B19'=="+dir2string(B19puvw_2_B2uvw_all[:,2,var])+'_B2'
#    #print('Varianta '+str(var+1)+':'+var1+','+var2+','+var3)
#    print('Varianta '+str(var+1))
#    print(plane2string(b19pvar, digits=2))

def mohr_circles(tensor):
    DD,VV = np.linalg.eig(tensor)
    Idxs = np.argsort(DD)[::-1]      
    DD=DD[Idxs]
    #print(DD)
    VV=VV[:,Idxs]    
    #EpsT =  np.matmul(matmul(VV.T,tensor),VV)
    
    
    #plot mohr cicrles
    R13=(DD[0]-DD[2])/2
    C13=DD[0]-R13
    R12=(DD[0]-DD[1])/2
    C12=DD[0]-R12#(DD[0]+DD[1])/2
    R23=(DD[1]-DD[2])/2
    C23=DD[1]-R23

    mohr_circles={'C12':C12,'C13':C13,'C23':C23,'R12':R12,'R13':R13,'R23':R23,\
                  'D1':DD[0],'D2':DD[1],'D3':DD[2],\
                  'V1':VV[:,0],'V2':VV[:,1],'V3':VV[:,2]}
    return mohr_circles,VV,DD

def generate_lattice_points(uvw2xyz,basal_dirs):
    Lattice=[]
    #basic lattice
    for idx in range(0,basal_dirs.shape[1]):
        v=uvw2xyz.dot(basal_dirs[:,idx])
        Lattice.append([[0,v[0]],[0,v[1]],[0,v[2]]])
        v3=np.array([0,0,0]);
        for idx2 in range(0,basal_dirs.shape[1]):
            if not idx2==idx:
                v2=uvw2xyz.dot(basal_dirs[:,idx2])
                Lattice.append([[0,v[0]]+v2[0],[0,v[1]]+v2[1],[0,v[2]]+v2[2]])
                v3=v2+v3
        Lattice.append([[0,v[0]]+v3[0],[0,v[1]]+v3[1],[0,v[2]]+v3[2]])
        Lattice2=copy.deepcopy(Lattice)
    #Translation of basic lattice
    #Lattice=[]
    for basal_dir in basal_dirs.T:
        TransDir = -1.
        Trans_vec = TransDir*uvw2xyz.dot(basal_dir)
        for point in Lattice2:
            trpoint=[];
            for p,trv in zip(point,Trans_vec):
                trpoint.append([p[0]+trv,p[1]+trv])
            Lattice.append(trpoint)
    for idxs in [[0,1],[0,2],[1,2],[0,1,2]]:
        TransDir = -1.
        Trans_vec=np.array([0.,0.,0.]);
        for idx in idxs:
            Trans_vec+=TransDir*uvw2xyz.dot(basal_dirs[:,idx])
        for point in Lattice2:
            trpoint=[];
            for p,trv in zip(point,Trans_vec):
                trpoint.append([p[0]+trv,p[1]+trv])
            Lattice.append(trpoint)
            
    #remove duplicates            
    Lattice2=[];
    eps=1.e-6
    for points in Lattice:
        if len(Lattice2)==0:
            Lattice2.append(points)
        else:
            isin=False
            for points2 in Lattice2:
                for point2,point in zip(points2,points):
                    if abs(point2[0]-point[0])<eps and abs(point2[1]-point[1])<eps:
                        isin=True
                    else:False
            if not isin:
                Lattice2.append(points)
#    Lattice=[]
#    for phix in [0,180]:    
#        Rx=passive_rotation(phix, 'x', deg=True)     
#
#        for phiz in [0,90,180,270]:    
#            Rz=passive_rotation(phiz, 'z', deg=True)     
#            for point in Lattice2:
#                p0=[]
#                p1=[]
#                for p in point:
#                    p0.append(p[0])    
#                    p1.append(p[1])    
#                p0r=Rz.dot(Rx.dot(p0))
#                p1r=Rz.dot(Rx.dot(p1))
#                B2point=[]
#                for p0ri,p1ri in zip(p0r,p1r):
#                    B2point.append([p0ri,p1ri])
#                Lattice.append(B2point)
    return Lattice
def plot_lattice_plane(axl,PlanePoints,**kwargs):
    BasalPlane=False
    for idx in range(3):
        if (PlanePoints[idx,:]==0).all():
            BasalPlane=True
            break
    
    if BasalPlane:
        idxs=list(range(3))
        idxs.remove(idx)
        hull = ConvexHull(PlanePoints[idxs,:].T)
    
        axl.add_collection3d(Poly3DCollection([PlanePoints[:,hull.vertices].T],**kwargs))
    else:
        axl.plot_trisurf(PlanePoints[0,:],PlanePoints[1,:], PlanePoints[2,:],**kwargs)

def plot_lattice_boundaries(axl,LatticePointsNew,allPoints=None,polygon=False,tol=1e-1,**kwargs):
    if allPoints is None:
        allPoints=np.hstack([p for points in LatticePointsNew for p in points])
    if polygon:
        for idx in range(3):
            Xbound=copy.deepcopy(allPoints)
            for extrval in [np.min(Xbound[idx,:]),np.max(Xbound[idx,:])]:
                Xbound=copy.deepcopy(allPoints)
                #Xbound=Xbound[:,Xbound[idx,:]==extrval]
                Xbound=Xbound[:,np.abs(Xbound[idx,:]-extrval)<tol]
                #np.abs(vertices[0,:]-np.min(vertices[0,:]))<1e-1
                #Xbound[idx,:]=Xbound[idx,:]*0+extrval
                #axl.plot_trisurf(Xbound[0,:],Xbound[1,:], Xbound[2,:],\
                #                 alpha=0.5,color='r', linewidths=0., edgecolors='grey',linestyle='-',\
                #                 linewidth = 0.0, antialiased = True) 
                if Xbound.shape[1]>=3:
                    try:
                        hull = ConvexHull(Xbound[np.delete(range(3),idx,0),:].T)
                        axl.add_collection3d(Poly3DCollection([Xbound[:,hull.vertices].T], **kwargs))
                    except:
                        pass
    else:
        axl.plot_trisurf(allPoints[0,:],allPoints[1,:], allPoints[2,:],triangles=ConvexHull(allPoints.T).simplices,
                         **kwargs)

            

def generate_lattice_faces(uvw2xyz,basal_dirs):

#uvw2xyz=Parent_uvw2xyz,basal_dirs=Parent
    LatticeFaces=[]
    #basic lattice
    for idx in range(0,basal_dirs.shape[1]):
        idxs=list(range(basal_dirs.shape[1]))#[0,1,2];
        idxs.remove(idx)
        face=[]
        face.append(np.array([0.,0.,0.]))
        for idx2 in idxs:
            face.append(face[-1]+uvw2xyz.dot(basal_dirs[:,idx2]))
        face.append(face[-1]-uvw2xyz.dot(basal_dirs[:,idxs[0]]))    
        face.append(face[0])    
        LatticeFaces.append(face)
        face2=copy.deepcopy(face)
        for i in range(0,len(face2)):
            face2[i]=face2[i]+uvw2xyz.dot(basal_dirs[:,idx])
        LatticeFaces.append(face2)
    
    LatticeFaces2=copy.deepcopy(LatticeFaces)
    #LatticeFaces=[]
    Lattices=[]
    Lattices.append(LatticeFaces)        
    #Translation of basic lattice
    #Lattice=[]
    for basal_dir in basal_dirs.T:
        LatticeFaces=[]
        TransDir = -1.
        Trans_vec = TransDir*uvw2xyz.dot(basal_dir)
        for face in LatticeFaces2:            
            trface=[];
            for edge in face:
                trface.append(edge+Trans_vec)
            LatticeFaces.append(trface)
        Lattices.append(LatticeFaces)
    for idxs in [[0,1],[0,2],[1,2],[0,1,2]]:
        LatticeFaces=[]
        TransDir = -1.
        Trans_vec=np.array([0.,0.,0.]);
        for idx in idxs:
            Trans_vec+=TransDir*uvw2xyz.dot(basal_dirs[:,idx])
        for face in LatticeFaces2:
            trface=[];
            for edge in face:
                trface.append(edge+Trans_vec)
            LatticeFaces.append(trface)
        Lattices.append(LatticeFaces)

            
    #remove duplicates            
    Lattices2=[];
    eps=1.e-6
    for lattice in Lattices:
        if len(Lattices2)==0:
            Lattices2.append(lattice)
        else:
            Lat=[]
            for face in lattice:
                ISIN=False
                for lattice2 in Lattices2:
                    for face2 in lattice2:
                        isin=False
                        for edge2,edge in zip(face2,face):
                            if abs(edge.dot(edge2)-edge.dot(edge))<eps:
                                isin=True
                            else:
                                isin=False
                        if isin:
                            ISIN=True
                        
                if not ISIN:
                    Lat.append(face)
                    
            Lattices2.append(Lat)
    return Lattices2

def generate_product_lattice_points(F,Parentlattice_points,Q=np.eye(3)):
    #F-deformation gradient
    Productlattice_points=[]
    
    for v in Parentlattice_points:
        p0=np.array([vi[0] for vi in v])
        p1=np.array([vi[1] for vi in v])
#        if not (p0==0).all():
#            pn0=p0/np.sqrt(p0.dot(p0))
#            p0B19p = np.ndarray.tolist(pn0.dot(F.dot(p0))*pn0)
#        else:
#            p0B19p = np.ndarray.tolist(p0);
#            
#        if not (p1==0).all():
#            pn1=p1/np.sqrt(p1.dot(p1))
#            p1B19p = np.ndarray.tolist(pn1.dot(F.dot(p1))*pn1)
#        else:
#            p1B19p = np.ndarray.tolist(p1);
            
        p0B19p=Q.dot(F.dot(p0))
        p1B19p=Q.dot(F.dot(p1))
        Productlattice_points.append([[p0B19pi,p1B19pi] for p0B19pi,p1B19pi in zip(p0B19p,p1B19p) ])
    
    return Productlattice_points
def generate_product_lattice_faces(F,Parentlattices):
    #F-deformation gradient
    Productlattices=[]
    for lattice in Parentlattices:
        plattice=[]
        for face in lattice:
            pface=[]
            for edge in face:
                pface.append(F.dot(edge))
            plattice.append(pface)
        Productlattices.append(plattice)
    
    return Productlattices


def plot_lattice3D(ax,VV,description,Parentlattice_points,Productlattice_points,Product_uvw_2_Parent_uvw_all_norm,Product_uvw2xyz,linewidth=2):
    xlim= np.array([-1.05,1.05])*np.sqrt(Product_uvw2xyz[:,0].dot(Product_uvw2xyz[:,0]))
    for point in Parentlattice_points:
        ax.plot(point[0],point[1],point[2],'r')
                
    for point in Productlattice_points:
        ax.plot(point[0],point[1],point[2],'b')
        
        
    Product_basal = np.matmul(Product_uvw_2_Parent_uvw_all_norm,np.matmul(Product_uvw2xyz,np.eye(3)))
    colors=['g','c','#800000']
        
    inc=-1;
    for v2 in Product_basal.T:
        inc+=1
        ax.plot([0,v2[0]],[0,v2[1]],[0,v2[2]],color=colors[inc],linewidth=linewidth)
    inc=0

    for v in VV.T:
        inc+=1
        #print(v)
        v2=1.5*np.sqrt(Product_uvw2xyz[:,0].dot(Product_uvw2xyz[:,0]))*v
        ax.plot([0,v2[0]],[0,v2[1]],[0,v2[2]],'k',linewidth=linewidth,linestyle='--')
        ax.text(v2[0],v2[1],v2[2],description.replace('{inc}','{'+str(inc)+'}'))       
        
    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
# equal aspect ratio
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(xlim)
    ax.set_zlim3d(xlim)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_zlim(xlim)
        
  

    set_aspect_equal_3d(ax)

def plot_latticefaces3D(ax,Parentlattices,linewidth=2,alpha=0.15,edgecolor='r',linestyle='-',facecolor=(1, 0, 0, 0.15)):
    
    for Lattice in Parentlattices:
        ax.add_collection3d(Poly3DCollection(Lattice, alpha=alpha,facecolors=facecolor, linewidths=linewidth, edgecolors=edgecolor,linestyle=linestyle))
        
    maxlimits=[]
    minlimits=[]
        
    for Lattice in Parentlattices:
        for face in Lattice:
            for point in face:
                if len(maxlimits)==0:
                    maxlimits=[point[0],point[1],point[2]]
                else:
                    for ii in range(0,3):
                        if maxlimits[ii]<point[ii]:
                            maxlimits[ii]=point[ii]
                if len(minlimits)==0:
                    minlimits=[point[0],point[1],point[2]]
                else:
                    for ii in range(0,3):
                        if minlimits[ii]>point[ii]:
                            minlimits[ii]=point[ii]
                    
                
    #ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
# equal aspect ratio
    ax.set_xlim3d([minlimits[0],maxlimits[0]])
    ax.set_ylim3d([minlimits[1],maxlimits[1]])
    ax.set_zlim3d([minlimits[2],maxlimits[2]])
    set_aspect_equal_3d(ax)

def plot_latticesfaces3D(ax,VV,description,Parentlattices,Productlattices,Product_uvw_2_Parent_uvw_all_norm,Product_uvw2xyz,linewidth=2,alpha=0.15,xlim=[-2,2]):
    if xlim is None:
        xlim= np.array([-1.05,1.05])*max([np.sqrt(V.dot(V)) for V in Product_uvw2xyz.T])

    for Lattice in Parentlattices:
        ax.add_collection3d(Poly3DCollection(Lattice, alpha=alpha,facecolors=(1, 0, 0, alpha), linewidths=linewidth, edgecolors='r'))
    for Lattice in Productlattices:
        ax.add_collection3d(Poly3DCollection(Lattice, alpha=alpha,facecolors=(0, 0, 1, alpha), linewidths=linewidth, edgecolors='b'))
        
        
    Product_basal = np.matmul(Product_uvw_2_Parent_uvw_all_norm,np.matmul(Product_uvw2xyz,np.eye(3)))
    colors=['g','c','#800000']
        
    inc=-1;
    for v2 in Product_basal.T:
        inc+=1
        ax.plot([0,v2[0]],[0,v2[1]],[0,v2[2]],color=colors[inc],linewidth=linewidth)
    inc=0

    for v in VV.T:
        inc+=1
        #print(v)
        v2=1.5*np.sqrt(Product_uvw2xyz[:,0].dot(Product_uvw2xyz[:,0]))*v
        ax.plot([0,v2[0]],[0,v2[1]],[0,v2[2]],'k',linewidth=linewidth,linestyle='--')
        ax.text(v2[0],v2[1],v2[2],description.replace('{inc}','{'+str(inc)+'}'))       
        
    #ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
# equal aspect ratio
    ax.axis('auto')
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(xlim)
    ax.set_zlim3d(xlim)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_zlim(xlim)
        
  

    set_aspect_equal_3d(ax)




def plot_lattice2D(ax,VV,description,Parentlattice_points,Parent_lattice,\
                   Productlattice_points,Product_uvw_2_Parent_uvw_all_norm,Product_uvw2xyz,linewidth=2,xlim=None):
    if xlim is None:
        xlim= np.array([-1.05,1.05])*max([np.sqrt(V.dot(V)) for V in Product_uvw2xyz.T])
    shiftx=xlim[1];
    shifty=xlim[1]*0;
    facx=1.8
    facy=1.3
    shifts=[[facx*xlim[1],-facy*xlim[1]],[-facx*xlim[1],-facy*xlim[1]],[-facx*xlim[1],facy*xlim[1]]]
    pairs=[[0,1],[2,1],[0,2]]
    signs = [[1,1],[-1,1],[-1,1]]
    coordlength=xlim[1]*0.5
    coordabasalvecs = [[[1,0,0],[0,1,0]],[[0,0,1],[0,1,0]],[[0,0,1],[1,0,0]]]
    colors=['g','c','#800000']
    for shift,pair,sgn,vecs in zip(shifts,pairs,signs,coordabasalvecs):
#        for point,vecs in zip([[xlim[1]*0.5,0],[0,xlim[1]*0.5]],):
            #point=np.array(point);
        ax.plot(np.array([0,sgn[0]*coordlength])+2*shift[0],np.array([0,0])+2*shift[1],'k')
        ax.plot(np.array([0,0])+2*shift[0],np.array([0,sgn[1]*coordlength])+2*shift[1],'k')
#        ax.plot(np.array([0,sgn[0]*coordpoint[0]])+2*shift[0],np.array([0,sgn[1]*coordpoint[1]])+2*shift[1],'k')
        ax.text(2*shift[0],sgn[1]*coordlength+2*shift[1], dir2string(vecs[1], digits=0)+r'$_{'+Parent_lattice+'}$',fontsize=10)
        addshiftx=0.0
        addshifty=0.0
        if sgn[0]<0:
            addshiftx=-coordlength*1;
            addshifty=coordlength*0.1;
        ax.text(sgn[0]*coordlength+2*shift[0]+addshiftx,addshifty+2*shift[1], dir2string(vecs[0], digits=0)+r'$_{'+Parent_lattice+'}$',fontsize=10)
            
        for point in Parentlattice_points:
            point=np.array(point);
            ax.plot(sgn[0]*point[pair[0]]+shift[0],sgn[1]*point[pair[1]]+shift[1],'r')
                    
        for point in Productlattice_points:
            point=np.array(point);
            ax.plot(sgn[0]*point[pair[0]]+shift[0],sgn[1]*point[pair[1]]+shift[1],'b')
    
        inc=-1;
        Product_basal = np.matmul(Product_uvw_2_Parent_uvw_all_norm,np.matmul(Product_uvw2xyz,np.eye(3)))
        for v2 in Product_basal.T:
            inc+=1
            ax.plot(sgn[0]*np.array([0,v2[pair[0]]])+shift[0],sgn[1]*np.array([0,v2[pair[1]]])+shift[1],color=colors[inc],linewidth=linewidth)
    
        inc=0
        for v in VV.T:
            inc+=1
            v2=1.8*np.sqrt(Product_uvw2xyz[:,0].dot(Product_uvw2xyz[:,0]))*v
            ax.plot(sgn[0]*np.array([0,v2[pair[0]]])+shift[0],sgn[1]*np.array([0,v2[pair[1]]])+shift[1],'k',linewidth=linewidth,linestyle='--')
            ax.text(sgn[0]*v2[pair[0]]*1.1+shift[0],sgn[1]*v2[pair[1]]*1.1+shift[1],description.replace('{inc}','{'+str(inc)+'}'))       

    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
# equal aspect ratio
    ax.set_xlim([-facx*2*xlim[1],facx*2*xlim[1]])
    ax.set_ylim([-facx*2*xlim[1],facx*2*xlim[1]])

def plot_lattice_2Dprojection(ax,VV,description,Parentlattice_points,Parent_lattice,\
                   Productlattice_points,Product_uvw_2_Parent_uvw_all_norm,Product_uvw2xyz,normals,verticals, linewidth=2,xlim=None):
    if xlim is None:
        xlim= np.array([-1.05,1.05])*max([np.sqrt(V.dot(V)) for V in Product_uvw2xyz.T])
    shiftx=xlim[1];
    shifty=xlim[1]*0;
    facx=1.8
    facy=1.3
    shifts=[[facx*xlim[1],-facy*xlim[1]],[-1.5*facx*xlim[1],-facy*xlim[1]],[-1.5*facx*xlim[1],1.4*facy*xlim[1]]]
    pairs=[[0,1],[2,1],[0,2]]
    signs = [[1,1],[-1,1],[-1,1]]
#    normals = [[0.,0.,1.],[1.,0.,0.],[0.,-1.,0.]]
#    verticals = [[0.,1.,0.],[0.,1.,0.],[1.,0.,0.]]
#    normals = [[0.,0.,1.],[1.,0.,0.],[-1.,1.,0.]]
#    verticals = [[0.,1.,0.],[0.,1.,0.],[0.,0.,1.]]
    coordlength=xlim[1]*0.5
    coordabasalvecs = [[[1,0,0],[0,1,0]],[[0,0,1],[0,1,0]],[[0,0,1],[1,0,0]]]
    colors=['g','c','#800000']
    
    for shift,normal,vertical,vecs in zip(shifts,normals,verticals,coordabasalvecs):
        normal=np.array(normal);
        vertical=np.array(vertical);
        horizontal=np.cross(vertical,normal)
        
        ax.plot(np.array([0,coordlength])+2*shift[0],np.array([0,0])+2*shift[1],'k')
        ax.plot(np.array([0,0])+2*shift[0],np.array([0,coordlength])+2*shift[1],'k')
        ax.text(2*shift[0],coordlength+2*shift[1], dir2string(vertical, digits=1)+r'$_{'+Parent_lattice+'}$',fontsize=10)
        #print(vertical)
        addshiftx=0.0
        addshifty=0.0
#        if sgn[0]<0:
#            addshiftx=-coordlength*1;
#            addshifty=coordlength*0.1;
        ax.text(coordlength+2*shift[0]+addshiftx,addshifty+2*shift[1], dir2string(horizontal, digits=1)+r'$_{'+Parent_lattice+'}$',fontsize=10)
        ax.text(-coordlength+2*shift[0]+addshiftx,-0.5*coordlength+addshifty+2*shift[1], plane2string(normal, digits=1)+r'$_{'+Parent_lattice+'}$',fontsize=10)
#        print(normal)
#        print(plane2string(normal, digits=0))
        for point in Parentlattice_points:
            point=np.array(point);
            point_proj_x = horizontal.dot(point)
            point_proj_y = vertical.dot(point)
            ax.plot(point_proj_x+shift[0],point_proj_y+shift[1],'r')
                    
        for point in Productlattice_points:
            point=np.array(point);
            point_proj_x = horizontal.dot(point)
            point_proj_y = vertical.dot(point)
            ax.plot(point_proj_x+shift[0],point_proj_y+shift[1],'b')
            
        inc=-1;
        Product_basal = np.matmul(Product_uvw_2_Parent_uvw_all_norm,np.matmul(Product_uvw2xyz,np.eye(3)))
        for v2 in Product_basal.T:
            inc+=1
            point_proj_x = horizontal.dot(v2)
            point_proj_y = vertical.dot(v2)

            ax.plot(np.array([0,point_proj_x])+shift[0],np.array([0,point_proj_y])+shift[1],color=colors[inc],linewidth=linewidth)
    
        inc=0
        for v in VV.T:
            inc+=1
            v2=1.8*np.sqrt(Product_uvw2xyz[:,0].dot(Product_uvw2xyz[:,0]))*v
            point_proj_x = horizontal.dot(v2)
            point_proj_y = vertical.dot(v2)
            
            ax.plot(np.array([0,point_proj_x])+shift[0],np.array([0,point_proj_y])+shift[1],'k',linewidth=linewidth,linestyle='--')
            ax.text(point_proj_x*1.1+shift[0],point_proj_y*1.1+shift[1],description.replace('{inc}','{'+str(inc)+'}'))       

    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
# equal aspect ratio
    ax.set_xlim([-facx*2*xlim[1],facx*2*xlim[1]])
    ax.set_ylim([-facx*2*xlim[1],facx*2*xlim[1]])        
    
def zero_normal_strains(Strain, mcircles,VV,normdiri,phi_around_normdiri,Parent_xyz2hkl):
    Re =  mcircles['R13']
    dRe = mcircles['C13']
    #phi_around_normdiri=np.linspace(0.,360,361);
    sheardiri=np.cross(normdiri,[0.,1.,0.])
    an = np.arccos(dRe/Re)/2.*180./np.pi        

    #strains, planes for orientation where normal strain==0 and \
    #for 90deg oriented plane 
    Shear={'rotabout':'V2','angle':[],'strainmag':[],'inprincipaldir':[],'inlatticedir':[],'hkldir':[]}
    Normal={'rotabout':'V2','angle':[],'strainmag':[],'inprincipaldir':[],'inlatticedir':[],'hkldir':[]}
    #strains in these planes rotated around normdiri by phi_around_normdiri
    Shears={'rotabout':'V1','angle':[],'angleaboutV2':[],'strainmag':[],'inprincipaldir':[],'inlatticedir':[],'hkldir':[]}
    Normals={'rotabout':'V1','angle':[],'angleaboutV2':[],'strainmag':[],'inprincipaldir':[],'inlatticedir':[],'hkldir':[]}
    InPlaneShears={'rotabout':[],'angle':[],'angleaboutV2':[],'strainmag':[],'inprincipaldir':[],'inlatticedir':[],'hkldir':[]}
    InPlaneNormals={'rotabout':[],'angle':[],'angleaboutV2':[],'strainmag':[],'inprincipaldir':[],'inlatticedir':[],'hkldir':[]}


    Shear['strainmag'].append([])
    Normal['strainmag'].append([])
    Shear['inprincipaldir'].append([])
    Normal['inprincipaldir'].append([])
    Shear['inlatticedir'].append([])
    Normal['inlatticedir'].append([])
    Shear['hkldir'].append([])
    Normal['hkldir'].append([])
    Shear['angle'].append([])
    Normal['angle'].append([])

    
    #sheardiri=[0,0,1]

    for angle in [an,an+90.]:
#        normdiri=[0,0,1]
#        sheardiri=[1,0,0]
        
        R=passive_rotation(angle,'y',deg=True)      

        Shear['inprincipaldir'][-1].append(R.dot(sheardiri))      
        Normal['inprincipaldir'][-1].append(R.dot(normdiri)) 
        Shear['inlatticedir'][-1].append(np.matmul(VV,R).dot(sheardiri))
        Normal['inlatticedir'][-1].append(np.matmul(VV,R).dot(normdiri)) 
        Shear['hkldir'][-1].append(xyz2fractional(Parent_xyz2hkl,np.matmul(VV,R).dot(sheardiri)))
        Normal['hkldir'][-1].append(xyz2fractional(Parent_xyz2hkl,np.matmul(VV,R).dot(normdiri)))
        
        Shear['strainmag'][-1].append(Shear['inlatticedir'][-1][-1].dot(Strain.dot(Normal['inlatticedir'][-1][-1])))
        Normal['strainmag'][-1].append(Normal['inlatticedir'][-1][-1].dot(Strain.dot(Normal['inlatticedir'][-1][-1])))
        
        Shear['angle'][-1].append(angle)
        Normal['angle'][-1].append(angle)
        InPlaneShears['angleaboutV2'].append(angle)
        InPlaneNormals['angleaboutV2'].append(angle)

        InPlaneShears['strainmag'].append([])
        InPlaneNormals['strainmag'].append([])
        InPlaneShears['inprincipaldir'].append([])
        InPlaneNormals['inprincipaldir'].append([])
        InPlaneShears['inlatticedir'].append([])
        InPlaneNormals['inlatticedir'].append([])
        InPlaneShears['hkldir'].append([])
        InPlaneNormals['hkldir'].append([])
        InPlaneShears['angle'].append([])
        InPlaneNormals['angle'].append([])

        if normdiri==[1,0,0]:
            InPlaneShears['rotabout']='V1'
            InPlaneNormals['rotabout']='V1'
        else:
            InPlaneShears['rotabout']='V3'
            InPlaneNormals['rotabout']='V3'

        StrainPrincipal=np.matmul(np.matmul(VV.T,Strain),VV)
        Strain_in_zero = np.matmul(np.matmul(R,StrainPrincipal),R.T)
        eps=1e-10;
        if Strain_in_zero[0,0]<eps:
            rotaxisidx = 2
#            InPlaneShears['rotabout'].append('V3')
#            InPlaneNormals['rotabout'].append('V3')
            inplanesheardiri=[1.,0.,0.]  
        else:
            rotaxisidx = 0
#            InPlaneShears['rotabout'].append('V1')
#            InPlaneNormals['rotabout'].append('V1')
            inplanesheardiri=[0.,0.,1.]  

        inplanenormdiri=[0.,1.,0.]   
        for phii in phi_around_normdiri:

            InPlaneShears['angle'][-1].append([phii])
            InPlaneNormals['angle'][-1].append([phii])

#            if normdiri==[1,0,0]:
#                R2=passive_rotation(phii,'x',deg=True) 
#            else:
#                R2=passive_rotation(phii,'z',deg=True)
            if rotaxisidx == 0:
                R2=passive_rotation(phii,'x',deg=True) 
                Norm=R.T.dot([1.,0.,0.])
            else:
                R2=passive_rotation(phii,'z',deg=True)
                Norm=R.T.dot([0.,0.,1.])

            ShDir = R2.dot(R.dot(inplanesheardiri))
            NoDir = R2.dot(R.dot(inplanenormdiri))
                
            ShDirg = VV.dot(ShDir)
            NoDirg = VV.dot(NoDir)

#            ShDirg = np.matmul(VV,R).dot(R2.dot(sheardiri))   
#            NoDirg = np.matmul(VV,R).dot(NoDir)
#            Strain_in_zero.dot()
#            Strain_in_zero_rot = np.matmul(np.matmul(R2,Strain_in_zero),R2.T)
#            vv=np.array([0,1,0])
#            vv.dot(Strain_in_zero_rot.dot([0,0,1]))
#            NoDir.dot(StrainPrincipal.dot(Norm))
#                                ShDir.dot(StrainPrincipal.dot(Norm))
#            
#            np.matmul(np.matmul(R2,Strain_in_zero),R2.T)
            
            InPlaneShears['strainmag'][-1].append(ShDirg.dot(Strain.dot(NoDirg)))
            InPlaneNormals['strainmag'][-1].append(NoDirg.dot(Strain.dot(NoDirg)))
            InPlaneShears['inprincipaldir'][-1].append(ShDir)
            InPlaneNormals['inprincipaldir'][-1].append(NoDir)
            InPlaneShears['inlatticedir'][-1].append(ShDirg)
            InPlaneNormals['inlatticedir'][-1].append(NoDirg)
            InPlaneShears['hkldir'][-1].append(xyz2fractional(Parent_xyz2hkl,ShDirg))
            print(NoDirg)
            InPlaneNormals['hkldir'][-1].append(xyz2fractional(Parent_xyz2hkl,NoDirg))

    return     Shear,Normal,InPlaneShears,InPlaneNormals,[an,an+90.]

def strains_along_13mohrcirle(Strain,VV,normdiri,phi_around_V2,Parent_xyz2hkl):
    ShearsOnCircle={'rotabout':'V2','angle':[],'strainmag':[],'inprincipaldir':[],'inlatticedir':[],'hkldir':[]}
    NormalsOnCircle={'rotabout':'V2','angle':[],'strainmag':[],'inprincipaldir':[],'inlatticedir':[],'hkldir':[]}

    
    sheardiri=np.cross(normdiri,[0.,1.,0.])
    
    for angle in phi_around_V2:
#        ShearsOnCircle['inprincipaldir'].append([])
#        NormalsOnCircle['inprincipaldir'].append([])
#        ShearsOnCircle['inlatticedir'].append([])
#        NormalsOnCircle['inlatticedir'].append([])
#        ShearsOnCircle['hkldir'].append([])
#        NormalsOnCircle['hkldir'].append([])
        
        ShDir = []
        NoDir = []
        ShDirg = []
        NoDirg = []
        ShDirhkl = []
        NoDirhkl = []
        a=[]
        for halfcircle in [0,90.]:
        
            anglei=angle+halfcircle
    
            R=passive_rotation(anglei,'y',deg=True)      
    
            ShDir.append(R.dot(sheardiri))
            NoDir.append(R.dot(normdiri))
            
                
            ShDirg.append(np.matmul(VV,R).dot(sheardiri))   
            NoDirg.append(np.matmul(VV,R).dot(normdiri))
            ShDirhkl.append(xyz2fractional(Parent_xyz2hkl,ShDirg[-1]))
            NoDirhkl.append(xyz2fractional(Parent_xyz2hkl,NoDirg[-1]))
    
        isin=False
        for nn in NormalsOnCircle['hkldir']:
            if (nn[1]==NoDirhkl[1]).all() and (nn[0]==NoDirhkl[0]).all():
                isin=True
                #print(nn[0])
        if not isin and NoDirhkl[1].dot(NoDirhkl[0])==0.:
        #if not isin:
            
            ShearsOnCircle['strainmag'].append([])
            NormalsOnCircle['strainmag'].append([])
            ShearsOnCircle['angle'].append([])
            NormalsOnCircle['angle'].append([])

            ShearsOnCircle['angle'][-1].append([angle,angle+halfcircle])
            NormalsOnCircle['angle'][-1].append([angle,angle+halfcircle])
            for sh,no in zip(ShDirg,NoDirg): 
                ShearsOnCircle['strainmag'][-1].append(sh.dot(Strain.dot(no)))
                NormalsOnCircle['strainmag'][-1].append(no.dot(Strain.dot(no)))
            ShearsOnCircle['inprincipaldir'].append(ShDir)
            NormalsOnCircle['inprincipaldir'].append(NoDir)
            ShearsOnCircle['inlatticedir'].append(ShDirg)
            NormalsOnCircle['inlatticedir'].append(NoDirg)
            ShearsOnCircle['hkldir'].append(ShDirhkl)
            NormalsOnCircle['hkldir'].append(NoDirhkl)
            
            
    return ShearsOnCircle, NormalsOnCircle

def select_crystal_planes(NormalsOnCircle,ShearsOnCircle,maxhkl):
    
    #NormalsAlli=[]
    idxmore=[]
    inc=-1
    #maxhkl=5
    for norm in NormalsOnCircle['hkldir']:
        inc+=1
        uvwi=norm[0]
        if uvwi[0]%1==0 and uvwi[1]%1==0 and uvwi[2]%1==0 and max(abs(uvwi))<=maxhkl:
            idxmore.append(inc)
            #NormalsAlli.append([np.round(norm[0]),np.round(norm[1])])
        else:
            uvwi=norm[1]
            if uvwi[0]%1==0 and uvwi[1]%1==0 and uvwi[2]%1==0 and max(abs(uvwi))<=maxhkl:
                idxmore.append(inc)
                #NormalsAlli.append([np.round(norm[0]),np.round(norm[1])])
            
    SelShearsOnCircle={}
    SelNormalsOnCircle={}
    for key in ShearsOnCircle.keys():
        if type(ShearsOnCircle[key])==list:
            SelShearsOnCircle[key]=[]
            SelNormalsOnCircle[key]=[]
        else:
            SelShearsOnCircle[key]=ShearsOnCircle[key]
            SelNormalsOnCircle[key]=NormalsOnCircle[key]
            
    for key in SelShearsOnCircle.keys():
        if type(SelShearsOnCircle[key])==list:
            for i in idxmore:
                SelShearsOnCircle[key].append(ShearsOnCircle[key][i])
                SelNormalsOnCircle[key].append(NormalsOnCircle[key][i])
    return  SelShearsOnCircle, SelNormalsOnCircle           
                

def plot_mohr_circles(mcircles,VV,DD,xyz2uvw,scale,xticks=None,yticks=None,ax=None,Parent_lattice='B2'):
    Return=True
    if ax==None:
        fig, ax = plt.subplots()
    else:
        fig=[]
        Return=False
#    fig, ax = plt.subplots()
#    ax.tick_params(
#        axis='both',
#        which='both',
#        bottom=False,
#        top=False,
#        left=False,
#        labelbottom=False,
#        labelleft=False)    
    phi=np.linspace(0,2*np.pi,1000)
    C13x = mcircles['C13']+mcircles['R13']*np.cos(phi)
    C13y = mcircles['R13']*np.sin(phi)
    C23x = mcircles['C23']+mcircles['R23']*np.cos(phi)
    C23y = mcircles['R23']*np.sin(phi)
    C12x = mcircles['C12']+mcircles['R12']*np.cos(phi)
    C12y = mcircles['R12']*np.sin(phi)
    
    ax.plot(C13x*scale,C13y*scale,'r')
    ax.plot(C12x*scale,C12y*scale,'g')
    ax.plot(C23x*scale,C23y*scale,'b')
    #spine placement data centered
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    
    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    xlim = [np.round(C13x.min()*scale*1.5),np.round(C13x.max()*scale*1.5)]
    ylim = [np.round(C13y.max()*scale*1.5),np.round(C13y.min()*scale*1.5)]
    
    if xticks is None:
        xlim2 = [np.round(C13x.min()*scale*1.5),np.round(C13x.max()*scale*1.5)]
        ax.set_xticks(np.round(np.linspace(xlim[0],xlim[1],10)))
    else:
        xlim2 = [xticks[0],xticks[-1]]
        ax.set_xticks(xticks)
    if yticks is None:
        ylim2 = [np.round(C13y.max()*scale*1.5),np.round(C13y.min()*scale*1.5)]
        ax.set_yticks(np.round(np.linspace(ylim[0],ylim[1],10)))
    else:
        ylim2 = [yticks[0],yticks[-1]]
        ax.set_yticks(yticks)
    ax.set_xlim(xlim2)
    ax.set_ylim(ylim2)
    
    #ax.set_yticks(np.linspace(9,-9,10))
    #ax.set_ylim([9,-10])
    ax.text(xlim[1]*0.65,max(ylim)/10*2,'Normal\nstrain, '+r'$\varepsilon$ [%]')
    ax.text(-max(ylim)/10*2,ylim[1]*0.75, 'Shear\nstrain, '+r'$\gamma$/2 [%]')
#    ax.set_xlim([-10,14])
#    ax.set_xticks(np.linspace(-10,12,12))
#    ax.set_ylim([9,-10])
#    ax.set_yticks(np.linspace(9,-9,10))
#    ax.set_ylim([9,-10])
#    ax.text(11,2,'Normal\nstrain, '+r'$\varepsilon$ [%]')
#    ax.text(-2,-9.5, 'Shear\nstrain, '+r'$\gamma$/2 [%]')
    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
  # equal aspect ratio
    #plt.show()
    
    #plot principal direction in mohr circles
    for i in [0,1,2]:
        ax.plot(DD[i]*scale,0,'ko')
        ax.text(DD[i]*scale+max(ylim)/100*2,-max(ylim)/100*2,r'$\varepsilon_'+str(i+1)+'$')
    textprincipaldirs=''
    for i in [0,1,2]:
        vd = xyz2fractional(xyz2uvw,VV[:,i])
        textprincipaldirs+=str(r'$\varepsilon_'+str(i+1)+'$~'+vec2string(vd)+'$_{'+Parent_lattice+'}$\n')
    ax.text(max(xlim)/2.2,max(ylim)/1.,textprincipaldirs)
    
    if Return:
        return fig,ax



def plot_planes_on_mohr_circle(ax,scale,SelNormalsOnCircle,SelShearsOnCircle,Parent_xyz2hkl, colors,text=False,Parent_lattice='B2'):
    #plot selected planes on 13 mohr cicle and triangle and wulffnet
    inc=-1;
    Upperhalftext = []
    Lowerhalftext=[]
    
    for ii in range(0,len(SelNormalsOnCircle['strainmag'])):

        if type(colors) is list:
            if len(colors)>=len(SelNormalsOnCircle['strainmag']):
                inc+=1
            else:
                colors=[colors[0]]
                inc=0
        else:
            colors=[colors]
            inc=0
            
        xx=[]
        yy=[]
        uvwtext=[]
        for i in [0,1]:
            xx.append(SelNormalsOnCircle['strainmag'][ii][i]*scale)
            yy.append(SelShearsOnCircle['strainmag'][ii][i]*scale)
            vd = xyz2fractional(Parent_xyz2hkl,SelNormalsOnCircle['inlatticedir'][ii][i])
            proj_dir=stereoprojection_intotriangle(SelNormalsOnCircle['inlatticedir'][ii][i])
            proj_dir2=stereoprojection_directions(SelNormalsOnCircle['inlatticedir'][ii][i])
    
            if yy[-1]>0:
                lowerhalftext=r'$\mathbf{\varepsilon}}$='+str(round((xx[-1]*10))/10)+','+r'$\mathbf{\gamma}$/2='+str(round((yy[-1]*10))/10)+',n='+r''+plane2string(vd)+'$_{\mathbf{'+Parent_lattice+'}}$'
                Lowerhalftext.append(lowerhalftext)
                lowerhalftext=r'$\varepsilon$='+str(round((xx[-1]*10))/10)+r',$\gamma$/2='+str(round((yy[-1]*10))/10)+' ,n='+plane2string(vd)+'$_{'+Parent_lattice+'}$'
                markerfacecolor=colors[inc]
                markersize=8
            else:
                markerfacecolor='None'
                markersize=12
                upperhalftext =r'$\mathbf{\varepsilon}$='+str(round((xx[-1]*10))/10)+','+r'$\mathbf{\gamma}$/2='+str(round((yy[-1]*10))/10)+',n='+r''+plane2string(vd)+'$_{\mathbf{'+Parent_lattice+'}}$'
                Upperhalftext.append(upperhalftext)
                upperhalftext=r'$\varepsilon$='+str(round((xx[-1]*10))/10)+r',$\gamma$/2='+str(round((yy[-1]*10))/10)+' ,n='+plane2string(vd)+'$_{'+Parent_lattice+'}$'
            ax.plot(xx[-1],yy[-1],'o',markerfacecolor=markerfacecolor,markeredgecolor=colors[inc])
        ax.plot(xx,yy,color=colors[inc])
        if text:
            idx = yy.index(max(yy))
            idx2 = yy.index(min(yy))
            ax.text(xx[idx],yy[idx]*1.1,lowerhalftext)
            ax.text(xx[idx2],yy[idx2]*1.05,upperhalftext)

    return Upperhalftext,Lowerhalftext

def plot_planes_on_stereotriangle(ax,SelNormalsOnCircle,SelShearsOnCircle,Parent_xyz2hkl,colors):
    inc=-1;
    for ii in range(0,len(SelNormalsOnCircle['strainmag'])):
        if type(colors) is list:
            if len(colors)>=len(SelNormalsOnCircle['strainmag']):
                inc+=1
            else:
                colors=[colors[0]]
                inc=0
        else:
            colors=[colors]
            inc=0
            
        xx=[]
        yy=[]
        for i in [0,1]:
            xx.append(SelNormalsOnCircle['strainmag'][ii][i])
            yy.append(SelShearsOnCircle['strainmag'][ii][i])
            proj_dir=stereoprojection_intotriangle(SelNormalsOnCircle['inlatticedir'][ii][i])
    
            if yy[-1]>0:
                markerfacecolor=colors[inc]
                markersize=8
            else:
                markerfacecolor='None'
                markersize=12
            ax.plot(proj_dir[0,:], proj_dir[1,:],'o',markerfacecolor=markerfacecolor,markeredgecolor=colors[inc],\
                     markeredgewidth=2,markersize=markersize)
#            ax4.plot(proj_dir2[0,:], proj_dir2[1,:],'o',markerfacecolor=markerfacecolor,markeredgecolor=colors[inc],\
#                     markeredgewidth=2,markersize=markersize)


def plot_planes_on_wulffnet(ax,SelNormalsOnCircle,SelShearsOnCircle,Parent_xyz2hkl,colors):
    inc=-1;
    for ii in range(0,len(SelNormalsOnCircle['strainmag'])):
        if type(colors) is list:
            if len(colors)>=len(SelNormalsOnCircle['strainmag']):
                inc+=1
            else:
                colors=[colors[0]]
                inc=0
        else:
            colors=[colors]
            inc=0
            
        xx=[]
        yy=[]
        for i in [0,1]:
            xx.append(SelNormalsOnCircle['strainmag'][ii][i])
            yy.append(SelShearsOnCircle['strainmag'][ii][i])
            proj_dir2=stereoprojection_directions(SelNormalsOnCircle['inlatticedir'][ii][i])
    
            if yy[-1]>0:
                markerfacecolor=colors[inc]
                markersize=8
            else:
                markerfacecolor='None'
                markersize=12
            ax.plot(proj_dir2[0,:], proj_dir2[1,:],'o',markerfacecolor=markerfacecolor,markeredgecolor=colors[inc],\
                     markeredgewidth=2,markersize=markersize)

def plot_princip_dir_on_stereotriangle(ax,VV,description,markersize=10,markerfacecolor='None',markeredgecolor='k',markeredgewidth=1.5):
    proj_dirs = stereoprojection_intotriangle(VV)
    
    for proj_dir,i in zip(proj_dirs.T,[0,1,2]):
        ax.plot(proj_dir[0], proj_dir[1], 'o',markerfacecolor=markerfacecolor,\
                 markeredgecolor=markeredgecolor,markeredgewidth=markeredgewidth,markersize=markersize)
        text=str(description.replace('{inc}','{'+str(i+1)+'}'))
        ax.text(proj_dir[0]-0.03, proj_dir[1]+0.01,text)

def plot_princip_dir_on_wulffnet(ax,VV,description,markersize=10,markerfacecolor='None',markeredgecolor='k',markeredgewidth=1.5):
    proj_dirs = stereoprojection_directions(VV)
    
    for proj_dir,i in zip(proj_dirs.T,[0,1,2]):
        ax.plot(proj_dir[0], proj_dir[1], 'o',markerfacecolor=markerfacecolor,\
                 markeredgecolor=markeredgecolor,markeredgewidth=markeredgewidth,markersize=markersize)
        text=str(description.replace('{inc}','{'+str(i+1)+'}'))
#        ax2.text(proj_dir[0]-0.03, proj_dir[1]+0.01,text)
        ax.text(proj_dir[0]-0.1, proj_dir[1]+0.03,text)
        
def write_mohr_planes(ax,Upperhalftext,Lowerhalftext,colors,markersize=8,markeredgewidth=2):
    #write planes
    uppery=1
    lowery=0.45
    xall=0    
    inc=-1;
    for ut,lt in zip(Upperhalftext,Lowerhalftext):          
        if type(colors) is list:
            if len(colors)>=len(Upperhalftext):
                inc+=1
            else:
                colors=[colors[0]]
                inc=0
        else:
            colors=[colors]
            inc=0
        ax.text(xall,uppery,ut,color=colors[inc],fontsize=10,fontweight="bold")
        ax.text(xall,lowery,lt,color=colors[inc],fontsize=10,fontweight="bold")
        uppery-=0.035
        lowery-=0.035

#    uppery-=0.035
#    lowery-=0.035
    xall+= 1./len(Upperhalftext)/4   
    inc=-1;
    for ut,lt in zip(Upperhalftext,Lowerhalftext):     
        if type(colors) is list:
            if len(colors)>=len(Upperhalftext):
                inc+=1
            else:
                colors=[colors[0]]
                inc=0
        else:
            colors=[colors]
            inc=0
        markerfacecolor='None'
        ax.plot(xall,uppery,'o',markerfacecolor=markerfacecolor,markeredgecolor=colors[inc],\
                     markeredgewidth=markeredgewidth,markersize=markersize+4)
        markerfacecolor=colors[inc]
        ax.plot(xall,lowery,'o',markerfacecolor=markerfacecolor,markeredgecolor=colors[inc],\
                     markeredgewidth=markeredgewidth,markersize=markersize)
        xall+=1./len(Upperhalftext)
    ax.plot([0,1],[0.5,0.5],'k')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])


def write_lattice_correspondence(ax,Product_uvw_2_Parent_uvw_all_norm,Product_uvw2xyz,Product_lattice,Parent_lattice,FontSize=10,Fontweight="bold"):
    #write lattice correspondence
    tt=''
    for vm in [[1,0,0],[0,1,0],[0,0,1]]:
        va = Product_uvw_2_Parent_uvw_all_norm.dot(vm)
        vmreal = Product_uvw2xyz.dot(vm)
        
        tt += r''+vec2string(va, digits=0)+'$^{\mathbf{uvw}}_{\mathbf{'+Parent_lattice+'}}$'+' -> '+ \
        vec2string(vm, digits=0)+"$^{\mathbf{uvw}}_{\mathbf{"+Product_lattice+"}}$="+\
        vec2string(vmreal,digits=3)+"$^{\mathbf{xyz}}_{\mathbf{"+Product_lattice+"}}$\n"
            
    tt=tt[:-1]
    ax.text(0,0,tt,color='k',fontsize=FontSize,fontweight=Fontweight)
    
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

def generate_lattite_atom_positions(atoms_xyz_position,uvw2xyz,S=1,Q=np.eye(3),R=np.eye(3),shift=np.zeros(3),xlim=[],ylim=[],zlim=[]):
    S_range = list(range(-S,S+1)) 
    Points=[]
    eps=1e-5
    for atoms in atoms_xyz_position:
        Points1=[]
        for atom in atoms:
            triplets = list(itertools.product(S_range, repeat=3)) 
            triplets = np.array(triplets) 
            triplets = triplets.T
            for ii,lim in zip(range(0,3),[xlim,ylim,zlim]):
                if len(lim)>0:
                    idxs = np.where(triplets[ii,:]<lim[0])[0]
                    triplets = np.delete(triplets,idxs,1)
                    if (atom==[0.,0.,0.]).all():
                        idxs = np.where(triplets[ii,:]>lim[1])[0]
                        triplets = np.delete(triplets,idxs,1) 
                    elif atom[ii]==0:
                        idxs = np.where(triplets[ii,:]>lim[1])[0]
                        triplets = np.delete(triplets,idxs,1) 
                    else:
                        idxs = np.where(triplets[ii,:]>lim[1]-1)[0]
                        triplets = np.delete(triplets,idxs,1)

            points = R.dot(uvw2xyz.dot(triplets)+np.repeat([atom],triplets.shape[1],axis=0).T)
            #print(points.shape)
            points+=np.tile(np.array([shift]).T,(1,points.shape[1]))
            #print(points.shape)
#            for ii,lim,ei in zip(range(0,3),[xlim,ylim,zlim],np.eye(3)):
#                if len(lim)>0:
#                    aei = sum(uvw2xyz.dot(ei))
#                    idxs = np.where(points[ii,:]<aei*lim[0])[0]
#                    points = np.delete(points,idxs,1)
#                    idxs = np.where(points[ii,:]>aei*lim[1])[0]
#                    points = np.delete(points,idxs,1)
            points=np.matmul(Q,points)
                
        #    if len(Max)>0:
        #        sum1=np.sum((points>np.repeat([Max],triplets.shape[1],axis=0).T).astype(int),axis=0)
        #        sum1=sum1+np.sum((points<np.repeat([Min],triplets.shape[1],axis=0).T).astype(int),axis=0)
        #        idx = np.where(sum1==0)
        #        points=points[:,idx]
#            Max=[max(p) for p in points]
#            Min=[min(p) for p in points]
#            points = np.matmul(R,points)
            Points1.append(points)
        Points.append(Points1)
    return Points
def generate_lattice_vectors(Points,uvw2xyz,S=1,Q=np.eye(3),xlim=[],ylim=[],zlim=[],fitpoints=False,shift=np.zeros(3)):
    S_range = list(range(-S,S+1)) 
    atom=np.array([0., 0., 0.]);
    if fitpoints:
        allPoints=np.hstack([p for points in Points for p in points])+np.array([shift]).T
    for points in Points[0][0]*0.:
        triplets = list(itertools.product(S_range, repeat=3)) 
        triplets = np.array(triplets) 
        triplets = triplets.T
        for ii,lim in zip(range(0,3),[xlim,ylim,zlim]):
            
            if len(lim)>0:
                #print('ok')
                idxs = np.where(triplets[ii,:]<lim[0])[0]
                triplets = np.delete(triplets,idxs,1)
                if (atom==[0.,0.,0.]).all():
                    idxs = np.where(triplets[ii,:]>lim[1])[0]
                    #print(ii)
                    #print(triplets[ii,idxs])
                    triplets = np.delete(triplets,idxs,1)                        
        
        #print('==============')
        points = uvw2xyz.dot(triplets)+np.repeat([atom],triplets.shape[1],axis=0).T
        #points+=np.tile(np.array([shift]).T,(1,points.shape[1]))
        points2 = np.eye(3).dot(triplets)+np.repeat([atom],triplets.shape[1],axis=0).T
    Max=[max(p) for p in points]
    Min=[min(p) for p in points]
    
    LatticeVectors=[]        
    for point,point2 in zip(points.T,points2.T):   
        LatticeVectors1=[]
        for ii,lattice_vec,lim in zip([0,1,2],uvw2xyz.T,[xlim,ylim,zlim]):
            vec=np.array([lattice_vec*0,lattice_vec]).T+np.repeat([point],2,axis=0).T
            #print('------------------')
            #print(np.array([shift]).T.shape)
            #sum1=np.sum((vec>np.repeat([Max],2,axis=0).T).astype(int),axis=0)
            #sum1=sum1+np.sum((vec<np.repeat([Min],2,axis=0).T).astype(int),axis=0)
            #idx = np.where(sum1>0)[0]
            #print(lim)
            if len(lim)==0:
                lim=[0,point2[ii]+10]
            if point2[ii]<lim[1]:#True:#not len(idx)>0:
                vec=np.matmul(Q,vec)
                if fitpoints:
                    fit=False
                    for idx2 in [0,1]:
                        #print(np.min(np.linalg.norm(allPoints-np.array([vec[:,idx2]]).T,axis=0)))
                        #print(vec)
                        if np.min(np.linalg.norm(allPoints-np.array([vec[:,idx2]]).T,axis=0))<1e-1:
                            fit=True
                            
                    #fit=True
                    if fit:
                        LatticeVectors1.append(vec)
                    #else:
                        #print(vec)
                        #print('======================================')
                        #print(allPoints)
                        #print('======================================')
                else:
                    LatticeVectors1.append(vec)
                #print(vec)
                #print('===============')
        LatticeVectors.append(LatticeVectors1)
        


    return LatticeVectors
    
def plot_lattice(Points,LatticeVectors,ax=None,colors=['r','b','g'],edgecolors=['r','b','g'],salpha=1.,lalpha=1.,gridcolor=[0.5,0.5,0.5],Q=np.eye(3),\
                 shift=np.zeros(3),atoms=True,linewidth=1,move=np.zeros(3),normal=np.array([0,0,0]),halfspace='upper',s=200,plot=True):
    #colors=['r','b','g']
    #normal=np.array([1,1,1.5])
    
    halfscp=False
    
    pplot=True
    if not (normal==np.array([0,0,0])).all():
        halfscp=True
        normal=normal/np.sqrt(normal.dot(normal))
    if ax==None and plot:
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho') 
    elif not plot:
        ax=[]
        fig=[]
    else:
        fig=[]
    eps=1e-1
    #S = 1
    if atoms:
        PointsNew=[]
        for Points1,col,ecol in zip(Points,colors,edgecolors):
            PointNew=[]
            for points in Points1:
                pointnew=[]
                points=np.matmul(np.eye(3),points)+np.repeat(np.array([shift]).T,points.shape[1],1)
                point_proj_n = normal.dot(points)
                if halfspace=='lower':
                    idxs = np.where((point_proj_n)<=(eps))[0]
                else:
                    idxs = np.where((point_proj_n)>(-eps))[0]
                #print(len(idxs))    
                if len(idxs)>0:
                    points=Q.dot(points)
                    PointNew.append(points[:,idxs])
                    if plot:
                        ax.scatter(points[0,idxs]+move[0], points[1,idxs]+move[1], points[2,idxs]+move[2], s = s,color=col,edgecolors=ecol,alpha=salpha,linewidths=2) 
                #plt.show()
            PointsNew.append(PointNew)
        #PointsNew.append(PointNew)
    LatticeVectorsNew=[]
    for LatticeVectors1 in LatticeVectors:     
        LatticeVectorNew=[]
        for vec in LatticeVectors1:
            pplot=True

            point1=vec[:,0]
            point2=vec[:,1]
            #print(vec)

            if halfscp:
                point1_proj_n = normal.dot(point1)
                point2_proj_n = normal.dot(point2)
                point_proj_n = normal.dot(vec)
                #print(point_proj_n)
                if halfspace=='lower':
                    idxs = np.where((point_proj_n)<=(eps))[0]
                else:
                    idxs = np.where((point_proj_n)>(-eps))[0]
                #print(len(idxs))    
                if len(idxs)==0:
                    pplot=False
                elif len(idxs)==1:
                    inp=plane_line_intersection(normal,shift,vec[:,0],vec[:,1])
                    vec=np.vstack((vec[:,idxs[0]],inp)).T
            
            if pplot:               
                vec=np.matmul(Q,vec)+np.repeat(np.array([shift]).T,vec.shape[1],1)+np.repeat(np.array([move]).T,vec.shape[1],1)
                LatticeVectorNew.append(vec)
#                ax.plot(vec[0,:]+move[0], vec[1,:]+move[1], vec[2,:]+move[2],color=gridcolor,alpha=lalpha,linewidth=linewidth)
                if plot:
                    ax.plot(vec[0,:], vec[1,:], vec[2,:],color=gridcolor,alpha=lalpha,linewidth=linewidth)
        LatticeVectorsNew.append(LatticeVectorNew)

#def plane_line_intersection(n,V0,P0,P1):
    # n: normal vector of the Plane 
    # V0: any point that belongs to the Plane 
    # P0: end point 1 of the segment P0P1
    # P1:  end point 2 of the segment P0P1
            
#    normal=normal/np.sqrt(normal.dot(normal))
#    PointsOut=[]    
#    for Points1 in LatticePoints:
#        PointsOut1=[]
#        for points in Points1:   
#            #point_proj_n = normal.dot(points/np.sqrt(np.sum(points**2,axis=0)))
#            #print(points[:,0]+shift)
#            
#            point_proj_n = normal.dot(points)
#            if side=='bottom':
#                idxs = np.where((point_proj_n-shift)<(eps))[0]
#            else:
#                idxs = np.where((point_proj_n-shift)>(-eps))[0]
#            #print(point_proj_n)
#            PointsOut1.append(points[:,idxs])
#        PointsOut.append(PointsOut1)
    #    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
    if plot:  
        ax.axis('auto')      
    
    if halfscp:
        if atoms:
            return fig,ax,LatticeVectorsNew,PointsNew
        else:
            return fig,ax,LatticeVectorsNew
    else:
        return fig,ax
def plot_lattice_proj(LatticeVectors,normalproj,verticalproj, ax=None, linewidth=2,color='b',eps=1e-1,Q=np.eye(3),Qprojr=np.eye(2),
                      shift=np.zeros(3),shiftproj=0,move=np.zeros(3),normal=np.array([0,0,0]),shifthalfspace=np.zeros(3),
                      halfspace='upper',shiftplot=np.array([0,0]),out=False):
    if not isinstance(normalproj, np.ndarray):
        normalproj=np.array(normalproj);
    normalproj=normalproj/np.sqrt(normalproj.dot(normalproj))
    if not isinstance(verticalproj, np.ndarray):
        verticalproj=np.array(verticalproj);
    verticalproj=verticalproj/np.sqrt(verticalproj.dot(verticalproj))
    halfscp=True
    pplot=True
    if not isinstance(normal, np.ndarray):
        normal=np.array(normal);
    if not (normal==np.array([0,0,0])).all():
        halfscp=True
        normal=normal/np.sqrt(normal.dot(normal))

    if ax==None:
        fig = plt.figure() 
        ax = fig.add_subplot(111) 
    else:
        fig=[]

    
    
    horizontalproj=np.cross(verticalproj,normalproj)
    pointOut=[]
    pointOutproj=[]
    for LatticeVectors1 in LatticeVectors:  
        pout=[]
        poutp=[]
        for vec in LatticeVectors1:
            #print(vec)
            pplot=True
            point_proj_n=normalproj.dot(vec)
            #print(abs((point_proj_n-shiftproj)))
            idxs = np.where(abs((point_proj_n-shiftproj))<=(eps))[0]
            #print(abs((point_proj_n-shiftproj)))
            #print(len(idxs))
            if len(idxs)==0:
                pplot=False
            #elif len(idxs)==1:
            #    inp=plane_line_intersection(normal,shifthalfspace,vec[:,0],vec[:,1])
            #    vec=np.vstack((vec[:,idxs[0]],inp)).T
            #    print(vec)

            if halfscp:
                point_proj_n = normal.dot(vec)
                #print(point_proj_n)
                if halfspace=='lower':
                    idxs = np.where(((point_proj_n))<=(eps))[0]
                else:
                    idxs = np.where((point_proj_n)>(-eps))[0]
                    #idxs = np.where(abs((point_proj_n))<=(eps))[0]
                #print(len(idxs))    
                if len(idxs)==0:
                    pplot=False
                elif len(idxs)==1:
                    inp=plane_line_intersection(normal,shifthalfspace,vec[:,0],vec[:,1])
                    vec=np.vstack((vec[:,idxs[0]],inp)).T
            
            if pplot:
                #print(vec)
                point = np.matmul(Q,vec)+np.repeat(np.array([shift]).T,vec.shape[1],1)
                point[0,:]=point[0,:]+move[0]
                point[1,:]=point[1,:]+move[1]
                point[2,:]=point[2,:]+move[2]
                
                point=np.array(point);
                point_proj_x = horizontalproj.dot(point)
                point_proj_y = verticalproj.dot(point)
                pv = Qprojr.dot(np.vstack((point_proj_x,point_proj_y)))
                #ax.plot(point_proj_x,point_proj_y,color=color,linewidth=linewidth)
                ax.plot(pv[0,:]+shiftplot[0],pv[1,:]+shiftplot[1],color=color,linewidth=linewidth)
                pout.append(point)
                poutp.append([pv[0,:]+shiftplot[0],pv[1,:]+shiftplot[1]])
        pointOut.append(pout)
        pointOutproj.append(poutp)
                
    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
 
    #plt.show()       
    if out:
        return fig,ax,pointOut,pointOutproj,horizontalproj,verticalproj
    else:                 
        return fig,ax

def plot_points_proj(Points,normalproj,verticalproj, ax=None, marker="o",markersize=10, color='b',Q=np.eye(3),Qprojr=np.eye(2),shift=np.zeros(3),move=np.zeros(3)):
    if not isinstance(normalproj, np.ndarray):
        normalproj=np.array(normalproj);
    normalproj=normalproj/np.sqrt(normalproj.dot(normalproj))
    if not isinstance(verticalproj, np.ndarray):
        verticalproj=np.array(verticalproj);
    verticalproj=verticalproj/np.sqrt(verticalproj.dot(verticalproj))

    if ax==None:
        fig = plt.figure() 
        ax = fig.add_subplot(111) 
    else:
        fig=[]

    eps=1e-1
    
    horizontalproj=np.cross(verticalproj,normalproj)
    points_proj=[]
    for point in Points:   
        #print(point)
        point = Q.dot(point)+shift + move
        
        point_proj_x = horizontalproj.dot(point)
        point_proj_y = verticalproj.dot(point)
        point_proj_n= normalproj.dot(point)
        pv = Qprojr.dot([point_proj_x,point_proj_y])
        points_proj.append([point_proj_x,point_proj_y,point_proj_n])
        #ax.plot(point_proj_x,point_proj_y,color=color,linewidth=linewidth)
        ax.plot(pv[0],pv[1],color=color,marker=marker,linestyle='',markersize=markersize)
                
    #ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
 
    #plt.show()                        
    return fig,ax,points_proj

def select_atomic_plane(LatticePoints,normal,eps=1e-1,shift=0.,eps2=None):
    if not isinstance(normal, np.ndarray):
        normal=np.array(normal);
    normal=normal/np.sqrt(normal.dot(normal))
    PointsOut=[]    
    for Points1 in LatticePoints:
        PointsOut1=[]
        #print(Points1)
        for points in Points1:   
            #point_proj_n = normal.dot(points/np.sqrt(np.sum(points**2,axis=0)))
            point_proj_n = normal.dot(points)
            
            idxs = np.where(abs(point_proj_n-shift)<(eps))[0]
            #print(point_proj_n)
            if len(idxs)>0:
                PointsOut1.append(points[:,idxs])
        PointsOut.append(PointsOut1)
    
    return PointsOut
def get_interface2d(pointOutproj,normal,horizontalproj,verticalproj):
    vertices=[]
    for points in pointOutproj:
        for p in points:
            for px,py in zip(p[0],p[1]):
                vertices.append([px,py])
    vertices=np.array(vertices)
    hull = ConvexHull(vertices)
    verts=vertices[hull.vertices,:].T
    projnormal=np.array([horizontalproj.dot(normal),verticalproj.dot(normal)])
    twpoints=verts[:,np.abs(projnormal.dot(verts))<1e-10]
    return twpoints
def select_plane(LatticeVectors,normal,eps=1e-1,shift=0.,Q=np.eye(3)):
    if not isinstance(normal, np.ndarray):
        normal=np.array(normal);
    normal=normal/np.sqrt(normal.dot(normal))
    normal=Q.dot(normal)
    PointsOut=[]    
    for LatticeVectors1 in LatticeVectors:        
        for vec1 in LatticeVectors1:
            vec=Q.dot(vec1)
            for points in vec.T:   
                #point_proj_n = normal.dot(points/np.sqrt(np.sum(points**2,axis=0)))
                point_proj_n = normal.dot(points)
                
                idxs = np.where(abs(point_proj_n-shift)<(eps))[0]
                #print(point_proj_n)
                if len(idxs)>0:
                    isin=False
                    for PointOut in PointsOut:
                        if (PointOut==points).all():
                            isin=True
                            break
                    if not isin:
                        if not np.isnan(points).any():
                            PointsOut.append(points)
                        #else:
                        #    print(points)
            
    for LatticeVectors1 in LatticeVectors:        
        for vec1 in LatticeVectors1:
            vec=Q.dot(vec1)
#            point1_proj_n = normal.dot(point1)
#            point2_proj_n = normal.dot(point2)
            point_proj_n = normal.dot(vec)-shift
            #print(point_proj_n)
            #print(point_proj_n)
            if np.sign(point_proj_n[0])==-1*np.sign(point_proj_n[1]):
                points=plane_line_intersection(normal,shift,vec[:,0],vec[:,1])
                if not np.isnan(points).any():
                    PointsOut.append(points)
                else:
                    PointsOut.append(list(vec[:,0]))
                    PointsOut.append(list(vec[:,1]))
                    #print(normal)
                    #print(vec)
                    #print(points)
                    
            elif abs(point_proj_n[0])<1e-10 and abs(point_proj_n[1])<1e-10:
                #print("ooo")
                PointsOut.append(list(vec[:,0]))
                PointsOut.append(list(vec[:,1]))
                

    
    return np.array(PointsOut).T

def generate_plane_vertices(PlanePoints,normal,Q=np.eye(3),move=np.zeros(3)):
    verts = list(zip(PlanePoints[0,:], PlanePoints[1,:],PlanePoints[2,:]))
    normal=normal/np.sqrt(normal.dot(normal))
    v2=perpendicular_vector(normal)
    v3=np.cross(normal,v2)
    Qr=np.vstack((normal,v2,v3))
    Qr.dot(v3)
    x=[]
    y=[]
    z=[]
    for points in verts:   
        v4=Qr.dot(points)
#        v4=v4/np.linalg.norm(v4)
        x.append(v4[0])
        y.append(v4[1])
        z.append(v4[2])
    
    yc=np.mean(y)
    zc=np.mean(z)
    phase=[]
    for yi,zi in zip(y,z):
        phase.append(np.arctan2(zi-zc,yi-yc))
        
        
#    normal2=normal-shift*normal
#    normal2=normal2/np.sqrt(normal2.dot(normal2))
#    v2=np.array(verts[0])-shift*normal
#    v2=v2/np.linalg.norm(v2)
#    v3=np.cross(normal2,v2)
#    Qr=np.vstack((normal2,v2,v3))
#    Qr.dot(v3)
#    phase=[]
#    x=[]
#    y=[]
#    z=[]
#    for points in verts:   
#        v4=np.array(points)-shift*normal
##        v4=v4/np.linalg.norm(v4)
#        x.append(v4[0])
#        y.append(v4[1])
#        z.append(v4[2])
#    rc=np.array([np.mean(x),np.mean(y),np.mean(z)])    
#    for points in verts:   
#        #v4r=Qr.dot(v4)+       
#        v4=np.array(points)-shift*normal-rc
#        v4=v4/np.linalg.norm(v4)
#        phase.append(np.arctan2(Qr.dot(v4)[2],Qr.dot(v4)[1]))
    
    phase=np.unwrap(phase)
    np.argsort(phase)
    verts2=np.array(verts)[np.argsort(phase)[::1],:]
    #verts3=np.vstack((verts2,verts2[0,:]))
    x = list(np.matmul(Q,verts2.T)[0,:]+move[0])
    y = list(np.matmul(Q,verts2.T)[1,:]+move[1])
    z = list(np.matmul(Q,verts2.T)[2,:]+move[2])
    verts = [list(zip(x,y,z))]
    return verts,phase

def select_atomic_region(LatticePoints,normal,side='lower',eps=1e-1,shift=0.):
    if not isinstance(normal, np.ndarray):
        normal=np.array(normal);
    normal=normal/np.sqrt(normal.dot(normal))
    PointsOut=[]    
    for Points1 in LatticePoints:
        PointsOut1=[]
        for points in Points1:   
            #point_proj_n = normal.dot(points/np.sqrt(np.sum(points**2,axis=0)))
            #print(points[:,0]+shift)
            
            point_proj_n = normal.dot(points)
            
            if side=='lower':
                idxs = np.where((point_proj_n-shift)<(eps))[0]
            else:
                idxs = np.where((point_proj_n-shift)>(-eps))[0]
                #print(np.where((point_proj_n-shift)<(-eps))[0])
            #print(point_proj_n)
            PointsOut1.append(points[:,idxs])
        PointsOut.append(PointsOut1)
    
    return PointsOut
    
def plot_atomic_plane2D(LatticePoints,normal,vertical,ax=None,colors=['r','b','g'],edgecolors=['r','b','g'],plot=True,\
                      salpha=1.,lalpha=1.,gridcolor=[0.5,0.5,0.5],linewidths=[1,1,1],markersizes=[200,200,200],
                      Q=np.eye(3),xlim=[],ylim=[],out=False,zorder=1):
    if not isinstance(normal, np.ndarray):
        normal=np.array(normal);
    if not isinstance(vertical, np.ndarray):
        vertical=np.array(vertical);
    normal=normal/np.sqrt(normal.dot(normal))
    vertical=vertical/np.sqrt(vertical.dot(vertical))
    horizontal=np.cross(vertical,normal)
    if ax==None and plot:
        fig = plt.figure() 
        ax = fig.add_subplot(111) 
    else:
        fig=[]
    Pointsout=[]
    for Points1,col,ecol,linewidth,markersize in zip(LatticePoints,colors,edgecolors,linewidths,markersizes):
        pointout=[]
        for points in Points1:
            points2=Q.dot(points)
            point_proj_x = horizontal.dot(points2)
            point_proj_y = vertical.dot(points2)
            if len(xlim)>0:
                idxs = np.where(point_proj_x<xlim[0])[0]
                point_proj_x=np.delete(point_proj_x,idxs)
                point_proj_y=np.delete(point_proj_y,idxs)
                idxs = np.where(point_proj_x>xlim[1])[0]
                point_proj_x=np.delete(point_proj_x,idxs)
                point_proj_y=np.delete(point_proj_y,idxs)               
            if len(ylim)>0:
                idxs = np.where(point_proj_y<ylim[0])[0]
                point_proj_x=np.delete(point_proj_x,idxs)
                point_proj_y=np.delete(point_proj_y,idxs)
                idxs = np.where(point_proj_y>ylim[1])[0]
                point_proj_x=np.delete(point_proj_x,idxs)
                point_proj_y=np.delete(point_proj_y,idxs)  
            if plot:
                ax.scatter(point_proj_x, point_proj_y, color=col,edgecolors=ecol,alpha=salpha,linewidths=linewidth,s = markersize, zorder=zorder) 
            pointout.append([point_proj_x,point_proj_y])
        Pointsout.append(pointout)
    #plt.show()        
    ax.set_aspect('equal', 'datalim')
    if out:
        return fig,ax,Pointsout,horizontal,vertical
    else:
        return fig,ax
def get_twinning_plane_points(K1,Pointsout,horizontal,vertical):
    K1/=norm(K1)
    horizontal/=norm(horizontal)
    vertical/=norm(vertical)
    projnormal=np.array([horizontal.dot(K1),vertical.dot(K1)])
    allPoints=np.hstack([p for points in Pointsout for p in points])
    hull = ConvexHull(allPoints.T)
    verts=allPoints[:,hull.vertices]
    #print(verts)
    twpoints=verts[:,np.abs(projnormal.dot(verts))<1e-10]
    return twpoints
    
def plot_atomic_plane3D(LatticePoints,ax=None,colors=['r','b','g'],edgecolors=['r','b','g'],\
                      salpha=1.,lalpha=1.,gridcolor=[0.5,0.5,0.5],Q=np.eye(3)):
    if ax==None:
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho') 
    else:
        fig=[]

    for Points1,col,ecol in zip(LatticePoints,colors,edgecolors):
        for points in Points1:
            points=np.matmul(Q,points)
            ax.scatter(points[0,:], points[1,:], points[2,:], s = 200,color=col,edgecolors=ecol,alpha=salpha) 
    
    #plt.show()        
    ax.axis('auto')      
        
    return fig,ax
   
    
def plot_atomlattice2D(atoms_xyz_position,uvw2xyz,normal,vertical,S=1,R=np.eye(3),ax=None,colors=['r','b','g'],edgecolors=['r','b','g'],salpha=1.,lalpha=1.,gridcolor=[0.5,0.5,0.5]):
    #colors=['r','b','g']
    if not isinstance(normal, np.ndarray):
        normal=np.array(normal);
    if not isinstance(vertical, np.ndarray):
        vertical=np.array(vertical);
    normal=normal/np.sqrt(normal.dot(normal))
    vertical=vertical/np.sqrt(vertical.dot(vertical))
    horizontal=np.cross(vertical,normal)

    if ax==None:
        fig = plt.figure() 
        ax = fig.add_subplot(111) 
    else:
        fig=[]
    Points=[]
    Max=[]
    Min=[]
    #S = 2
    S_range = list(range(-1,S+1)) 
    S_range = list(range(-S,S+1)) 
    for atoms,col,ecol in zip(atoms_xyz_position,colors,edgecolors):
        for atom in atoms:
            #print('ok')
            triplets = list(itertools.product(S_range, repeat=3)) 
            triplets = np.array(triplets) 
            triplets = triplets.T
            points = uvw2xyz.dot(triplets)+np.repeat([atom],triplets.shape[1],axis=0).T
            Points.append(points)
            point_proj_x = horizontal.dot(points)
            point_proj_y = vertical.dot(points)
            
            #pn=points
            #sq = np.sqrt(np.sum(points**2,axis=0))
            #idxs = np.where(sq>0.0)[0]
            #pn[:,idxs] = pn[:,idxs]/sq[idxs]
            point_proj_n = normal.dot(points)
            #print(point_proj_n)
            idxs = np.where(abs(point_proj_n)<1e-3)[0]
#            ax3i.plot(point_proj_x,point_proj_y,'r',alpha=alpha[0])            
            #if abs(point_proj_n)<1e-5:
            ax.scatter(point_proj_x[idxs], point_proj_y[idxs], s = 200,color=col,edgecolors=ecol,alpha=salpha) 
            
#            else:
#                ax.scatter(point_proj_x, point_proj_y, s = 200,color=col,edgecolors=ecol,alpha=salpha) 

#            #plt.show()
    
#    S = S+1
#    S_range = list(range(-1,S+1)) 
#    for atom in atoms_xyz_position[0][0]*0.:
#        triplets = list(itertools.product(S_range, repeat=3)) 
#        triplets = np.array(triplets) 
#        triplets = triplets.T
#        points = uvw2xyz.dot(triplets)+np.repeat([atom],triplets.shape[1],axis=0).T
#    Max=[max(p) for p in points]
#    Min=[min(p) for p in points]
#    
#            
#    for point in points.T:        
#        for lattice_vec in uvw2xyz.T:
#            vec=np.array([lattice_vec*0,lattice_vec]).T+np.repeat([point],2,axis=0).T
#            sum1=np.sum((vec>np.repeat([Max],2,axis=0).T).astype(int),axis=0)
#            sum1=sum1+np.sum((vec<np.repeat([Min],2,axis=0).T).astype(int),axis=0)
#            idx = np.where(sum1>0)[0]
#            if not len(idx)>0:
#                vec=np.matmul(R,vec)
#                ax.plot(vec[0,:], vec[1,:], vec[2,:],color=gridcolor,alpha=lalpha)
#            
    #plt.show()        
    ax.set_aspect('equal', 'datalim')
    
    return fig,ax,Points,normal,vertical, horizontal


def an_between_vecs(v1,v2,deg=True,full2pi=False):
    #output 0-360 deg or -180-180 angle from v1 to v2
    #if an>0 counter clockwise otherwire clockwise
    v1=v1/np.sqrt(v1.dot(v1))
    v2=v2/np.sqrt(v2.dot(v2))
    cosa=v1.dot(v2)
    crossp=np.cross(v1,v2)
    an=np.arccos(cosa)
    if np.cross(crossp,v1).dot(v2)<0:
        an*=-1
    if deg:
        an*=180./np.pi
    if full2pi and an<0:
        an+=360.
    return an
def habitplane_equation_solution(Uj,Ui,Qj,n,a,tol=1e-10):
    #Qj*Uj-Ui=nxa

    #habit plane according to Bhattacharya
    delta=a.T.dot(Ui).dot(inv(Ui.dot(Ui)-np.eye(3)).dot(n))
    #print(delta)
    eta=np.trace(Ui.dot(Ui))-np.linalg.det(Ui.dot(Ui))-2+norm(a)**2/2/delta
    #print(eta)
    lam=0.5*(1-np.sqrt(1+2/delta))
    #print('Volum fraction of Uj: {}'.format(lam))
    C=(Ui+lam*np.outer(n,a)).dot(Ui+lam*np.outer(a,n))
    #print(C)
    D,V = np.linalg.eig(C)
    Idxs = np.argsort(D)
        
    Lambda=D[Idxs]
    V = V[:,Idxs]

    if Lambda[0]<1 and abs(Lambda[1]-1)<tol and  Lambda[0]>1:
        HB=True
    else:
        HB=False
        
    if HB:
        habitpdata={}
        keys=['shear_angle','s','m_a','b_a','Q_a','m_m','b_m','Q_m']
        for key in keys:
            habitpdata[key]=[]
    
        for k in [-1,1]:
            hpd={}
            
            m1_a=(np.sqrt(Lambda[2])-np.sqrt(Lambda[0]))/np.sqrt(Lambda[2]-Lambda[0])*(-1*np.sqrt(1-Lambda[0])*V[:,0]+k*np.sqrt(Lambda[2]-1)*V[:,2]);
            rho =1* norm(m1_a)
            m1_a=1/rho*m1_a
            b1_a=rho*(np.sqrt(Lambda[2]*(1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]+k*np.sqrt(Lambda[0]*(Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]);
            b1_a/=norm(b1_a)
            Q1_a=(np.eye(3)+np.outer(b1_a,m1_a)).dot(inv(lam*Qj.dot(Uj)+(1-lam)*Ui))

            # hpd['m_a']=m1_a#np.linalg.inv(Ui).dot(n1_a)
            # hpd['b_a']=b1_a
            # hpd['Q_a']=Q_a
            # hpd['s']=s1
            # hpd['shear_angle']=np.arctan(s1/2)*2


        # k=-1
        # m2_a=(np.sqrt(Lambda[2])-np.sqrt(Lambda[0]))/np.sqrt(Lambda[2]-Lambda[0])*(-1*np.sqrt(1-Lambda[0])*V[:,0]+k*np.sqrt(Lambda[2]-1)*V[:,2]);
        # rho =1* norm(m2_a)
        # m2_a=1/rho*m2_a
        # b2_a=rho*(np.sqrt(Lambda[2]*(1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]+k*np.sqrt(Lambda[0]*(Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]);
        # b2_a/=norm(b2_a)
        # Q2_a=(np.eye(3)+np.outer(b1_a,m1_a)).dot(inv(lam*Qij.dot(Uj)+(1-lam)*Ui))
        #         twind['s']=s1
        #         twind['shear_angle']=np.arctan(s1/2)*2

def twinnedhabitplane(Ui,Uj,Qij,a1,n1,hbplanes=[],addondata={},method='bhata'):        
#    if len(hbplanes.keys())==0:
#        hbplanes={}
#        titles='m_a b_a F_a Q_a Lambda epsilon gamma s alpha Li Lj var_i var_j var_i2 var_j2'
#        for key in titles.split():
#            hbplanes[key]=[]
    titles='m_a b_a b_an F_a U_a Q_a Lambda epsilon gamma s alpha Li Lj sgn'    
    if method=='bhata':
        #solution to :
        #1. QijUj - Ui =a1 x n1
        #2. Q_a(Lambda*QijUj+(1-Lambda)Ui)= I + b_a x m_a
        #Rotation of the habit plane with respect to variant i Q_a*Qi^-1 (Ui=Qi^-1*Fi)
        SOL=False
        #habit plane according to Bhattacharya: Microstructure of martensite page 113
        delta=a1.T.dot(Ui).dot(inv(Ui.dot(Ui)-np.eye(3)).dot(n1))
        eta=np.trace(Ui.dot(Ui))-np.linalg.det(Ui.dot(Ui))-2+norm(a1)**2/2/delta
        if delta<=-2 and eta >=0:
            SOL=True
            f=0.5*(1-np.sqrt(1+2/delta))
            for lam in [f,1-f]:
            #print('Volum fraction of Ui: {}, Uj: {}'.format(1-lam,lam))
                C=(Ui+lam*np.outer(n1,a1)).dot(Ui+lam*np.outer(a1,n1))
                D,V = np.linalg.eig(C)
                Idxs = np.argsort(D)  
                #print(D)
                #print(V)
                Lambda=D[Idxs]
                V = V[:,Idxs]
                #print(Lambda)
                #print(V)
                
                for kk in [-1,1]:
                    m_a=(np.sqrt(Lambda[2])-np.sqrt(Lambda[0]))/np.sqrt(Lambda[2]-Lambda[0])*(-1*np.sqrt(1-Lambda[0])*V[:,0]+kk*np.sqrt(Lambda[2]-1)*V[:,2]);
                    rho =1* norm(m_a)
                    m_a=1/rho*m_a
                    b_a=rho*(np.sqrt(Lambda[2]*(1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]+kk*np.sqrt(Lambda[0]*(Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]);
                    Q_a=(np.eye(3)+np.outer(b_a,m_a)).dot(inv(lam*Qij.dot(Uj)+(1-lam)*Ui))
                    F_a=Q_a.dot(lam*Qij.dot(Uj)+(1-lam)*Ui)
                    U_a=Q_a.dot(lam*Qij.dot(Uj)+(1-lam)*Ui)
                    #F=np.eye(3)+np.outer(b_a,m_a)
                    #F=Ainv=np.eye(3)+gamma*np.outer(s,m_a)+alpha*np.outer(m_a,m_a)

                    b_an=b_a/norm(b_a)
                    epsilon=(Lambda[2]-Lambda[0])/2
                    gamma=epsilon*norm(b_an-(b_an.dot(m_a))*m_a)
                    s=(b_an-(b_an.dot(m_a))*m_a)/norm(b_an-(b_an.dot(m_a))*m_a)
                    alpha=epsilon*b_an.dot(m_a)
                    #b_a/=norm(b_a)
                    #Li=1-f
                    #Lj=f
                    Li=1-lam
                    Lj=lam
                    hbplane={}
                    sgn=kk
                    for key in titles.split():#hbplanes.keys():
                        exec('hbplane["{}"]={}'.format(key,key))
                        #exec('hbplane["{}"].append({})'.format(key,key))
                    for key in addondata.keys():
                        hbplane[key]=addondata[key]
                    hbplanes.append(hbplane)
    elif  method=='Yongmei':
        #habit plane according to Yongmei https://www.sciencedirect.com/science/article/pii/S1359645402001234
        SOL=False
        x = sympy.Symbol('x')
        
        #Asw=(1-x)*Ui+x*Qij.dot(Uj)
        Asw=(x)*Ui+(1-x)*Qij.dot(Uj)
        fx=sympy.Matrix(Asw.T.dot(Asw)-np.eye(3)).det()
        all_terms=sympy.Poly(fx, x).all_terms()
        fx=sum(x**(len(all_terms)-n-1) * term[1] for n,term in enumerate(all_terms) if abs(term[1])>1e-10)
        sol=np.roots(fx.as_poly().coeffs())
        if np.isreal(sol[0]) and np.isreal(sol[1]):
            SOL=True
            for soli in sol:
                Aswi=np.array(sympy.Matrix(Asw).subs(x,soli)).astype(np.float64)
                F2=Aswi.T.dot(Aswi)
                D,V = np.linalg.eig(F2)
                Idxs = np.argsort(D)
                Lambda=D[Idxs]
                V = V[:,Idxs]
                for kk in [-1,1]:
                    m_a=np.sqrt((Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]-kk*np.sqrt((1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]
                    b_a=np.sqrt(Lambda[0]*(Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]+kk*np.sqrt(Lambda[2]*(1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]
                    b_an=np.sqrt(Lambda[0]*(Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]+kk*np.sqrt(Lambda[2]*(1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]
                    epsilon=(Lambda[2]-Lambda[0])/2
                    gamma=epsilon*norm(b_a-(b_a.dot(m_a))*m_a)
                    s=(b_a-(b_a.dot(m_a))*m_a)/norm(b_a-(b_a.dot(m_a))*m_a)
                    alpha=epsilon*b_a.dot(m_a)
                    F=V*0
                    for i in range(0,3):
                        F+=Lambda[i]**0.5*np.outer(V[:,i],V[:,i])
                    
                    Ainv=np.eye(3)+gamma*np.outer(s,m_a)+alpha*np.outer(m_a,m_a)
                    Q_a=Ainv.dot(np.linalg.inv(Aswi))#Rinv
                    F_a=np.eye(3)+np.outer(epsilon*b_a, m_a)
                    U_a=np.eye(3)+np.outer(epsilon*b_a, m_a)
                    Lj=1-soli
                    Li=soli
                    hbplane={}
                    sgn=kk
                    for key in titles.split():#hbplanes.keys():
                        exec('hbplane["{}"]={}'.format(key,key))
                        #exec('hbplanes["{}"].append({})'.format(key,key))
                    for key in addondata.keys():
                        hbplane[key]=addondata[key]
                    hbplanes.append(hbplane)
    
        

    return hbplanes
def twin_equation_solution_ini(Uj,Ui,Parent_uvw2xyz,Parent_hkl2xyz,Product_uvw2xyz,Product_hkl2xyz, 
                           Parent_uvw_2_Product_uvw_rot, Parent_uvw_2_Product_uvw,Parent_hkl_2_Product_hkl,tol=1e-10,miller='greaterthanone',printlambda=False,
                           Qj=None,Qi=None):
    from numpy import matmul
    from numpy import sqrt
    from numpy.linalg import inv
    from scipy.linalg import sqrtm
    from numpy.linalg import norm
    if miller=='greaterthanone':
        MIN=True
    else:
        MIN=False
    #Solves equation Q*Uj-Ui=axn
    #C=matmul(matmul(inv(Ui).T,Uj.T),matmul(Uj,inv(Ui)))
    
    #C=matmul(matmul(Uj,inv(Ui)).T,matmul(Uj,inv(Ui)))
    #print('ok')
    C=np.linalg.multi_dot([inv(Ui.T),Uj.T,Uj,inv(Ui)])
    
    
    D,V = np.linalg.eig(C)
    #V[:,1]=-1*V[:,1]
    
    matmul(C,V) - matmul(V,D*np.eye(3))
    
    #sorting according to eigenvalues Lambda[0]<Lambda[1]<Lambda[2]
    Idxs = np.argsort(D)
    
    Lambda=D[Idxs]
    #print(Lambda)
    if printlambda:
        print(Lambda)
    V = V[:,Idxs]
    
    twindata={}
    keys=['shear_angle','s','b_a', 'Rij_a', 'k','Type','n_a','a_a','Q_a','R_a','eta_a','K_a','eta_a_type','K_a_type','n_m','a_m','Q_m','R_m','eta_m','K_m','eta_m_type','K_m_type','C_m','C_a']
    for key in keys:
        twindata[key]=[]
        
    # shear_angle=[]
    # s=[]
    # n_a=[]
    # a_a=[]

    # Q_a=[]
    # R_a=[]
    # eta_a=[]
    # K_a=[]

    # n_m=[]
    # a_m=[]
    # Q_m=[]
    # R_m=[]
    # eta_m=[]
    # K_m=[]
    TWINDATA=[]
    #print(Lambda[2]-Lambda[1])
    #solution exists if Lambda[0]<1,Lambda[1]=1,Lambda[2]>1
    if Lambda[0]<1 and np.abs(Lambda[1]-1)<tol and Lambda[2]>1:
        
        #solution in the frame where Ui Uj are defioned
        #solutions are in terms of
        #twinning shear direction a
        #twinning plane normal n
        #rotation Q so that Q*Uj-Ui=axn
        #180 twinning rotation R so that  R1_a*Ui*R1_a.T=Uj, R1_a*Ui*R1_a.T=Uj

        for k in [-1,1]:
            twind={}
            
            n1_a=(sqrt(Lambda[2])-sqrt(Lambda[0]))/sqrt(Lambda[2]-Lambda[0])*(-1*sqrt(1-Lambda[0])*Ui.T.dot(V[:,0])+k*sqrt(Lambda[2]-1)*Ui.T.dot(V[:,2]));
            rho =1* norm(n1_a)
            n1_a=1/rho*n1_a
            a1_a=rho*(sqrt(Lambda[2]*(1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]+k*sqrt(Lambda[0]*(Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]);
            
            
            n1_a,c = flipvector(n1_a);
            
            #c=1
            #a1_a=c*np.linalg.inv(Ui).dot(a1_a);
            a1_a=c*a1_a;
            #twind['n_a']=n1_a
            #twind['a_a']=a1_a
            
            twind['n_a']=n1_a#np.linalg.inv(Ui).dot(n1_a)
            twind['a_a']=a1_a
            twind['k']=k
            
            
            
            #np.linalg.inv(Ui).dot(a1_a)
            
            
            
            eta1_a=vector2miller(np.linalg.inv(Parent_uvw2xyz).dot(np.linalg.inv(Ui).dot(a1_a)),MIN=MIN)
            #eta1_a=vector2miller(np.linalg.inv(Parent_uvw2xyz).dot(a1_a))
            twind['eta_a']=eta1_a
            twind['eta_a_type']=eta1_a
            K1_a=vector2miller(inv(Parent_hkl2xyz).dot(n1_a),MIN=MIN)
            #K1_a=vector2miller(inv(Parent_hkl2xyz).dot(np.linalg.inv(Ui).dot(n1_a)))
            #print(K1_a)
            twind['K_a']=K1_a
            twind['K_a_type']=K1_a

            
            eta1_m = vector2miller(Parent_uvw_2_Product_uvw.dot(eta1_a),MIN=MIN)
            
            K1_m=vector2miller(Parent_hkl_2_Product_hkl.dot(K1_a),MIN=MIN)
            
            
            n1_m = Product_hkl2xyz.dot(K1_m)
            a1_m = Product_uvw2xyz.dot(eta1_m)
            #n1_m = Parent_uvw_2_Product_uvw_rot.dot(n1_a)
            #a1_m = Parent_uvw_2_Product_uvw_rot.dot(a1_a)
            n1_m=n1_m/norm(n1_m)
            a1_m=a1_m/norm(a1_m)
            #n1_m,c=flipvector2negative(n1_m)
            #a1_m=c*a1_m
            c=1.0            
            #K1_m=vector2miller(inv(Product_hkl2xyz).dot(n1_m))
            #eta1_m=vector2miller(inv(Product_uvw2xyz).dot(a1_m))
            twind['eta_m']=c*eta1_m
            twind['K_m']=c*K1_m

            twind['eta_m_type']=c*eta1_m
            twind['K_m_type']=c*K1_m

            
            # n1_m = Parent_uvw_2_Product_uvw_rot.dot(twind['n_a'])
            # a1_m = Parent_uvw_2_Product_uvw_rot.dot(twind['a_a'])
            # n1_m=n1_m/norm(n1_m)
            # a1_m=a1_m/norm(a1_m)
            # n1_m,c=flipvector2negative(n1_m)
            # a1_m=c*a1_m
            
            # eta1_m = vector2miller(np.linalg.inv(Product_uvw2xyz).dot(a1_m))
            # twind['eta_m']=eta1_m
            # print(a1_m) 
            # print(eta1_m)
            # K1_m=vector2miller(np.linalg.inv(Product_hkl2xyz).dot(n1_m))
            # twind['K_m']=K1_m
            # print(n1_m) 
            # print(K1_m)
            # print('============================================================================')
            #print(Parent_hkl_2_Product_hkl.dot(K1_a))
            #print(twind['K_m'])
            
            #eta1_m = vector2miller(Parent_uvw_2_Product_uvw.dot(eta1_a))
            #twind['eta_m']=eta1_m
            #K1_m=vector2miller(Parent_hkl_2_Product_hkl.dot(K1_a))
            #twind['K_m']=K1_m
            
            #n1_m = Product_hkl2xyz.dot(K1_m)
            #a1_m = Parent_uvw_2_Product_uvw_rot.dot(a1_a)
            #n1_m=n1_m/norm(n1_m)
            #a1_m=a1_m/norm(a1_m)
            
            # eta1_m = vector2miller(Parent_uvw_2_Product_uvw.dot(eta1_a))
            # twind['eta_m']=eta1_m
            # K1_m=vector2miller(Parent_hkl_2_Product_hkl.dot(K1_a))
            # twind['K_m']=K1_m
            
            # n1_m = Product_hkl2xyz.dot(K1_m)
            # a1_m = Product_uvw2xyz.dot(eta1_m)
            # n1_m=n1_m/norm(n1_m)
            # a1_m=a1_m/norm(a1_m)


            twind['n_m']=n1_m
            twind['a_m']=a1_m
            Q1_a=matmul(np.outer(a1_a,n1_a)+Ui,inv(Uj))
            
            #(matmul(Q1_a,Uj)-Ui)-np.outer(a1_a,n1_a)
            R1_a=-np.eye(3)+2*np.outer(a1_a,a1_a)
            twind['Q_a']=Q1_a
            twind['R_a']=R1_a
            if Qj is not None and Qi is not None:
                twind['b_a']=Qi.dot(twind['a_a'])
                twind['Rij_a']=Qi.dot(twind['Q_a']).dot(Qj.T)
                #print(twind['Rij_a'])
            else:
                twind['b_a']=None
                twind['Rij_a']=None
                
            
            R1_m=Parent_uvw_2_Product_uvw_rot.dot(R1_a.dot(inv(Parent_uvw_2_Product_uvw_rot)))
            #np.outer(a1_m,a1_m).dot(v) - it is 0 if v is perpendicular to a1_m, otherwise it is a projection of the vector into a1_m
            #therefore if v is a vector along a1_m R1_m*v+I*v=v+v=2*np.outer(a1_m,a1_m)*v=2v
            R1_m=-np.eye(3)+2*np.outer(a1_m,a1_m)
            #or R1_m=np.eye(3)-2*np.outer(n1_m,n1_m)
            #if v is perpendicular to twinning plane thne R1_m*v=-v=>-v-I*v=-2*np.outer(n1_m,n1_m)*v=-2v
            #
            #change of lattice directions in the twin
            #if vector v lies in the twinning plane np.outer(n1_m,n1_m) is zeros and directions change signs C1_m*v=-I*v+0, the signes along n1 dop not change sign
            if (K1_m != np.round(K1_m)).any() and (eta1_m == np.round(eta1_m)).all():
                #for Type II twins - eta1 is preserved - rotation around eta1
                C1_m=-np.eye(3)+2*np.outer(a1_m,a1_m)
                Type='Type II'
                #print(K1_a)
            else:
                #for other K1 and eta1 is preserved - rotation 180 around K1
                #C1_m=np.eye(3)-2*np.outer(n1_m,n1_m)
                if (twind['eta_m'] == np.round(twind['eta_m'])).all():
                    Type='compound'
                else:
                    Type='Type I'
                C1_m=-np.eye(3)+2*np.outer(a1_m,a1_m)
            
            Q1_m=Parent_uvw_2_Product_uvw_rot.dot(Q1_a.dot(inv(Parent_uvw_2_Product_uvw_rot)))
            
            twind['Q_m']=Q1_m
            twind['R_m']=C1_m
            twind['C_m']=C1_m
            twind['C_a']=C1_m
            twind['Type']=Type
            s1=norm(a1_a)*norm(inv(Ui).dot(n1_a))
            twind['s']=s1
            twind['shear_angle']=np.arctan(s1/2)*2
            
            
            
            for key in keys:
                twindata[key].append(twind[key])
            # s.append(s1)
            
            # n_a.append(n1_a)
            # a_a.append(a1_a)
            # eta_a.append(eta1_a)
            # K_a.append(K1_a)
            # Q_a.append(Q1_a)
            # R_a.append(R1_a)
            
            # n_m.append(n1_m)
            # a_m.append(a1_m)
            # eta_m.append(eta1_m)
            # K_m.append(K1_m)
            # Q_m.append(Q1_m)
            # R_m.append(R1_m)
    
        KEYS=['shear_angle','s','b_a','Rij_a','k','Type','n1_a','a1_a','n2_a','a2_a','Q_a','R_a','eta1_a','K1_a','eta1_a_type',
              'K1_a_type','eta2_a','K2_a','eta2_a_type','K2_a_type','n1_m','a1_m','n2_m','a2_m','Q_m','R_m',
              'eta1_m','K1_m','eta2_m','K2_m','C_m','C_a']
        
        for ij in [[1,0],[0,1]]:
            TWIN={}
            for key in KEYS:
                if '1' in key:
                    TWIN[key]=twindata[key.replace('1','')][ij[0]]
                elif '2' in key:
                    TWIN[key]=twindata[key.replace('2','')][ij[1]]
                else:
                    TWIN[key]=twindata[key][ij[0]]
            TWINDATA.append(TWIN)   
        
    return TWINDATA


def twin_equation_solution(Uj,Ui,L_A,Lr_A,L_M,Lr_M, 
                           R_AM, Ci_d,Ci_p,tol=1e-10,miller='greaterthanone',printlambda=False,
                           Qj=None,Qi=None):
    #Uj transformation stretches of a possible twinned variant related to corresponding transformation gradient Fj=Qj*Uj
    #Ui transformation matrix of the matrix variant related to corresponding transformation gradient Fi=Qi*Ui
    #Qj rotational part of the transformation gradien Fj=Qj*Uj
    #Qi rotational part of the transformation gradien Fi=Qi*Ui
    #L_A/L_M matrix converting UVW->XYZ for parent phase, austenite/product phase, martensite
    #Lr_A/Lr_M matrix converting HKL->XYZ for parent phase, austenite/product phase, martensite
    #R_AM rotation matrix rotating the reference space of the austenite phase aligned with basal directions x=[1,0,0],y=[0,1,0]
    #into the space where x,y,z are aligned with austenite directions corresponding to martensite directions [1,0,0]m,[0,1,0]m,[0,0,1]m of variant i.
    #Basically R_AM.dot(inv(Ci_d))= diagonal matrix
    #Ci_d lattice correspondence for directions converting UVW of austenite with lattice corresponding UVW of martensite
    #Ci_p lattice correspondence for planes converting HKL of austenite with lattice corresponding HKL of martensite
    
    from numpy import matmul
    from numpy import sqrt
    from numpy.linalg import inv
    from scipy.linalg import sqrtm
    from numpy.linalg import norm
    if miller=='greaterthanone':
        MIN=True
    else:
        MIN=False
    #Solves equation Q*Uj-Ui=axn for stretches so the reference space Ui rotated Qi^(-1) with respect to real parent space
    #C=matmul(matmul(inv(Ui).T,Uj.T),matmul(Uj,inv(Ui)))
    
    #C=matmul(matmul(Uj,inv(Ui)).T,matmul(Uj,inv(Ui)))
    #print('ok')
    C=np.linalg.multi_dot([inv(Ui.T),Uj.T,Uj,inv(Ui)])
    
    
    D,V = np.linalg.eig(C)
    #V[:,1]=-1*V[:,1]
    
    matmul(C,V) - matmul(V,D*np.eye(3))
    
    #sorting according to eigenvalues Lambda[0]<Lambda[1]<Lambda[2]
    Idxs = np.argsort(D)
    
    Lambda=D[Idxs]
    #print(Lambda)
    if printlambda:
        print(Lambda)
    V = V[:,Idxs]
    
    twindata={}
    keys=['shear_angle','s','b_a', 'Rij_a', 'k','Type','n_a','a_a','Q_a','eta_a','K_a','eta_a_type','K_a_type','n_m','a_m','Q_m','R_m','eta_m','K_m','eta_m_type','K_m_type','C_m','C_a']
    for key in keys:
        twindata[key]=[]
        
    TWINDATA=[]
    #print(Lambda[2]-Lambda[1])
    #solution exists if Lambda[0]<1,Lambda[1]=1,Lambda[2]>1
    if Lambda[0]<1 and np.abs(Lambda[1]-1)<tol and Lambda[2]>1:
        
        #solution in the frame where Ui Uj are defioned
        #solutions are in terms of
        #twinning shear direction a
        #twinning plane normal n
        #rotation Q so that Q*Uj-Ui=axn
        #180 twinning rotation R so that  R1_a*Ui*R1_a.T=Uj, R1_a*Ui*R1_a.T=Uj

        for k in [-1,1]:
            twind={}
            
            n1_a=(sqrt(Lambda[2])-sqrt(Lambda[0]))/sqrt(Lambda[2]-Lambda[0])*(-1*sqrt(1-Lambda[0])*Ui.T.dot(V[:,0])+k*sqrt(Lambda[2]-1)*Ui.T.dot(V[:,2]));
            rho =1* norm(n1_a)
            n1_a=1/rho*n1_a
            a1_a=rho*(sqrt(Lambda[2]*(1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]+k*sqrt(Lambda[0]*(Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]);
            
            #n1_a is twinning normal undistorted by transformation!! It is real normal but only lattice corresponding to martensite twinning normal            
            n1_a,c = flipvector(n1_a);
            
            #a1_a is real twinning direction in parent phase but with subtracted 
            a1_a=c*a1_a;
            
            twind['n_a']=n1_a#this is twinning normal undistorted by transformation!! It is real normal lattice corresponding to martensite twinning normal 
            twind['a_a']=a1_a#This is real twinning direction rotated Qi^(-1) with respect to real parent space
            twind['k']=k#auxiliary
            
            
            
            #np.linalg.inv(Ui).dot(a1_a)
            eta1_a=vector2miller(np.linalg.inv(L_A).dot(np.linalg.inv(Ui).dot(a1_a)),MIN=MIN)##this is twinning direction undistorted by 
            #transformation!! See np.linalg.inv(Ui).dot(a1_a). It is real twinning shear direction lattice corresponding to martensite twinning direction 
            
            #eta1_a=vector2miller(np.linalg.inv(L_A).dot(a1_a))
            twind['eta_a']=eta1_a
            twind['eta_a_type']=eta1_a
            K1_a=vector2miller(inv(Lr_A).dot(n1_a),MIN=MIN)#Miller indexes of twinning normal undistorted by transformation!! It is HKL lattice corresponding to martensite twinning normal 
            twind['K_a']=K1_a
            twind['K_a_type']=K1_a

            #now for martensite
            #variant A based on lattice correspondence
            c=1.0            
            eta1_m = vector2miller(Ci_d.dot(eta1_a),MIN=MIN)#UVW of twinning direction in the martensite system          
            K1_m=vector2miller(Ci_p.dot(K1_a),MIN=MIN)#HKL of twinning direction in the martensite system           
            twind['eta_m']=c*eta1_m
            twind['K_m']=c*K1_m

            twind['eta_m_type']=c*eta1_m
            twind['K_m_type']=c*K1_m

            
            n1_m = Lr_M.dot(K1_m)#Real space of twinning normal in the martensite system
            a1_m = L_M.dot(eta1_m)#Real space twinning direction in the martensite system
            n1_m=n1_m/norm(n1_m)
            a1_m=a1_m/norm(a1_m)
            twind['n_m']=n1_m
            twind['a_m']=a1_m
            
            #shear of the twinning
            s1=norm(a1_a)*norm(inv(Ui).dot(n1_a))
            twind['s']=s1
            twind['shear_angle']=np.arctan(s1/2)*2


            #rigid body rotation necessary to apply to streteches of j variant to make twin with variant i
            #however there are other rotations Qj Qi already subtracted from deformation gradients Fj, Fi.
            #therefore complete rotation Rij_a is calculated below if Qj, Qi provided
            Q1_a=matmul(np.outer(a1_a,n1_a)+Ui,inv(Uj))
            
            #(matmul(Q1_a,Uj)-Ui)-np.outer(a1_a,n1_a)
            #R1_a=-np.eye(3)+2*np.outer(a1_a,a1_a)
            
            twind['Q_a']=Q1_a
            #twind['R_a']=R1_a
            if Qj is not None and Qi is not None: 
                #this is real twinning shear direction accounting for the subtracted rotation
                twind['b_a']=Qi.dot(twind['a_a'])
                twind['Rij_a']=Qi.dot(twind['Q_a']).dot(Qj.T)
                #print(twind['Rij_a'])
                #the other way to reach twinning elements in martensite
                #get real twinning normal (distorted by transformation)
                n_m=twind['n_a'].dot(np.linalg.inv(Qi.dot(Ui)))
                #rotate it into martensite frame, normalize, convert to HKL and make miller
                n_m=R_AM.dot(n_m)
                n_m=n_m/norm(n_m)
                twind['n_m']=n_m
                K_m=vector2miller(inv(Lr_M).dot(n_m),MIN=MIN)
                twind['K_m']=K_m
                twind['K_m_type']=K_m
                #eta_m corresponds to real twinning shear direction  
                a_m=Qi.dot(twind['a_a'])
                a_m=R_AM.dot(a_m)
                a_m/=norm(a_m)
                twind['a_m']=a_m
                eta_m=vector2miller(inv(L_M).dot(a_m),MIN=MIN)
                twind['eta_m']=eta_m
                twind['eta_m_type']=eta_m
            else:
                twind['b_a']=None
                twind['Rij_a']=None
                
            
            #change of lattice directions in the twin
            #this is passive and active, it is symmetric since it is 180 deg rotation
            #if vector v lies in the twinning plane np.outer(n1_m,n1_m) is zeros and directions change signs C1_m*v=-I*v+0, the signes along n1 dop not change sign
            if (K1_m != np.round(K1_m)).any() and (eta1_m == np.round(eta1_m)).all():
                #for Type II twins - eta1 is preserved - rotation around eta1
                C1_m=-np.eye(3)+2*np.outer(a1_m,a1_m)
                Type='Type II'
                #print(K1_a)
            else:
                #for other K1 and eta1 is preserved - rotation 180 around K1
                #C1_m=np.eye(3)-2*np.outer(n1_m,n1_m)
                if (twind['eta_m'] == np.round(twind['eta_m'])).all():
                    Type='compound'
                else:
                    Type='Type I'
                C1_m=-np.eye(3)+2*np.outer(a1_m,a1_m)
            
            Q1_m=R_AM.dot(Q1_a.dot(inv(R_AM)))
            
            twind['Q_m']=Q1_m
            twind['R_m']=C1_m
            twind['C_m']=C1_m
            twind['C_a']=C1_m
            twind['Type']=Type
            
            
            
            for key in keys:
                twindata[key].append(twind[key])
    
        KEYS=['shear_angle','s','b_a','Rij_a','k','Type','n1_a','a1_a','n2_a','a2_a','Q_a','eta1_a','K1_a','eta1_a_type',
              'K1_a_type','eta2_a','K2_a','eta2_a_type','K2_a_type','n1_m','a1_m','n2_m','a2_m','Q_m','R_m',
              'eta1_m','K1_m','eta2_m','K2_m','C_m','C_a']
        
        for ij in [[1,0],[0,1]]:
            TWIN={}
            for key in KEYS:
                if '1' in key:
                    TWIN[key]=twindata[key.replace('1','')][ij[0]]
                elif '2' in key:
                    TWIN[key]=twindata[key.replace('2','')][ij[1]]
                else:
                    TWIN[key]=twindata[key][ij[0]]
            TWINDATA.append(TWIN)   
        
    return TWINDATA



def def_gradient_stressfree(Cd,LA, LM,CId=None):

    T_MA = np.empty((3,3,Cd.shape[2]))
    T_AM = np.empty((3,3,Cd.shape[2]))
    for Variant in range(Cd.shape[2]):        
        T_MA[:,:,Variant] = Cd[:,:,Variant].dot(inv(sqrtm((Cd[:,:,Variant].T.dot(Cd[:,:,Variant])))))
        T_AM[:,:,Variant] = inv(T_MA[:,:,Variant])
        
        
    #Bain strain calculation
    #Deformation gradient
    F_AM = np.empty((3,3,Cd.shape[2]))
    
    #stretch - right Cauchy-Green deformation tensor
    U_AM = np.empty((3,3,Cd.shape[2]))
    #rotation
    Q_M = np.empty((3,3,Cd.shape[2]))
    for Variant in range(Cd.shape[2]):
        if CId is not None:
            F_AM[:,:,Variant]=T_MA[:,:,Variant].dot(LM.dot(CId[:,:,Variant].dot(inv(LA))))
            #print("Using Ci_d - correspondense for austenite directions")
        else:
            F_AM[:,:,Variant]=T_MA[:,:,Variant].dot(LM.dot(inv(Cd[:,:,Variant]).dot(inv(LA))))
            #print("Using C_d - correspondense for martensite directions")

    
        #stretch matrix
        U_AM[:,:,Variant] = sqrtm(F_AM[:,:,Variant].T.dot(F_AM[:,:,Variant]));
        
        #rotation matrix
        Q_M[:,:,Variant] = F_AM[:,:,Variant].dot(inv(U_AM[:,:,Variant]));
        

    return F_AM, U_AM, Q_M, T_MA, T_AM




def def_gradient_stressfree_ini(Product_uvw_2_Parent_uvw_all,parent_lattice_param, product_lattice_param,Ci_d=None):

    Parent_uvw2xyz_stressfree = np.array(lattice_vec(parent_lattice_param)).T

    
    Parent_uvw2xyz=Parent_uvw2xyz_stressfree
    


    parent_lattice_param_ortho=parent_lattice_param
    parent_lattice_param_ortho['alpha']=np.pi/2.
    parent_lattice_param_ortho['beta']=np.pi/2.
    parent_lattice_param_ortho['gamma']=np.pi/2.
    Parent_uvw2xyz_ortho = np.array(lattice_vec(parent_lattice_param_ortho)).T


    Product_uvw2xyz_stressfree = np.array(lattice_vec(product_lattice_param)).T;
    
    product_lattice_param_ortho=product_lattice_param.copy()
    product_lattice_param_ortho['alpha']=np.pi/2.
    product_lattice_param_ortho['beta']=np.pi/2.
    product_lattice_param_ortho['gamma']=np.pi/2.
    Product_uvw2xyz_ortho = np.array(lattice_vec(product_lattice_param_ortho)).T

    Product_uvw_2_Parent_uvw_all_norm = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    Parent_uvw_2_Product_uvw_all_norm = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    for Variant in range(Product_uvw_2_Parent_uvw_all.shape[2]):
        F_ortho = np.matmul(Product_uvw2xyz_ortho,\
               np.matmul(np.linalg.inv(Product_uvw_2_Parent_uvw_all[:,:,Variant]),np.linalg.inv(Parent_uvw2xyz_ortho)))
#        F_ortho = np.matmul(Product_uvw2xyz_ortho,\
#                            np.linalg.inv(np.matmul(Parent_uvw2xyz_ortho,Parent_uvw_2_Product_uvw_all[:,:,Variant])))                
#        F_ortho = np.matmul(np.matmul(Product_uvw2xyz_ortho,Product_uvw_2_Parent_uvw_all[:,:,Variant]),\
#                            np.linalg.inv(Parent_uvw2xyz_ortho))                
        #Stretch = sqrtm(np.matmul(F_ortho.T,F_ortho));
        #Parent_uvw_2_Product_uvw_all_norm[:,:,Variant] = np.matmul(F_ortho,inv(Stretch))
        #Product_uvw_2_Parent_uvw_all_norm[:,:,Variant] = np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:,:,Variant])

        PP=Product_uvw_2_Parent_uvw_all[:,:,Variant]
        Product_uvw_2_Parent_uvw_all_norm[:,:,Variant] = np.matmul(PP,inv(sqrtm(np.matmul(PP.T,PP))))
        Parent_uvw_2_Product_uvw_all_norm[:,:,Variant] = np.linalg.inv(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant])
        
        
    #Bain strain calculation
    #Deformation gradient
    Fv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    
    #stretch
    Uv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    #rotation
    Qv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    for Variant in range(Product_uvw_2_Parent_uvw_all.shape[2]):
        #real coords of Parent in the system aligned with Parent
        #Variant=5
        #In case of deformation Product_hkl2xyz changes
        #rotation of martensite compliance to to the system aligned with Parent
        A=Parent_uvw_2_Product_uvw_all_norm[:,:,Variant]
        #Parent_uvw_2_Product_uvw_all[:,:,Variant].dot(loaddir)
        #Product_ST_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,Product_ST)
        #deformation gradient
        Product_uvw2xyz = Product_uvw2xyz_stressfree#np.matmul(Product_strain +np.eye(3),Product_e_in_parent[:,:,Variant])
    
        
        #Deformation gradient    
#        Fv[:,:,Variant] = np.matmul(Product_uvw2xyz,\
#               np.matmul(np.linalg.inv(Product_uvw_2_Parent_uvw_all[:,:,Variant]),np.linalg.inv(Parent_uvw2xyz)))
#        Fv[:,:,Variant] = np.matmul(Product_uvw2xyz,\
#               np.linalg.inv(np.matmul(Parent_uvw2xyz,Product_uvw_2_Parent_uvw_all[:,:,Variant])))

        # Fv[:,:,Variant] = np.matmul(Product_uvw2xyz,\
        #        np.linalg.inv(np.matmul(Parent_uvw2xyz,Product_uvw_2_Parent_uvw_all[:,:,Variant])))
        Fv[:,:,Variant] = np.matmul(np.matmul(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant],Product_uvw2xyz),\
               np.linalg.inv(np.matmul(Parent_uvw2xyz,Product_uvw_2_Parent_uvw_all[:,:,Variant])))
        #print("Using C_d - correspondense for martensite directions")

        if Ci_d is not None:
            Fv[:,:,Variant] = Product_uvw_2_Parent_uvw_all_norm[:,:,Variant].dot(Product_uvw2xyz.dot(Ci_d[:,:,Variant])).dot(\
                   np.linalg.inv(Parent_uvw2xyz))
            #print("Using Ci_d - correspondense for austenite directions")



#        Fv[:,:,Variant] = np.matmul(np.matmul(Product_uvw2xyz,np.linalg.inv(Product_uvw_2_Parent_uvw_all[:,:,Variant])),\
#               np.linalg.inv(np.matmul(Parent_uvw2xyz,Product_uvw_2_Parent_uvw_all[:,:,Variant])))
#        Fv[:,:,Variant] = np.matmul(np.matmul(Product_uvw2xyz,np.linalg.inv(Product_uvw_2_Parent_uvw_all[:,:,Variant])),\
#               np.linalg.inv(np.matmul(Parent_uvw2xyz,np.eye(3))))
#        Fv[:,:,Variant] = np.matmul(np.matmul(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant],Product_uvw2xyz),\
#               np.linalg.inv(Parent_uvw2xyz))
    
    
        #stretch matrix
        Uv[:,:,Variant] = sqrtm(np.matmul(Fv[:,:,Variant].T,Fv[:,:,Variant]));
        
        #rotation matrix
        Qv[:,:,Variant] = np.matmul(Fv[:,:,Variant],inv(Uv[:,:,Variant]));
        

    return Fv, Uv, Qv, Product_uvw_2_Parent_uvw_all_norm, Parent_uvw_2_Product_uvw_all_norm, Parent_uvw2xyz, Product_uvw2xyz

def def_gradient(Cd,LA, LM,StressT=np.zeros((3,3)),STA=np.zeros((3,3,3,3)),STM=np.zeros((3,3,3,3)),CId=None):

 

    Parent_strain = np.einsum('ijkl,kl',STA,StressT)
    F_parent=Parent_strain+np.eye(3)
    LAStress=F_parent.dot(LA)
    
    T_MA = np.empty((3,3,Cd.shape[2]))
    T_AM = np.empty((3,3,Cd.shape[2]))
    for Variant in range(Cd.shape[2]):        
        T_MA[:,:,Variant] = Cd[:,:,Variant].dot(inv(sqrtm((Cd[:,:,Variant].T.dot(Cd[:,:,Variant])))))
        T_AM[:,:,Variant] = inv(T_MA[:,:,Variant])
        
        
    #Bain strain calculation
    #Deformation gradient
    F_AM = np.empty((3,3,Cd.shape[2]))
    
    #stretch - right Cauchy-Green deformation tensor
    U_AM = np.empty((3,3,Cd.shape[2]))
    #rotation
    Q_M = np.empty((3,3,Cd.shape[2]))
    
    Product_strain= np.empty((3,3,Cd.shape[2]))
    F_product= np.empty((3,3,Cd.shape[2]))
    Product_e_in_parent = np.empty((3,3,Cd.shape[2]))
    for Variant in range(Cd.shape[2]):        
        #transformation of the stress into martensite coordinate system
        StressT_in_Product=np.einsum('ia,jb,ab->ij',T_AM[:,:,Variant], T_AM[:,:,Variant],StressT)
        
        
        Product_strain[:,:,Variant] = np.einsum('ijkl,kl',STM,StressT_in_Product)
        #deformation gradient
        F_product[:,:,Variant] =Product_strain[:,:,Variant] +np.eye(3)
        LMStress = F_product[:,:,Variant].dot(LM)
        
        #print(LM)
        #Deformation gradient   
        Product_e_in_parent[:,:,Variant]  = T_MA[:,:,Variant].dot(LMStress)
        #print(inv(Cd[:,:,Variant]))
        if CId is not None:
            F_AM[:,:,Variant] = Product_e_in_parent[:,:,Variant] .dot(CId[:,:,Variant].dot(inv(LAStress)))
            #print("Using Ci_d - correspondense for austenite directions")
        else:
            F_AM[:,:,Variant] = Product_e_in_parent[:,:,Variant] .dot(inv(Cd[:,:,Variant]).dot(inv(LAStress)))
            #print("Using C_d - correspondense for martensite directions")
        #print(F_AM[:,:,Variant])
        #stretch matrix
        U_AM[:,:,Variant] = sqrtm(F_AM[:,:,Variant].T.dot(F_AM[:,:,Variant]));
        
        #rotation matrix
        Q_M[:,:,Variant] = F_AM[:,:,Variant].dot(inv(U_AM[:,:,Variant]));
        

    return F_AM, U_AM, Q_M, T_MA, T_AM, LAStress, LMStress, Parent_strain, Product_strain, F_parent, F_product



def def_gradient_ini(Product_uvw_2_Parent_uvw_all,parent_lattice_param, product_lattice_param,StressT=np.zeros((3,3)),Parent_ST=np.zeros((3,3,3,3)),Product_ST=np.zeros((3,3,3,3))):

    Parent_uvw2xyz_stressfree = np.array(lattice_vec(parent_lattice_param)).T

    Parent_strain = np.einsum('ijkl,kl',Parent_ST,StressT)
    
    F_parent=Parent_strain+np.eye(3)
    Parent_uvw2xyz=np.matmul(F_parent,Parent_uvw2xyz_stressfree)
    



    Product_uvw2xyz_stressfree = np.array(lattice_vec(product_lattice_param)).T;
    

    Product_uvw_2_Parent_uvw_all_norm = np.empty((3,3,12))
    Parent_uvw_2_Product_uvw_all_norm = np.empty((3,3,12))
    for Variant in range(Product_uvw_2_Parent_uvw_all.shape[2]):
        PP=Product_uvw_2_Parent_uvw_all[:,:,Variant]
        Product_uvw_2_Parent_uvw_all_norm[:,:,Variant] = np.matmul(PP,inv(sqrtm(np.matmul(PP.T,PP))))
        Parent_uvw_2_Product_uvw_all_norm[:,:,Variant] = np.linalg.inv(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant])
        #print(F_ortho)
        #print(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant] )        
        #print(np.matmul(PP,inv(sqrtm(np.matmul(PP.T,PP)))))
        #print("======================================================")
        #print(np.matmul((Stretch),inv(F_ortho)) )
        
        
    #Bain strain calculation
    #Deformation gradient
    Fv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    
    #stretch
    Uv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    #rotation
    Qv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    Product_e_in_parent = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    Parent_e = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    for Variant in range(Product_uvw_2_Parent_uvw_all.shape[2]):
        #real coords of Parent in the system aligned with Parent
        #Variant=5
        #In case of deformation Product_hkl2xyz changes
        #rotation of martensite compliance to to the system aligned with Parent
        A=Parent_uvw_2_Product_uvw_all_norm[:,:,Variant]
        #Parent_uvw_2_Product_uvw_all[:,:,Variant].dot(loaddir)
        #Product_ST_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,Product_ST)
        StressT_in_Product=np.einsum('ia,jb,ab->ij',A, A,StressT)
        
        
        Product_strain = np.einsum('ijkl,kl',Product_ST,StressT_in_Product)
        #deformation gradient
        F_product =Product_strain +np.eye(3)
        Product_uvw2xyz = np.matmul(F_product,Product_uvw2xyz_stressfree)#np.matmul(Product_strain +np.eye(3),Product_e_in_parent[:,:,Variant])
        #print(Product_uvw2xyz_stressfree)
        Product_e = np.matmul(Product_uvw2xyz,np.eye(3));
        Product_e_in_parent[:,:,Variant] =  np.matmul(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant],Product_e);
        
        Parent_e[:,:,Variant] =np.matmul(Parent_uvw2xyz,np.matmul(Product_uvw_2_Parent_uvw_all[:,:,Variant],np.eye(3)));
        #Deformation gradient    
        Fv[:,:,Variant] = np.matmul(Product_uvw2xyz,\
               np.matmul(np.linalg.inv(Product_uvw_2_Parent_uvw_all[:,:,Variant]),np.linalg.inv(Parent_uvw2xyz)))
        Fv[:,:,Variant] = np.matmul(Product_uvw2xyz,\
               np.linalg.inv(np.matmul(Parent_uvw2xyz,Product_uvw_2_Parent_uvw_all[:,:,Variant])))
    
        Fv[:,:,Variant] = np.matmul(np.matmul(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant],Product_uvw2xyz),\
               np.linalg.inv(np.matmul(Parent_uvw2xyz,Product_uvw_2_Parent_uvw_all[:,:,Variant])))
        #Calculation of deformation gradient
        Fv[:,:,Variant] = np.matmul(Product_e_in_parent[:,:,Variant],inv(Parent_e[:,:,Variant]));    
        #stretch matrix
        Uv[:,:,Variant] = sqrtm(np.matmul(Fv[:,:,Variant].T,Fv[:,:,Variant]));
        
        #rotation matrix
        Qv[:,:,Variant] = np.matmul(Fv[:,:,Variant],inv(Uv[:,:,Variant]));
        

    return Fv, Uv, Qv, Product_uvw_2_Parent_uvw_all_norm, Parent_uvw_2_Product_uvw_all_norm, Parent_uvw2xyz, Product_uvw2xyz, Parent_strain, Product_strain, F_parent, F_product




def def_gradient_ini2(Product_uvw_2_Parent_uvw_all,parent_lattice_param, product_lattice_param,StressT=np.zeros((3,3)),Parent_ST=np.zeros((3,3,3,3)),Product_ST=np.zeros((3,3,3,3))):

    Parent_uvw2xyz_stressfree = np.array(lattice_vec(parent_lattice_param)).T

    Parent_strain = np.einsum('ijkl,kl',Parent_ST,StressT)
    
    F_parent=Parent_strain+np.eye(3)
    Parent_uvw2xyz=np.matmul(F_parent,Parent_uvw2xyz_stressfree)
    


    parent_lattice_param_ortho=parent_lattice_param
    parent_lattice_param_ortho['alpha']=np.pi/2.
    parent_lattice_param_ortho['beta']=np.pi/2.
    parent_lattice_param_ortho['gamma']=np.pi/2.
    Parent_uvw2xyz_ortho = np.array(lattice_vec(parent_lattice_param_ortho)).T


    Product_uvw2xyz_stressfree = np.array(lattice_vec(product_lattice_param)).T;
    
    product_lattice_param_ortho=product_lattice_param
    product_lattice_param_ortho['alpha']=np.pi/2.
    product_lattice_param_ortho['beta']=np.pi/2.
    product_lattice_param_ortho['gamma']=np.pi/2.
    Product_uvw2xyz_ortho = np.array(lattice_vec(product_lattice_param_ortho)).T

    Product_uvw_2_Parent_uvw_all_norm = np.empty((3,3,12))
    Parent_uvw_2_Product_uvw_all_norm = np.empty((3,3,12))
    for Variant in range(Product_uvw_2_Parent_uvw_all.shape[2]):
#        F_ortho = np.matmul(Product_uvw2xyz_ortho,\
#               np.matmul(np.linalg.inv(Product_uvw_2_Parent_uvw_all[:,:,Variant]),np.linalg.inv(Parent_uvw2xyz_ortho)))
#        F_ortho = np.matmul(np.eye(3,3),\
#               np.matmul(np.linalg.inv(Product_uvw_2_Parent_uvw_all[:,:,Variant]),np.linalg.inv(np.eye(3,3))))
#        Stretch = sqrtm(np.matmul(F_ortho.T,F_ortho));
#        Parent_uvw_2_Product_uvw_all_norm[:,:,Variant] = np.matmul(F_ortho,inv(Stretch))
#        #print(np.matmul(F_ortho,inv(Stretch)))
#        #print(F_ortho.T)
#        Product_uvw_2_Parent_uvw_all_norm[:,:,Variant] = np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:,:,Variant])
        PP=Product_uvw_2_Parent_uvw_all[:,:,Variant]
        Product_uvw_2_Parent_uvw_all_norm[:,:,Variant] = np.matmul(PP,inv(sqrtm(np.matmul(PP.T,PP))))
        Parent_uvw_2_Product_uvw_all_norm[:,:,Variant] = np.linalg.inv(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant])
        #print(F_ortho)
        print(Product_uvw_2_Parent_uvw_all_norm[:,:,Variant] )        
        print(np.matmul(PP,inv(sqrtm(np.matmul(PP.T,PP)))))
        print("======================================================")
        #print(np.matmul((Stretch),inv(F_ortho)) )
        
        
    #Bain strain calculation
    #Deformation gradient
    Fv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    
    #stretch
    Uv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    #rotation
    Qv = np.empty((3,3,Product_uvw_2_Parent_uvw_all.shape[2]))
    for Variant in range(Product_uvw_2_Parent_uvw_all.shape[2]):
        #real coords of Parent in the system aligned with Parent
        #Variant=5
        #In case of deformation Product_hkl2xyz changes
        #rotation of martensite compliance to to the system aligned with Parent
        A=Parent_uvw_2_Product_uvw_all_norm[:,:,Variant]
        #Parent_uvw_2_Product_uvw_all[:,:,Variant].dot(loaddir)
        #Product_ST_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,Product_ST)
        StressT_in_Product=np.einsum('ia,jb,ab->ij',A, A,StressT)
        
        
        Product_strain = np.einsum('ijkl,kl',Product_ST,StressT_in_Product)
        #deformation gradient
        F_product =Product_strain +np.eye(3)
        Product_uvw2xyz = np.matmul(F_product,Product_uvw2xyz_stressfree)#np.matmul(Product_strain +np.eye(3),Product_e_in_parent[:,:,Variant])
    
        
        #Deformation gradient    
        Fv[:,:,Variant] = np.matmul(Product_uvw2xyz,\
               np.matmul(np.linalg.inv(Product_uvw_2_Parent_uvw_all[:,:,Variant]),np.linalg.inv(Parent_uvw2xyz)))
        Fv[:,:,Variant] = np.matmul(Product_uvw2xyz,\
               np.linalg.inv(np.matmul(Parent_uvw2xyz,Product_uvw_2_Parent_uvw_all[:,:,Variant])))
    
    
        #stretch matrix
        Uv[:,:,Variant] = sqrtm(np.matmul(Fv[:,:,Variant].T,Fv[:,:,Variant]));
        
        #rotation matrix
        Qv[:,:,Variant] = np.matmul(Fv[:,:,Variant],inv(Uv[:,:,Variant]));
        

    return Fv, Uv, Qv, Product_uvw_2_Parent_uvw_all_norm, Parent_uvw_2_Product_uvw_all_norm, Parent_uvw2xyz, Product_uvw2xyz, Parent_strain, Product_strain, F_parent, F_product








def niti_twinning(B2,B19p,Uv,Parent_uvw2xyz,Parent_hkl2xyz,Product_uvw2xyz,
                  Product_hkl2xyz,Parent_uvw_2_Product_uvw_all, Parent_hkl_2_Product_hkl_all,
                  Parent_uvw_2_Product_uvw_all_norm,SymOps,SymOpsR,miller='greaterthanone',Qv=None):
    L_A=Parent_uvw2xyz
    L_M=Product_uvw2xyz
    #Meteric tensor ||[x,y,x]||^2=[uvw]*G_A*[uvw]^T
    G_A=np.matmul(L_A.T,L_A)
    G_M=np.matmul(L_M.T,L_M)
    #Reciprocal Meteric tensor ||[x,y,x]||^2=d_hkl^2=1/[hkl]*Gr_A*[hkl]^T
    Gr_A = np.linalg.inv(G_A)
    Gr_M = np.linalg.inv(G_M)
    
    i=4
    j=5
    if Qv is None:
        twindata=twin_equation_solution(Uv[:,:,i],Uv[:,:,j],
                               Parent_uvw2xyz,Parent_hkl2xyz,
                               Product_uvw2xyz,Product_hkl2xyz, 
                               Parent_uvw_2_Product_uvw_all_norm[:,:,j], 
                               Parent_uvw_2_Product_uvw_all[:,:,j],
                               Parent_hkl_2_Product_hkl_all[:,:,j],tol=1e-10)
    else:
        twindata=twin_equation_solution(Uv[:,:,i],Uv[:,:,j],
                               Parent_uvw2xyz,Parent_hkl2xyz,
                               Product_uvw2xyz,Product_hkl2xyz, 
                               Parent_uvw_2_Product_uvw_all_norm[:,:,j], 
                               Parent_uvw_2_Product_uvw_all[:,:,j],
                               Parent_hkl_2_Product_hkl_all[:,:,j],tol=1e-10,miller=miller,Qj=Qv[:,:,i],Qi=Qv[:,:,j])
        #print(list(twindata[0].keys()))
        
    varnotation={0:'1',1:'1\'',2:'2',3:'2\'',4:'3',5:'3\'',6:'4',7:'4\'',8:'5',9:'5\'',10:'6',11:'6\''}
    twin_systems={}
    for twintype in ['100','001','Type I','Type II']:
        twin_systems[twintype]={}
        for key in list(twindata[0].keys()):
            twin_systems[twintype][key]=[]
        twin_systems[twintype]['vars']=[]
        twin_systems[twintype]['uvw2xyz_m']=[]
        twin_systems[twintype]['uvw2xyz_a']=[]
        twin_systems[twintype]['StrainTensor_m']=[]
        twin_systems[twintype]['F_m']=[]
        twin_systems[twintype]['F_a']=[]
        twin_systems[twintype]['Tension_m']=[]
        twin_systems[twintype]['Compression_m']=[]
        twin_systems[twintype]['StrainTensor_a']=[]
        twin_systems[twintype]['Tension_a']=[]
        twin_systems[twintype]['Compression_a']=[]
        twin_systems[twintype]['Variant pairs']=[]
        twin_systems[twintype]['Variant pairs 2']=[]
        twin_systems[twintype]['dm_a']=[]
        twin_systems[twintype]['dm_m']=[]
    #print(twin_systems[twintype])
    i=5
    j=4
    i=4
    j=5
    
    #Trnaformation twins
    Twins = np.zeros((Parent_hkl_2_Product_hkl_all.shape[2],Parent_hkl_2_Product_hkl_all.shape[2]), dtype=object)
    numtype2=0
    numtype1=0
    num100=0
    num001=0
    numall=0
    for j in range(0,Parent_hkl_2_Product_hkl_all.shape[2]): 
        for i in range(0,Parent_hkl_2_Product_hkl_all.shape[2]): 
            
            if i!=j:
                #print((i,j))
                #twin_equation_solution(Uj,Ui,...) - Solves equation Q*Uj-Ui=axn
                if Qv is None:
                    twindata=twin_equation_solution(Uv[:,:,i],Uv[:,:,j],
                                           Parent_uvw2xyz,Parent_hkl2xyz,
                                           Product_uvw2xyz,Product_hkl2xyz, 
                                           Parent_uvw_2_Product_uvw_all_norm[:,:,j], 
                                           Parent_uvw_2_Product_uvw_all[:,:,j],
                                           Parent_hkl_2_Product_hkl_all[:,:,j],tol=1e-10,miller=miller)
                else:
                    twindata=twin_equation_solution(Uv[:,:,i],Uv[:,:,j],
                                           Parent_uvw2xyz,Parent_hkl2xyz,
                                           Product_uvw2xyz,Product_hkl2xyz, 
                                           Parent_uvw_2_Product_uvw_all_norm[:,:,j], 
                                           Parent_uvw_2_Product_uvw_all[:,:,j],
                                           Parent_hkl_2_Product_hkl_all[:,:,j],tol=1e-10,miller=miller,Qj=Qv[:,:,i],Qi=Qv[:,:,j])

                    #print(twin_systems[twintype])
                #print(len(twindata))
                #print(i)
                if len(twindata)>0:

                    K1_arounddigit=30
                    eta1_arounddigit=30
                    arounddigit=0
                    for twin in  twindata:
                        #if twintype == '001':
                        #    print(list(twin.keys()))

                        twintype='no'
                        if (twin['eta1_m'] == np.round(twin['eta1_m'])).all():
                            uvw2xyz_m = np.matmul((2*np.outer(twin['a1_m'],twin['a1_m'])-np.eye(3)),Product_uvw2xyz)
                            uvw2xyz_a = np.matmul((2*np.outer(twin['a1_a'],twin['a1_a'])-np.eye(3)),np.matmul(np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:,:,i]),Product_uvw2xyz))
                            if (twin['K1_m'] == np.round(twin['K1_m'])).all():
                                Twins[i,j]='Compound'
                                if (np.abs(twin['K1_m'])-np.array([0,0,1])==0).all(): #or (K_m[0]-np.array([0,0,1])==0).all():
                                    twintype='001'
                                    num001+=1
                                    #print('001: {}'.format(num001))
                                elif (np.abs(twin['K1_m'])-np.array([1,0,0])==0).all():
                                    twintype='100'
                                    num100+=1
                                    #print('100: {}'.format(num100))
                                else:
                                    numall+=1
                                    print('noncategorized twin')
                                    print('K1_m: {}, eta1_m:{}'.format(twin['K1_m'],twin['eta1_m']))        
                                    print((twin['eta1_m'] == np.around(twin['eta1_m'],decimals=arounddigit)))
                                    print(np.around(twin['eta1_m'],decimals=arounddigit))
                            else:
                                #print(twin['eta1_m'])
                                Twins[i,j]='Type II'
                                twintype='Type II'
                                K1_arounddigit=1
                                numtype2+=1
                                #print('Type II: {}'.format(numtype2))
                                #print(twin['K1_a'])
                        elif (twin['K1_m'] == np.round(twin['K1_m'])).all():
                            uvw2xyz_m = np.matmul((-2*np.outer(twin['n1_m'],twin['n1_m'])+np.eye(3)),Product_uvw2xyz)
                            uvw2xyz_a = np.matmul((-2*np.outer(twin['n1_a'],twin['n1_a'])+np.eye(3)),np.matmul(np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:,:,i]),Product_uvw2xyz))
                            twintype='Type I'
                            Twins[i,j]='Type I'
                            eta1_arounddigit=1
                            numtype1+=1
                            #print('Type I: {}'.format(numtype1))
                            #if (twin['K1_a']==[0.,0.,1.0]).all():
                                #print('{}-{}'.format(i,j))
                        #else:
                            #numall+=1
                            #print('numall: {}'.format(numall))                        
                            
                        #if  twintype =='no':
                        #   numall+=1
                        #    print('numall: {}'.format(numall))                        
                        if twintype !='no':
                            #isin=False
                            #print(twin_systems[twintype])
                            #if twintype=='Type II':
                            #    numtype2+=1
                            #    print(numtype2)
                            twels=['K1_a_type','eta1_a_type','K2_a_type','eta2_a_type']
                            twels=['K1_m_type','eta1_m_type']
                            twels=['K1_a_type','eta1_a_type']
                            SymO=[SymOpsR,SymOps]
                            
                            ISIN=False
                            isin=[]
                            if len(twin_systems[twintype][twels[0]])>0:
                                isin=[]
                                for twi in range(0,len(twin_systems[twintype][twels[0]])):
                                    isin.append(False)
                                    alldiff=[]
                                    for twel,Sym in zip(twels,SymO):
                                        alldiff.append(0)
                                        for symops in Sym:
                                            twitwel = twin_systems[twintype][twel][twi]
                                            twintwel=twin[twel]                                            
                                            if 'K1' in twel:
                                                twitwel=np.around(twitwel,decimals=K1_arounddigit)
                                                twintwel=np.around(twintwel,decimals=K1_arounddigit)
                                            elif 'eta1' in twel:
                                                twitwel=np.around(twitwel,decimals=eta1_arounddigit)
                                                twintwel=np.around(twintwel,decimals=eta1_arounddigit)   

                                            if (abs(twitwel-symops.dot(twintwel))>1e-10).any():
                                                # if twintype=='Type I' and len(twin_systems[twintype][twels[0]])==1:
                                                #     print(twin_systems[twintype][twel][twi])
                                                #     print(symops.dot(twin[twel]))
                                                #     print('================================================================================')
                                                alldiff[-1]+=1
                                    if  alldiff[0]==len(SymO[0]) or alldiff[1]==len(SymO[0]):
                                        isin[-1]=False
                                    else:
                                        isin[-1]=True
                                    #K1in=[1 for K1Mi in [twin_systems[twintype]['K1_m'][twi]] if (twin['K1_m'] == K1Mi).all() or (-1*twin['K1_m'] == K1Mi).all()]
                                    #E1in=[1 for E1Mi in [twin_systems[twintype]['eta1_m'][twi]] if (twin['eta1_m'] == E1Mi).all() or (-1*twin['eta1_m'] == E1Mi).all()]
                                    #K1E1in = [1 for K1Mi,E1Mi in zip([twin_systems[twintype]['K1_m'][twi]],[twin_systems[twintype]['eta1_m'][twi]]) 
                                    #                                 if ((twin['K1_m'] == K1Mi).all() and (twin['eta1_m'] == E1Mi).all()) or 
                                    #                                 ((-1*twin['K1_m'] == K1Mi).all() and (-1*twin['eta1_m'] == E1Mi).all())] 
                                    #print(twin_systems[twintype]['K1_m'][twi])
                                    #print('=================')
                                    #if len(K1in)!=0 and len(E1in)!=0:
                                    #if len(K1E1in)!=0:
                                    #    isin[-1]=True
                                    #if twintype=='Type II':
                                    #    print(twin_systems[twintype]['eta1_m'][twi])
                                    #    print('================================================')
                            
                            keys2list=['Q_a','b_a','Rij_a','a1_a','n1_a','a2_a','n2_a','K1_a','eta1_a','eta2_a','K2_a','eta1_m','K1_m','eta2_m','K2_m']#,'eta1_a','K1_a']            
                            #keys2list=['Q_a','a1_a','n1_a','a2_a','n2_a']  
                            try:
                                idx=isin.index(True)
                                #if twintype=='Type II':
                                #    print(twin[twels[0]])
                                #    print(twin_systems[twintype][twels[0]][idx])
                                #    print(twin[twels[1]])
                                #    print(twin_systems[twintype][twels[1]][idx])
                                #    print('=============================================')
                                ISIN=True
                                #twin_systems[twintype]['Variant pairs'][idx].append((i,j))
                                #twin_systems[twintype]['Variant pairs 2'][idx].append((varnotation[i],varnotation[j]))
                                twin_systems[twintype]['Variant pairs'][idx].append((j,i))
                                twin_systems[twintype]['Variant pairs 2'][idx].append((varnotation[j],varnotation[i]))
                                
                                for key in keys2list:
                                    twin_systems[twintype][key][idx].append(twin[key])
                                #twin_systems[twintype]['Q_a'][idx].append(twin['Q_a'])
                                #twin_systems[twintype]['a1_a'][idx].append(twin['a1_a'])
                                #twin_systems[twintype]['n1_a'][idx].append(twin['n1_a'])
                                #twin_systems[twintype]['a2_a'][idx].append(twin['a2_a'])
                                #twin_systems[twintype]['n2_a'][idx].append(twin['n2_a'])
                                #twin_systems[twintype]['vars'].append((i,j))
                                twin_systems[twintype]['vars'].append((j,i))
                            except:
                                pass
                            #ISIN=False        
                            
                            if not ISIN:
                                for key in list(twin.keys()):                                    
                                    if key in keys2list:
                                        twin_systems[twintype][key].append([twin[key]])
                                    else:
                                        twin_systems[twintype][key].append(twin[key])
                                #twin_systems[twintype]['vars'].append((i,j))
                                twin_systems[twintype]['vars'].append((j,i))
                                twin_systems[twintype]['uvw2xyz_m'].append(uvw2xyz_m)
                                #twin_systems[twintype]['uvw2xyz_a'].append(uvw2xyz_a)
                                
                                n2eta1K1=np.cross(twin_systems[twintype]['a1_m'][-1],twin_systems[twintype]['n1_m'][-1]);
                                Qrot=np.vstack((twin_systems[twintype]['a1_m'][-1],n2eta1K1,twin_systems[twintype]['n1_m'][-1]))
                                Qrot.dot(twin_systems[twintype]['a1_m'][-1])
                                Qri=np.linalg.inv(Qrot)
                                StrainTensor=np.array([[0,0,twin_systems[twintype]['s'][-1]/2],[0,0,0],[twin_systems[twintype]['s'][-1]/2,0,0]])
                                DefGrad=np.array([[1,0,twin_systems[twintype]['s'][-1]],[0,1,0],[0,0,1]])
                                
                                twin_systems[twintype]['StrainTensor_m'].append(np.matmul(np.matmul(Qri,StrainTensor),Qrot))
                                twin_systems[twintype]['F_m'].append(np.matmul(np.matmul(Qri,DefGrad),Qrot))
                                #if twintype=='001':
                                #    print(np.matmul(np.matmul(Qri,DefGrad),Qrot))
                                #    print(Qrot)
                                D,V = np.linalg.eig(twin_systems[twintype]['StrainTensor_m'][-1])
                                Idxs = np.argsort(D)[::-1]
                                Lambda=D[Idxs]
                                V = V[:,Idxs]
                                #if twintype == '001':
                                #    print('ok')
                                twin_systems[twintype]['Tension_m'].append(V[:,0])
                                twin_systems[twintype]['Compression_m'].append(V[:,2])
                                #Qa=np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:,:,i])
                                #twin_systems[twintype]['StrainTensor_a'].append(Qa.dot(twin_systems[twintype]['StrainTensor_m'][-1]).dot(Qa.T))
                                #twin_systems[twintype]['Tension_a'].append(Qa.dot(twin_systems[twintype]['Tension_m'][-1]))
                                #twin_systems[twintype]['Compression_a'].append(Qa.dot(twin_systems[twintype]['Compression_m'][-1]))
                                
                                #twin_systems[twintype]['Variant pairs'].append([(i,j)])
                                #twin_systems[twintype]['Variant pairs 2'].append([(varnotation[i],varnotation[j])])
                                twin_systems[twintype]['Variant pairs'].append([(j,i)])
                                twin_systems[twintype]['Variant pairs 2'].append([(varnotation[j],varnotation[i])])
                                #print(twin_systems[twintype]['K1_m'][-1])
                                
                                
                                twin_systems[twintype]['dm_m'].append(get_twinning_dislocation(twin_systems[twintype]['K1_m'][-1][0],
                                                                                        twin_systems[twintype]['eta1_m'][-1][0],
                                                                                        twin_systems[twintype]['eta2_m'][-1][0],
                                                                                        L_M,G=G_M,Gr=Gr_M))
                           
                            twin_systems[twintype]['uvw2xyz_a'].append(uvw2xyz_a)
                            Qa=np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:,:,i])
                            twin_systems[twintype]['StrainTensor_a'].append(Qa.dot(twin_systems[twintype]['StrainTensor_m'][-1]).dot(Qa.T))
                            twin_systems[twintype]['F_a'].append(Qa.dot(twin_systems[twintype]['F_m'][-1]).dot(Qa.T))
                            twin_systems[twintype]['Tension_a'].append(Qa.dot(twin_systems[twintype]['Tension_m'][-1]))
                            twin_systems[twintype]['Compression_a'].append(Qa.dot(twin_systems[twintype]['Compression_m'][-1]))                           
                            twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1][0],
                                                                                     twin_systems[twintype]['eta1_a'][-1][0],
                                                                                     twin_systems[twintype]['eta2_a'][-1][0],
                                                                                     L_A,G=G_A,Gr=Gr_A))   
    
    
    #deformation twins in martensite
    #20-1
    
    twintype='20-1'
    twin_systems[twintype]={}
    for key in list(twindata[0].keys()):
        twin_systems[twintype][key]=[]
    twin_systems[twintype]['vars']=[]
    twin_systems[twintype]['uvw2xyz_m']=[]
    twin_systems[twintype]['uvw2xyz_a']=[]
    twin_systems[twintype]['StrainTensor_m']=[]
    twin_systems[twintype]['Tension_m']=[]
    twin_systems[twintype]['Compression_m']=[]
    twin_systems[twintype]['Variant pairs']=[]
    twin_systems[twintype]['dm_a']=[]
    twin_systems[twintype]['dm_m']=[]
 
    twin_systems[twintype]['eta1_m'].append(np.array([-1,0,-2]))
    twin_systems[twintype]['K1_m'].append(np.array([2,0,-1]))
    twin_systems[twintype]['eta2_m'].append(np.array([1,0,0]))
    twin_systems[twintype]['K2_m'].append(np.array([0,0,-1]))
    uvw2xyz=Product_uvw2xyz
    hkl2xyz=Product_hkl2xyz
    a1=uvw2xyz.dot(twin_systems[twintype]['eta1_m'][0])
    twin_systems[twintype]['a1_m'].append(a1/np.linalg.norm(a1))
    a2=uvw2xyz.dot(twin_systems[twintype]['eta2_m'][0])
    twin_systems[twintype]['a2_m'].append(a2/np.linalg.norm(a2))
    n1=hkl2xyz.dot(twin_systems[twintype]['K1_m'][0])
    twin_systems[twintype]['n1_m'].append(n1/np.linalg.norm(n1))
    n2=hkl2xyz.dot(twin_systems[twintype]['K2_m'][0])
    twin_systems[twintype]['n2_m'].append(n2/np.linalg.norm(n2))
    
    twin_systems[twintype]['uvw2xyz_m'].append(np.matmul((2*np.outer(twin_systems[twintype]['a1_m'][0],twin_systems[twintype]['a1_m'][0])-np.eye(3)),uvw2xyz))
    twin_systems[twintype]['R_m'].append(twin_systems[twintype]['uvw2xyz_m'][-1].dot(inv(uvw2xyz)))
    #C1_m=-np.eye(3)+2*np.outer(n1_m,n1_m)
    #twin_systems[twintype]['C_m'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_m'][0],twin_systems[twintype]['n1_m'][0]))
    twin_systems[twintype]['C_m'].append(2*np.outer(twin_systems[twintype]['a1_m'][0],twin_systems[twintype]['a1_m'][0])-np.eye(3))
    twin_systems[twintype]['shear_angle'].append(2*abs(np.pi/2-np.arccos((twin_systems[twintype]['uvw2xyz_m'][0].dot(twin_systems[twintype]['eta2_m'][0])
                                         /norm(twin_systems[twintype]['uvw2xyz_m'][0].dot(twin_systems[twintype]['eta2_m'][0]))).dot(
                                             twin_systems[twintype]['a1_m'][0]))))
    twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][0]/2)*2)
    
    
    
    n2eta1K1=np.cross(twin_systems[twintype]['a1_m'][-1],twin_systems[twintype]['n1_m'][-1]);
    Qrot=np.vstack((twin_systems[twintype]['a1_m'][-1],n2eta1K1,twin_systems[twintype]['n1_m'][-1]))
    Qrot.dot(twin_systems[twintype]['a1_m'][-1])
    Qri=np.linalg.inv(Qrot)
    StrainTensor=np.array([[0,0,twin_systems[twintype]['s'][-1]/2],[0,0,0],[twin_systems[twintype]['s'][-1]/2,0,0]])
    twin_systems[twintype]['StrainTensor_m'].append(np.matmul(np.matmul(Qri,StrainTensor),Qrot))
    
    D,V = np.linalg.eig(twin_systems[twintype]['StrainTensor_m'][-1])
    Idxs = np.argsort(D)
    Lambda=D[Idxs]
    V = V[:,Idxs]
    twin_systems[twintype]['Tension_m'].append(V[:,0])
    twin_systems[twintype]['Compression_m'].append(V[:,2])
    twin_systems[twintype]['dm_m'].append(get_twinning_dislocation(twin_systems[twintype]['K1_m'][-1],
                                                            twin_systems[twintype]['eta1_m'][-1],
                                                            twin_systems[twintype]['eta2_m'][-1],
                                                            L_M,G=G_M,Gr=Gr_M))
    
    
    for symhkl,symuvw in zip( B19p.reciprocal_symmetry_operations(),B19p.symmetry_operations()):
        K1=np.round(symhkl[0:3,0:3].dot(twin_systems[twintype]['K1_m'][0]),5)
        K2=np.round(symhkl[0:3,0:3].dot(twin_systems[twintype]['K2_m'][0]),5)
        eta1=np.round(symuvw[0:3,0:3].dot(twin_systems[twintype]['eta1_m'][0]),5)
        eta2=np.round(symuvw[0:3,0:3].dot(twin_systems[twintype]['eta2_m'][0]),5)
        isin=False
        for i in range(0,len(twin_systems[twintype]['eta1_m'])):
            if (twin_systems[twintype]['eta1_m'][i]==eta1).all():
                if (twin_systems[twintype]['eta2_m'][i]==eta2).all():
                    if (twin_systems[twintype]['K1_m'][i]==K1).all():
                        if (twin_systems[twintype]['K2_m'][i]==K2).all():
                            isin=True  
        K1in=[1 for K1Mi in twin_systems[twintype]['K1_m'] if (K1 == K1Mi).all() or (-1*K1 == K1Mi).all()]
        E1in=[1 for E1Mi in twin_systems[twintype]['eta1_m'] if (eta1 == E1Mi).all() or (-1*eta1 == E1Mi).all()]
        if len(K1in)==0 or len(E1in)==0:

#        if not isin:
            twin_systems[twintype]['eta1_m'].append(eta1)
            twin_systems[twintype]['K1_m'].append(K1)
            twin_systems[twintype]['eta2_m'].append(eta2)
            twin_systems[twintype]['K2_m'].append(K2)
            a1=uvw2xyz.dot(twin_systems[twintype]['eta1_m'][-1])
            twin_systems[twintype]['a1_m'].append(a1/np.linalg.norm(a1))
            a2=uvw2xyz.dot(twin_systems[twintype]['eta2_m'][-1])
            twin_systems[twintype]['a2_m'].append(a2/np.linalg.norm(a2))
            n1=hkl2xyz.dot(twin_systems[twintype]['K1_m'][-1])
            twin_systems[twintype]['n1_m'].append(n1/np.linalg.norm(n1))
            n2=hkl2xyz.dot(twin_systems[twintype]['K2_m'][-1])
            twin_systems[twintype]['n2_m'].append(n2/np.linalg.norm(n2))
            
            twin_systems[twintype]['uvw2xyz_m'].append(np.matmul((2*np.outer(twin_systems[twintype]['a1_m'][-1],twin_systems[twintype]['a1_m'][-1])-np.eye(3)),uvw2xyz))
            twin_systems[twintype]['R_m'].append(twin_systems[twintype]['uvw2xyz_m'][-1].dot(inv(uvw2xyz)))
            
            #C1_m=-np.eye(3)+2*np.outer(n1_m,n1_m)
            #twin_systems[twintype]['C_m'].append(-np.eye(3)+2*np.outer(twin_systems[twintype]['n1_m'][0],twin_systems[twintype]['n1_m'][0]))
            twin_systems[twintype]['C_m'].append(2*np.outer(twin_systems[twintype]['a1_m'][-1],twin_systems[twintype]['a1_m'][-1])-np.eye(3))
            twin_systems[twintype]['shear_angle'].append(2*abs(np.pi/2-np.arccos((twin_systems[twintype]['uvw2xyz_m'][-1].dot(twin_systems[twintype]['eta2_m'][-1])
                                                 /norm(twin_systems[twintype]['uvw2xyz_m'][-1].dot(twin_systems[twintype]['eta2_m'][-1]))).dot(
                                                     twin_systems[twintype]['a1_m'][-1]))))
            twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][-1]/2)*2)
            n2eta1K1=np.cross(twin_systems[twintype]['a1_m'][-1],twin_systems[twintype]['n1_m'][-1]);
            Qrot=np.vstack((twin_systems[twintype]['a1_m'][-1],n2eta1K1,twin_systems[twintype]['n1_m'][-1]))
            Qrot.dot(twin_systems[twintype]['a1_m'][-1])
            Qri=np.linalg.inv(Qrot)
            StrainTensor=np.array([[0,0,twin_systems[twintype]['s'][-1]/2],[0,0,0],[twin_systems[twintype]['s'][-1]/2,0,0]])
            twin_systems[twintype]['StrainTensor_m'].append(np.matmul(np.matmul(Qri,StrainTensor),Qrot))
            
            D,V = np.linalg.eig(twin_systems[twintype]['StrainTensor_m'][-1])
            Idxs = np.argsort(D)
            Lambda=D[Idxs]
            V = V[:,Idxs]
            twin_systems[twintype]['Tension_m'].append(V[:,0])
            twin_systems[twintype]['Compression_m'].append(V[:,2])
            
            twin_systems[twintype]['dm_m'].append(get_twinning_dislocation(twin_systems[twintype]['K1_m'][-1],
                                                                    twin_systems[twintype]['eta1_m'][-1],
                                                                    twin_systems[twintype]['eta2_m'][-1],
                                                                    L_M,G=G_M,Gr=Gr_M))
            
        
    
    
    
    #deformation twins in austenite
    #114
    twintype='114'
    twin_systems[twintype]={}
    for key in list(twindata[0].keys()):
        twin_systems[twintype][key]=[]
    twin_systems[twintype]['vars']=[]
    twin_systems[twintype]['uvw2xyz_m']=[]
    twin_systems[twintype]['uvw2xyz_a']=[]
    twin_systems[twintype]['StrainTensor_a']=[]
    twin_systems[twintype]['Tension_a']=[]
    twin_systems[twintype]['Compression_a']=[]
    twin_systems[twintype]['Variant pairs']=[]
    twin_systems[twintype]['dm_a']=[]
    twin_systems[twintype]['dm_m']=[]
 
    # twin_systems[twintype]['eta1_a'].append([-1,-2,-2])
    # twin_systems[twintype]['K1_a'].append([-1,-1,4])
    # twin_systems[twintype]['eta2_a'].append([0,0,1])
    # twin_systems[twintype]['K2_a'].append([1,1,0])
    
    twin_systems[twintype]['eta1_a'].append([1,2,2])
    twin_systems[twintype]['K1_a'].append([-4,1,1])
    twin_systems[twintype]['eta2_a'].append([-1,0,0])
    twin_systems[twintype]['K2_a'].append([0,1,1])
    uvw2xyz=Parent_uvw2xyz
    hkl2xyz=Parent_hkl2xyz
    a1=uvw2xyz.dot(twin_systems[twintype]['eta1_a'][0])
    twin_systems[twintype]['a1_a'].append(a1/np.linalg.norm(a1))
    a2=uvw2xyz.dot(twin_systems[twintype]['eta2_a'][0])
    twin_systems[twintype]['a2_a'].append(a2/np.linalg.norm(a2))
    n1=hkl2xyz.dot(twin_systems[twintype]['K1_a'][0])
    twin_systems[twintype]['n1_a'].append(n1/np.linalg.norm(n1))
    n2=hkl2xyz.dot(twin_systems[twintype]['K2_a'][0])
    twin_systems[twintype]['n2_a'].append(n2/np.linalg.norm(n2))
    
    twin_systems[twintype]['uvw2xyz_a'].append(np.matmul((2*np.outer(twin_systems[twintype]['a1_a'][0],twin_systems[twintype]['a1_a'][0])-np.eye(3)),uvw2xyz))
    
    #twin_systems[twintype]['R_a'].append(twin_systems[twintype]['uvw2xyz_a'][-1].dot(inv(uvw2xyz)))
    #C1_m=-np.eye(3)+2*np.outer(n1_m,n1_m)
    #np.eye(3)-2*np.outer(n1_m,n1_m)
    #print('cc')
    #twin_systems[twintype]['C_a'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_a'][0],twin_systems[twintype]['n1_a'][0]))
    twin_systems[twintype]['C_a'].append(2*np.outer(twin_systems[twintype]['a1_a'][0],twin_systems[twintype]['a1_a'][0])-np.eye(3))
    
    twin_systems[twintype]['shear_angle'].append(2*abs(np.pi/2-np.arccos((twin_systems[twintype]['uvw2xyz_a'][0].dot(twin_systems[twintype]['eta2_a'][0])
                                         /norm(twin_systems[twintype]['uvw2xyz_a'][0].dot(twin_systems[twintype]['eta2_a'][0]))).dot(
                                             twin_systems[twintype]['a1_a'][0]))))
    twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][0]/2)*2)
    
    n2eta1K1=np.cross(twin_systems[twintype]['a1_a'][-1],twin_systems[twintype]['n1_a'][-1]);
    Qrot=np.vstack((twin_systems[twintype]['a1_a'][-1],n2eta1K1,twin_systems[twintype]['n1_a'][-1]))
    Qrot.dot(twin_systems[twintype]['a1_a'][-1])
    Qri=np.linalg.inv(Qrot)
    StrainTensor=np.array([[0,0,twin_systems[twintype]['s'][-1]/2],[0,0,0],[twin_systems[twintype]['s'][-1]/2,0,0]])
    twin_systems[twintype]['StrainTensor_a'].append(np.matmul(np.matmul(Qri,StrainTensor),Qrot))
    
    D,V = np.linalg.eig(twin_systems[twintype]['StrainTensor_a'][-1])
    Idxs = np.argsort(D)[::-1]
    Lambda=D[Idxs]
    V = V[:,Idxs]
    twin_systems[twintype]['Tension_a'].append(V[:,0])
    twin_systems[twintype]['Compression_a'].append(V[:,2])
    
    twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1],
                                                            twin_systems[twintype]['eta1_a'][-1],
                                                            twin_systems[twintype]['eta2_a'][-1],
                                                            L_A,G=G_A,Gr=Gr_A))
    
    
    
    
    
    for symhkl,symuvw in zip( B2.reciprocal_symmetry_operations(),B2.symmetry_operations()):
        K1=np.round(symhkl[0:3,0:3].dot(twin_systems[twintype]['K1_a'][0]),5)
        K2=np.round(symhkl[0:3,0:3].dot(twin_systems[twintype]['K2_a'][0]),5)
        eta1=np.round(symuvw[0:3,0:3].dot(twin_systems[twintype]['eta1_a'][0]),5)
        eta2=np.round(symuvw[0:3,0:3].dot(twin_systems[twintype]['eta2_a'][0]),5)
        isin=False
        for i in range(0,len(twin_systems[twintype]['eta1_a'])):
            if (twin_systems[twintype]['eta1_a'][i]==eta1).all():
                if (twin_systems[twintype]['eta2_a'][i]==eta2).all():
                    if (twin_systems[twintype]['K1_a'][i]==K1).all():
                        if (twin_systems[twintype]['K2_a'][i]==K2).all():
                            isin=True  
        if not isin:
            K1in=[1 for K1Mi in twin_systems[twintype]['K1_a'] if (K1 == K1Mi).all() or (-1*K1 == K1Mi).all()]
            E1in=[1 for E1Mi in twin_systems[twintype]['eta1_a'] if (eta1 == E1Mi).all() or (-1*eta1 == E1Mi).all()]
            if len(K1in)==0 or len(E1in)==0:
                twin_systems[twintype]['eta1_a'].append(eta1)
                twin_systems[twintype]['K1_a'].append(K1)
                twin_systems[twintype]['eta2_a'].append(eta2)
                twin_systems[twintype]['K2_a'].append(K2)
                a1=uvw2xyz.dot(twin_systems[twintype]['eta1_a'][-1])
                twin_systems[twintype]['a1_a'].append(a1/np.linalg.norm(a1))
                a2=uvw2xyz.dot(twin_systems[twintype]['eta2_a'][-1])
                twin_systems[twintype]['a2_a'].append(a2/np.linalg.norm(a2))
                n1=uvw2xyz.dot(twin_systems[twintype]['K1_a'][-1])
                twin_systems[twintype]['n1_a'].append(n1/np.linalg.norm(n1))
                n2=uvw2xyz.dot(twin_systems[twintype]['K2_a'][-1])
                twin_systems[twintype]['n2_a'].append(n2/np.linalg.norm(n2))
                
                twin_systems[twintype]['uvw2xyz_a'].append(np.matmul((2*np.outer(twin_systems[twintype]['a1_a'][-1],twin_systems[twintype]['a1_a'][-1])-np.eye(3)),uvw2xyz))
                #twin_systems[twintype]['R_a'].append(twin_systems[twintype]['uvw2xyz_a'][-1].dot(inv(uvw2xyz)))
                #twin_systems[twintype]['C_a'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_a'][-1],twin_systems[twintype]['n1_a'][-1]))
                twin_systems[twintype]['C_a'].append(2*np.outer(twin_systems[twintype]['a1_a'][-1],twin_systems[twintype]['a1_a'][-1])-np.eye(3))
                twin_systems[twintype]['shear_angle'].append(2*abs(np.pi/2-np.arccos((twin_systems[twintype]['uvw2xyz_a'][-1].dot(twin_systems[twintype]['eta2_a'][-1])
                                                     /norm(twin_systems[twintype]['uvw2xyz_a'][-1].dot(twin_systems[twintype]['eta2_a'][-1]))).dot(
                                                         twin_systems[twintype]['a1_a'][-1]))))
                twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][-1]/2)*2)
                
                n2eta1K1=np.cross(twin_systems[twintype]['a1_a'][-1],twin_systems[twintype]['n1_a'][-1]);
                Qrot=np.vstack((twin_systems[twintype]['a1_a'][-1],n2eta1K1,twin_systems[twintype]['n1_a'][-1]))
                Qrot.dot(twin_systems[twintype]['a1_a'][-1])
                Qri=np.linalg.inv(Qrot)
                StrainTensor=np.array([[0,0,twin_systems[twintype]['s'][-1]/2],[0,0,0],[twin_systems[twintype]['s'][-1]/2,0,0]])
                twin_systems[twintype]['StrainTensor_a'].append(np.matmul(np.matmul(Qri,StrainTensor),Qrot))
                
                D,V = np.linalg.eig(twin_systems[twintype]['StrainTensor_a'][-1])
                Idxs = np.argsort(D)[::-1]
                Lambda=D[Idxs]
                V = V[:,Idxs]
                twin_systems[twintype]['Tension_a'].append(V[:,0])
                twin_systems[twintype]['Compression_a'].append(V[:,2])

                twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1],
                                                                        twin_systems[twintype]['eta1_a'][-1],
                                                                        twin_systems[twintype]['eta2_a'][-1],
                                                                        L_A,G=G_A,Gr=Gr_A))


    #114
    twintypes=['112']
    eta1s_a=[[1,1,1]]
    K1s_a=[[-2,1,1]]
    eta2s_a=[[-1,0,0]]
    K2s_a=[[0,1,1]]

    twintypes.append('115')
    eta1s_a.append([2,5,5])
    K1s_a.append([-5,1,1])
    eta2s_a.append([-1,0,0])
    K2s_a.append([0,1,1])

    twintypes.append('111')
    eta1s_a.append([-1,-1,-2])
    K1s_a.append([-1,-1,1])
    eta2s_a.append([-1,-1,2])
    K2s_a.append([-1,-1,-1])

    

    for twintype,eta1_a,K1_a,eta2_a,K2_a in zip(twintypes,eta1s_a,K1s_a,eta2s_a,K2s_a):   
        twin_systems[twintype]={}
        for key in list(twindata[0].keys()):
            twin_systems[twintype][key]=[]
        twin_systems[twintype]['vars']=[]
        twin_systems[twintype]['uvw2xyz_m']=[]
        twin_systems[twintype]['uvw2xyz_a']=[]
        twin_systems[twintype]['StrainTensor_a']=[]
        twin_systems[twintype]['Tension_a']=[]
        twin_systems[twintype]['Compression_a']=[]
        twin_systems[twintype]['Variant pairs']=[]
        twin_systems[twintype]['dm_a']=[]
        twin_systems[twintype]['dm_m']=[]
         
        # twin_systems[twintype]['eta1_a'].append([-1,-2,-2])
        # twin_systems[twintype]['K1_a'].append([-1,-1,4])
        # twin_systems[twintype]['eta2_a'].append([0,0,1])
        # twin_systems[twintype]['K2_a'].append([1,1,0])
        
        twin_systems[twintype]['eta1_a'].append(eta1_a)
        twin_systems[twintype]['K1_a'].append(K1_a)
        twin_systems[twintype]['eta2_a'].append(eta2_a)
        twin_systems[twintype]['K2_a'].append(K2_a)
        uvw2xyz=Parent_uvw2xyz
        hkl2xyz=Parent_hkl2xyz
        a1=uvw2xyz.dot(twin_systems[twintype]['eta1_a'][0])
        twin_systems[twintype]['a1_a'].append(a1/np.linalg.norm(a1))
        a2=uvw2xyz.dot(twin_systems[twintype]['eta2_a'][0])
        twin_systems[twintype]['a2_a'].append(a2/np.linalg.norm(a2))
        n1=hkl2xyz.dot(twin_systems[twintype]['K1_a'][0])
        twin_systems[twintype]['n1_a'].append(n1/np.linalg.norm(n1))
        n2=hkl2xyz.dot(twin_systems[twintype]['K2_a'][0])
        twin_systems[twintype]['n2_a'].append(n2/np.linalg.norm(n2))
        
        twin_systems[twintype]['uvw2xyz_a'].append(np.matmul((2*np.outer(twin_systems[twintype]['a1_a'][0],twin_systems[twintype]['a1_a'][0])-np.eye(3)),uvw2xyz))
        #twin_systems[twintype]['R_a'].append(twin_systems[twintype]['uvw2xyz_a'][-1].dot(inv(uvw2xyz)))
        #C1_m=-np.eye(3)+2*np.outer(n1_m,n1_m)
        #np.eye(3)-2*np.outer(n1_m,n1_m)
        #twin_systems[twintype]['C_a'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_a'][0],twin_systems[twintype]['n1_a'][0]))
        twin_systems[twintype]['C_a'].append(2*np.outer(twin_systems[twintype]['a1_a'][0],twin_systems[twintype]['a1_a'][0])-np.eye(3))
        twin_systems[twintype]['shear_angle'].append(2*abs(np.pi/2-np.arccos((twin_systems[twintype]['uvw2xyz_a'][0].dot(twin_systems[twintype]['eta2_a'][0])
                                             /norm(twin_systems[twintype]['uvw2xyz_a'][0].dot(twin_systems[twintype]['eta2_a'][0]))).dot(
                                                 twin_systems[twintype]['a1_a'][0]))))
        twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][0]/2)*2)
        
        n2eta1K1=np.cross(twin_systems[twintype]['a1_a'][-1],twin_systems[twintype]['n1_a'][-1]);
        Qrot=np.vstack((twin_systems[twintype]['a1_a'][-1],n2eta1K1,twin_systems[twintype]['n1_a'][-1]))
        Qrot.dot(twin_systems[twintype]['a1_a'][-1])
        Qri=np.linalg.inv(Qrot)
        StrainTensor=np.array([[0,0,twin_systems[twintype]['s'][-1]/2],[0,0,0],[twin_systems[twintype]['s'][-1]/2,0,0]])
        twin_systems[twintype]['StrainTensor_a'].append(np.matmul(np.matmul(Qri,StrainTensor),Qrot))
        
        D,V = np.linalg.eig(twin_systems[twintype]['StrainTensor_a'][-1])
        Idxs = np.argsort(D)[::-1]
        Lambda=D[Idxs]
        V = V[:,Idxs]
        twin_systems[twintype]['Tension_a'].append(V[:,0])
        twin_systems[twintype]['Compression_a'].append(V[:,2])
        
        
        twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1],
                                                                twin_systems[twintype]['eta1_a'][-1],
                                                                twin_systems[twintype]['eta2_a'][-1],
                                                                L_A,G=G_A,Gr=Gr_A))
            
        #print(twin_systems[twintype]['dm_a'])
        #print(twin_systems[twintype]['eta2_a'][-1])
        #print(G_A)
        
        for symhkl,symuvw in zip( B2.reciprocal_symmetry_operations(),B2.symmetry_operations()):
            K1=np.round(symhkl[0:3,0:3].dot(twin_systems[twintype]['K1_a'][0]),5)
            K2=np.round(symhkl[0:3,0:3].dot(twin_systems[twintype]['K2_a'][0]),5)
            eta1=np.round(symuvw[0:3,0:3].dot(twin_systems[twintype]['eta1_a'][0]),5)
            eta2=np.round(symuvw[0:3,0:3].dot(twin_systems[twintype]['eta2_a'][0]),5)
            isin=False
            for i in range(0,len(twin_systems[twintype]['eta1_a'])):
                if (twin_systems[twintype]['eta1_a'][i]==eta1).all():
                    if (twin_systems[twintype]['eta2_a'][i]==eta2).all():
                        if (twin_systems[twintype]['K1_a'][i]==K1).all():
                            if (twin_systems[twintype]['K2_a'][i]==K2).all():
                                isin=True  
            if not isin:
                K1in=[1 for K1Mi in twin_systems[twintype]['K1_a'] if (K1 == K1Mi).all() or (-1*K1 == K1Mi).all()]
                E1in=[1 for E1Mi in twin_systems[twintype]['eta1_a'] if (eta1 == E1Mi).all() or (-1*eta1 == E1Mi).all()]
                if len(K1in)==0 or len(E1in)==0:
                    twin_systems[twintype]['eta1_a'].append(eta1)
                    twin_systems[twintype]['K1_a'].append(K1)
                    twin_systems[twintype]['eta2_a'].append(eta2)
                    twin_systems[twintype]['K2_a'].append(K2)
                    a1=uvw2xyz.dot(twin_systems[twintype]['eta1_a'][-1])
                    twin_systems[twintype]['a1_a'].append(a1/np.linalg.norm(a1))
                    a2=uvw2xyz.dot(twin_systems[twintype]['eta2_a'][-1])
                    twin_systems[twintype]['a2_a'].append(a2/np.linalg.norm(a2))
                    n1=uvw2xyz.dot(twin_systems[twintype]['K1_a'][-1])
                    twin_systems[twintype]['n1_a'].append(n1/np.linalg.norm(n1))
                    n2=uvw2xyz.dot(twin_systems[twintype]['K2_a'][-1])
                    twin_systems[twintype]['n2_a'].append(n2/np.linalg.norm(n2))
                    
                    twin_systems[twintype]['uvw2xyz_a'].append(np.matmul((2*np.outer(twin_systems[twintype]['a1_a'][-1],twin_systems[twintype]['a1_a'][-1])-np.eye(3)),uvw2xyz))
                    #twin_systems[twintype]['R_a'].append(twin_systems[twintype]['uvw2xyz_a'][-1].dot(inv(uvw2xyz)))
                    #twin_systems[twintype]['C_a'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_a'][-1],twin_systems[twintype]['n1_a'][-1]))
                    twin_systems[twintype]['C_a'].append(2*np.outer(twin_systems[twintype]['a1_a'][-1],twin_systems[twintype]['a1_a'][-1])-np.eye(3))
                    twin_systems[twintype]['shear_angle'].append(2*abs(np.pi/2-np.arccos((twin_systems[twintype]['uvw2xyz_a'][-1].dot(twin_systems[twintype]['eta2_a'][-1])
                                                         /norm(twin_systems[twintype]['uvw2xyz_a'][-1].dot(twin_systems[twintype]['eta2_a'][-1]))).dot(
                                                             twin_systems[twintype]['a1_a'][-1]))))
                    twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][-1]/2)*2)
                    
                    n2eta1K1=np.cross(twin_systems[twintype]['a1_a'][-1],twin_systems[twintype]['n1_a'][-1]);
                    Qrot=np.vstack((twin_systems[twintype]['a1_a'][-1],n2eta1K1,twin_systems[twintype]['n1_a'][-1]))
                    Qrot.dot(twin_systems[twintype]['a1_a'][-1])
                    Qri=np.linalg.inv(Qrot)
                    StrainTensor=np.array([[0,0,twin_systems[twintype]['s'][-1]/2],[0,0,0],[twin_systems[twintype]['s'][-1]/2,0,0]])
                    twin_systems[twintype]['StrainTensor_a'].append(np.matmul(np.matmul(Qri,StrainTensor),Qrot))
                    
                    D,V = np.linalg.eig(twin_systems[twintype]['StrainTensor_a'][-1])
                    Idxs = np.argsort(D)[::-1]
                    Lambda=D[Idxs]
                    V = V[:,Idxs]
                    twin_systems[twintype]['Tension_a'].append(V[:,0])
                    twin_systems[twintype]['Compression_a'].append(V[:,2])
    
                    twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1],
                                                                            twin_systems[twintype]['eta1_a'][-1],
                                                                            twin_systems[twintype]['eta2_a'][-1],
                                                                            L_A,G=G_A,Gr=Gr_A))
    
    

    #print('test')        
    return twin_systems           

def get_twinningdata(orim,eus,Ldir_css,twin_systems,twt,phase, tension=True):
    #inputs:
    #orim: list of orientations as list of orientation matrices csc=orim[gi]*css (css - coord.sys of sample, csc - coord sys of lattice)
    #eus - list of Euler angles 
    #Ldir_css - unit vector of loading
    #twin_systems: dictionary of twinning systems
    #twt: string with type of twinning to be considered - will be used as twin_systems[twt]
    #phase: 'a' for austenite twinning system, 'm' for martensite twinning system
    #outputs:
    #Dictionary containing under key ['n_csl'] list of the normal of the twinning system with higesthest probability to by activated
    #in individual grains (in the coordinate system of the lattice,
    #under key ['n_css'] list of normals the coordinate system of the sample,
    #under key ['angle_n_css'] list of angles between normals and loading direction
    #and under key ['SF'] list of Schmid factors ranging between -0.5 to 0.5, i.e. propensity to twinning
    #and under key ['twsimax'] index of twinning system with highest SF
    #and under key ['neworim'] list of orientations updated by rotation due to twinning
    #and under key ['neweus'] euler angles of updated orientations
    TwinnigData={}
    TwinnigData['n_csl']=[]
    TwinnigData['n_css']=[]
    TwinnigData['angle_n_css']=[]
    TwinnigData['SF']=[]
    TwinnigData['neworim']=[]
    TwinnigData['neweus']=[]
    TwinnigData['eus']=[]
    TwinnigData['twsimax']=[]
    TwinnigData['n1']=[]
    TwinnigData['a1']=[]
    titles='StrainNonSym DefGrad StrainSymEng StrainSymGL DefGradTwin StrainSymEngTwin StrainSymGLTwin StrainLdirSymGl StrainLdirSymEng'.split()
    for title in titles:
        TwinnigData[title]=[]
    #grain index
    gi=0
    #Unit vector of the loading direction in coordinate system of the sample
    #Ldir_css=np.array([0,0,1])
    for gi in range(0,len(orim)):
        # Loading direction in coordinate system of the lattice
        Ldir_csl=orim[gi].dot(Ldir_css)
        SFgi=[]
        #index over all the twinning systems within twt family and get all their Schmid factors
        for twsi in range(0,len(twin_systems[twt]['n1_'+phase])):
            #Unit vector of the normal to twinning plane (correspond to K1)
            n1=twin_systems[twt]['n1_'+phase][twsi]
            #Unit vector of the shear direction (correspond to eta1)
            a1=twin_systems[twt]['a1_'+phase][twsi]
            #Get propensity to twinning
            #SFgi.append(np.sign(a1.dot(Ldir_csl))*n1.dot(Ldir_csl)*a1.dot(Ldir_csl))
            SFgi.append(n1.dot(Ldir_csl)*a1.dot(Ldir_csl))

        SFgi=np.array(SFgi)
        if tension:
            twsimax=np.argmax(SFgi)
        else:
            twsimax=np.argmax(SFgi)
        TwinnigData['n_csl'].append(twin_systems[twt]['n1_'+phase][twsimax])
        TwinnigData['n_css'].append(orim[gi].T.dot(twin_systems[twt]['n1_'+phase][twsimax]))
        TwinnigData['angle_n_css'].append(np.arccos(TwinnigData['n_csl'][-1].dot(Ldir_csl))*180/np.pi)
        TwinnigData['SF'].append(SFgi[twsimax])
        TwinnigData['neworim'].append(twin_systems[twt]['C_'+phase][twsimax].dot(orim[gi]))
        TwinnigData['neweus'].append(euler_angles_from_matrix(TwinnigData['neworim'][-1]))
        TwinnigData['eus'].append(eus[gi])
        TwinnigData['twsimax'].append(twsimax)
        TwinnigData['n1'].append(twin_systems[twt]['n1_'+phase][twsimax])
        TwinnigData['a1'].append(twin_systems[twt]['a1_'+phase][twsimax])
        a1=twin_systems[twt]['a1_'+phase][twsimax]
        n1=twin_systems[twt]['n1_'+phase][twsimax]
        s=twin_systems[twt]['s'][twsimax]
        z=np.cross(a1,n1)
        #transformation matrix into system aligned with a1, n1
        T=np.array([n1,a1,z])
        #transformation matrix into system aligned with basal directions of the twinned lattice
        Tij=2*np.outer(a1,a1)-np.eye(3)
        #Strains and def. gradient in the system aligned with basal directions of the matrix
        StrainNonSym=s*np.outer(a1,n1)
        DefGrad=np.eye(3)+StrainNonSym
        StrainSymEng=0.5*(StrainNonSym+StrainNonSym.T)#0.5*(DefGrad+DefGrad.T) - np.eye(3)
        #Green Lagrange takes into account the elongation of the unit side [010] of the reference square that is sheared along [100] by s
        #so that [010] becomes [s10] (relatively longer (sqrt(s**2+1)) compared to [010])
        StrainSymGL=0.5*(DefGrad.T.dot(DefGrad)-np.eye(3))
        StrainLdirSymGl=np.array(Ldir_csl).dot(StrainSymGL.dot(Ldir_csl))
        StrainLdirSymEng=np.array(Ldir_csl).dot(StrainSymEng.dot(Ldir_csl))
        #Strains and def. gradient in the system aligned with basal directions of the twin
        DefGradTwin=Tij.dot(DefGrad.dot(Tij.T))
        StrainSymEngTwin=0.5*(DefGradTwin+DefGradTwin.T) - np.eye(3)
        StrainSymGLTwin=0.5*(DefGradTwin.T.dot(DefGradTwin)-np.eye(3))
        for title in titles:
            exec(f'TwinnigData["{title}"].append({title})')


    return TwinnigData


def get_twinning_dislocation(K1,eta1,eta2,L,G=None,Gr=None):
    #Checked for cubic lattice and K1=-1-11,eta1=-1-1-2,eta2=-1-12 =1/6
    #https://www.sciencedirect.com/science/article/pii/S0966979514000892?via%3Dihub
    #L -latice tensor converting uvw->xyz
    #if type(K1) is list:
        #K1=K1[0]   
    #    print(K1)        
    #if type(eta1) is list:
    #    eta1=eta1[0]
    #if type(eta2) is list:
    #    eta2=eta2[0]
    #print(K1)
    #print(eta1)
    #print(eta2)
    if True:
        if G is None:
            #Meteric tensor ||[x,y,x]||^2=[uvw]*G_A*[uvw]^T
            G = np.matmul(L.T,L)
        if Gr is None:
            #Reciprocal Meteric tensor ||[x,y,x]||^2=d_hkl^2=1/[hkl]*Gr_A*[hkl]^T
            Gr = np.linalg.inv(G)   
        if type(K1) is not np.ndarray:
            K1=np.array(K1)
        if type(eta1) is not np.ndarray:
            eta1=np.array(eta1)
        if type(eta2) is not np.ndarray:
            eta2=np.array(eta2)  
        #rotation angle of the twinning = shear is defined as 2* tan(shear angle/2)
        shear_angle=2*abs(np.pi/2-np.arccos((L.dot(eta2)/np.linalg.norm(L.dot(eta2))).dot(L.dot(eta1)/np.linalg.norm(L.dot(eta1)))))
        #print(shear_angle
        #d-spacing of twinning plane K1
        dK1=1/np.sqrt(K1.dot(Gr).dot(K1))
        #length of eta1
        deta1=np.sqrt(eta1.dot(G).dot(eta1))
        #slip of the firs K1 plane above the twinning plane
        s=2*np.tan(shear_angle/2)*dK1
        #twinning dislocation magnitude as fraction of eta1 (dm*eta1)
        dm=s/deta1
    return dm

def gen_twinned_lattice_points(ParentLatticePoints,eta1,shear_angle,K1,shift=0.0,dK1=None,bvr=None,deta1=None):
    #bvr
    #bvr - burgers vector length in terms of fraction of eta1 direction - deta1
    #dK1 - d-spacing of K1
    #deta1 - length of eta1 direction
    TwinnedPoints=[]
    for Points in ParentLatticePoints:
        twpoints=[]
        for points in Points:
            #Spoints=copy.deepcopy(points)*0
            #for idx in range(3):
                #Spoints[idx,:]=shift[idx]
            if bvr is None:
                twpoints.append(points+2*np.array([eta1]).T*np.tan(shear_angle/2)*(K1.dot(points)+shift))
            else:
                twpoints.append(points+bvr*deta1*np.array([eta1]).T*(np.modf(((K1.dot(points)+shift*np.sign(K1.dot(points)))/dK1))[1]))
        
        TwinnedPoints.append(twpoints)
    return TwinnedPoints

def write_txt(filename,Header,DATA):
    f=open(filename, "w")
    f.write('\t'.join(Header)+'\n')
    
    inc=0;
    for data in DATA:
        inc+=1
        print(str(inc)+'/'+str(len(DATA)))
        formatted = ['%.8f'% item  for item in data]
        f.write('\t'.join(formatted)+'\n')
    
    f.close()

def read_txt(filename,delimiter='\t',skiprows=1):
    f=open(filename, "r")
    Header=f.readline().strip().split(delimiter)
    f.close()
    Data = np.loadtxt(filename,  skiprows=skiprows, delimiter=delimiter);
    return Header,Data

def plane_line_intersection(n,V0,P0,P1):
    # n: normal vector of the Plane 
    # V0: any point that belongs to the Plane 
    # P0: end point 1 of the segment P0P1
    # P1:  end point 2 of the segment P0P1


    w = P0 - V0;
    u = P1-P0;
    N = -np.dot(n,w);
    D = np.dot(n,u)
    sI = N / D
    I = P0+ sI*u
    return I


def plot_cut2D(ax,Lattice_points,normal,horizontal,vertical,col,alpha=1):
    IP=[]
    eps=1e-2
    for point in Lattice_points:
        point=np.array(point);
        point_proj_x = horizontal.dot(point)
        point_proj_y = vertical.dot(point)
        point_proj_n = normal.dot(point)
        if (point_proj_n<= eps).any():
            if not (point_proj_n<= eps).all():
                idxp0=np.where(point_proj_n> eps)[0][0]                                           
                point[:,idxp0]=plane_line_intersection(normal,np.array([0,0,0]),point[:,0],point[:,1])
                point_proj_x = horizontal.dot(point)
                point_proj_y = vertical.dot(point)
                IP.append([point_proj_x[idxp0],point_proj_y[idxp0]])
            ax.plot(point_proj_x,point_proj_y,col,alpha=alpha)
                
#        else:
#            ax.plot(point_proj_x,point_proj_y,col)
        r=[]
        theta =[]
        eps=1e-4
        IPnew=[]
        for ip in IP:
            #print(ip)
            x=ip[0]
            y=ip[1]
            R=np.sqrt(x*x + y*y)
            TH=np.arctan2(y, x)*180./np.pi
            if TH<0:
                TH=360-abs(TH)
            if len(r)>0 and R>eps:
                include=True
                if min(abs(r-R))<eps: 
                    idx = np.where(abs(r-R)==min(abs(r-R)))[0][0]
                    if abs(theta[idx]-TH)<eps:
                        include=False
                        
                if include:
                    r.append(R)
                    theta.append(TH)
                    IPnew.append(ip)
            else:
                if R>eps:
                    r.append(R)
                    theta.append(TH)
                    IPnew.append(ip)
        idxs=np.argsort(theta)
#        for IPi in range(0,len(idxs)-1):
#            ax.plot([IPnew[idxs[IPi]][0],IPnew[idxs[IPi+1]][0]],[IPnew[idxs[IPi]][1],IPnew[idxs[IPi+1]][1]],'r')
        IPnew=np.array(IPnew)
        IPnew=IPnew[idxs]
    return IPnew
#        if len(IPnew)>0:
#            ax.add_patch(Polygon(IPnew, color=col,closed=True,fill=False, hatch=hatch))

def flipvector(v, Tol=1e-9):
    vm = np.round(v/Tol)*Tol
    c=1.0
#    if len(vm[vm<0])==len(vm[vm<>0]):
    if len(vm[vm<0])==len(vm[vm!=0]):
        vm=-1*vm
        c=-1.0
    return vm,c

def flipvector2negative(v, Tol=1e-9):
    vm = np.round(v/Tol)*Tol
    c=1.0
#    if len(vm[vm<0])==len(vm[vm<>0]):
    if len(vm[vm>0])==len(vm[vm!=0]):
        vm=-1*vm
        c=-1.0
    return vm,c
  
def vector2miller(v, MIN=True, Tol=1e-9,tol=1e5,text=False,decimals=3):
    vm = np.round(v/Tol)*Tol
    #print((np.round(vm)==vm).all())
    #if (vm==np.array([ 2. ,-1. , 1.])).all():
        #print((np.round(vm)==vm).all())
        #print(vm)
    if (np.abs(vm)<=1).all():
        vm=np.round(vm/abs(min(vm[np.abs(vm)>1/tol]))*tol)/tol
        vm/=min(abs(vm[np.nonzero(vm)[0]]))   
    if not (np.round(vm)==vm).all():
        if MIN:
            vm=np.round(vm/abs(min(vm[np.abs(vm)>1/tol]))*tol)/tol
            vm/=min(abs(vm[np.nonzero(vm)[0]]))
        else:
            #print(vm)
            vm=np.round(vm/abs(max(vm[np.abs(vm)>1/tol]))*tol)/tol
            vm/=max(abs(vm[np.nonzero(vm)[0]]))
    else:
        #print(vm)
        vm=vm.astype('int')
        #print(vm)
        gcd=math.gcd(math.gcd(vm[0],vm[1]),vm[2])
        vm=vm.astype('float')
        vm/=gcd
    #if (vm==np.array([-2.,-1., 1.])).all():
    #    print('====================================================')
    #    print(v)
    #    print('====================================================')
    vm=np.around(vm,decimals=decimals)
    if text:
        if (vm==vm.astype(int)).all():
            vm=f"$[{{{int(vm[0])}}}{{{int(vm[1])}}}{{{int(vm[2])}}}]$".replace('{-','\\overline{')
        else:
            f"$[{{{(vm[0])}}}{{{(vm[1])}}}{{{(vm[2])}}}]$".replace('{-','\\overline{')
    return(vm)

def vectors2miller(V, MIN=True, Tol=1e-9,tol=1e5,text=False):
    VM=[]
    for v in V.T:
        VM.append(vector2miller(v,MIN=MIN,Tol=Tol,tol=tol,text=text))
    return np.array(VM).T
 

    
