"""
=========================================================
File: crystal.py
---------------------------------------------------------
Description:
    Contains functions related to crystal lattice manipulation 
    and twinning, such as converting Euler angles to rotation 
    matrices, calculating lattice parameters, and generating 
    twinning systems based on crystal symmetry operations. 
    
Author:
    Ludek Heller

Created:
    03-02-2025

License:
    General Public License
=========================================================
"""


import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from numpy.linalg import norm
import math


def eu2mat(eus):
    """
    Function that transforms Euler angles
    into a rotation matrix.
    """

    # Separate Euler angles.
    phi1, phi, phi2 = eus

    # Compute matrix indices
    g11 = np.cos(phi1) * np.cos(phi2) - np.sin(phi1) * np.sin(phi2) * np.cos(phi)
    g12 = np.sin(phi1) * np.cos(phi2) + np.cos(phi1) * np.sin(phi2) * np.cos(phi)
    g13 = np.sin(phi2) * np.sin(phi)
    g21 = -np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(phi)
    g22 = -np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(phi)
    g23 = np.cos(phi2) * np.sin(phi)
    g31 = np.sin(phi1) * np.sin(phi)
    g32 = -np.cos(phi1) * np.sin(phi)
    g33 = np.cos(phi)

    # Prepare for return
    matrix = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])

    return matrix


def euler_angles_from_matrix(Rl):
    if list != type(Rl):
        Rl = [Rl]

    Phi1 = []
    Phi2 = []
    PHI = []
    for R in Rl:
        PHI.append(np.arccos(R[2, 2]))
        if PHI[-1] == 0.0:
            Phi1.append(np.arctan2(-R[1, 0], R[0, 0]))
            Phi2.append(0.0)
        elif PHI[-1] == np.pi:
            Phi1.append(np.arctan2(R[1, 0], R[0, 0]))
            Phi2.append(0.0)
        else:
            Phi1.append(np.arctan2(R[2, 0], -R[2, 1]))
            Phi2.append(np.arctan2(R[0, 2], R[1, 2]))

    if len(Phi1) == 1:
        return Phi1[0], PHI[0], Phi2[0]
    else:
        return Phi1, PHI, Phi2


def lattice_vec(lattice_param):
    V = np.zeros((3, 3))

    if lattice_param['type'].lower() == 'cubic':
        a = lattice_param['a']
        V = a * np.eye(3)
    elif lattice_param['type'].lower() == 'tetragonal':
        a = lattice_param['a']
        b = lattice_param['b']
        c = lattice_param['c']
        V = np.zeros((3, 3))
        V[:, 0] = np.array([a, 0., 0])
        V[:, 1] = np.array([0, b, 0])
        V[:, 2] = np.array([0, 0, c])
    elif lattice_param['type'].lower() == 'hexagonal':
        a = lattice_param['a']
        c = lattice_param['c']
        V = np.zeros((3, 3))
        V[:, 0] = np.array([a, 0., 0])
        V[:, 1] = np.array([-a / 2, np.sqrt(3) / 2 * a, 0])
        # V[:,2] = np.array([-a/2,-np.sqrt(3)/2*a,0])
        V[:, 2] = np.array([0, 0, c])
    elif lattice_param['type'].lower() == 'monoclinic':
        a = lattice_param['a']
        b = lattice_param['b']
        c = lattice_param['c']
        beta = lattice_param['beta']
        V = np.zeros((3, 3))
        V[:, 0] = np.array([a, 0., 0])
        V[:, 1] = np.array([0, b, 0])
        V[:, 2] = np.array([c * np.cos(beta), 0, c * np.sin(beta)])
    elif lattice_param['type'].lower() == 'triclinic':
        a = lattice_param['a']
        b = lattice_param['b']
        c = lattice_param['c']
        alpha = lattice_param['alpha']
        beta = lattice_param['beta']
        gamma = lattice_param['gamma']
        V = np.zeros((3, 3))
        V[:, 0] = np.array([a, 0., 0])
        V[:, 1] = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = np.sqrt(c ** 2 - cx ** 2 - cy ** 2)
        V[:, 2] = np.array([cx, cy, cz])
    elif lattice_param['type'].lower() == 'trigonal':
        a = lattice_param['a']
        c = lattice_param['c']
        V = np.zeros((3, 3))
        V[:, 0] = np.array([1. / 2. * a, -np.sqrt(3) / 2. * a, 0])
        V[:, 1] = np.array([1. / 2. * a, np.sqrt(3) / 2. * a, 0])
        V[:, 2] = np.array([0, 0, c])

    return V[:, 0], V[:, 1], V[:, 2]


def flipvector(v, Tol=1e-9):
    vm = np.round(v / Tol) * Tol
    c = 1.0
    #    if len(vm[vm<0])==len(vm[vm<>0]):
    if len(vm[vm < 0]) == len(vm[vm != 0]):
        vm = -1 * vm
        c = -1.0
    return vm, c


def vector2miller(v, MIN=True, Tol=1e-9, tol=1e5, text=False, decimals=3):
    vm = np.round(v / Tol) * Tol
    # print((np.round(vm)==vm).all())
    # if (vm==np.array([ 2. ,-1. , 1.])).all():
    # print((np.round(vm)==vm).all())
    # print(vm)
    if (np.abs(vm) <= 1).all():
        vm = np.round(vm / abs(min(vm[np.abs(vm) > 1 / tol])) * tol) / tol
        vm /= min(abs(vm[np.nonzero(vm)[0]]))
    if not (np.round(vm) == vm).all():
        if MIN:
            vm = np.round(vm / abs(min(vm[np.abs(vm) > 1 / tol])) * tol) / tol
            vm /= min(abs(vm[np.nonzero(vm)[0]]))
        else:
            # print(vm)
            vm = np.round(vm / abs(max(vm[np.abs(vm) > 1 / tol])) * tol) / tol
            vm /= max(abs(vm[np.nonzero(vm)[0]]))
    else:
        # print(vm)
        vm = vm.astype('int')
        # print(vm)
        gcd = math.gcd(math.gcd(vm[0], vm[1]), vm[2])
        vm = vm.astype('float')
        vm /= gcd
    # if (vm==np.array([-2.,-1., 1.])).all():
    #    print('====================================================')
    #    print(v)
    #    print('====================================================')
    vm = np.around(vm, decimals=decimals)
    if text:
        if (vm == vm.astype(int)).all():
            vm = f"$[{{{int(vm[0])}}}{{{int(vm[1])}}}{{{int(vm[2])}}}]$".replace('{-', '\\overline{')
        else:
            f"$[{{{(vm[0])}}}{{{(vm[1])}}}{{{(vm[2])}}}]$".replace('{-', '\\overline{')
    return (vm)


def B19p_B2_lattice_correspondence(notation='Miyazaki'):
    # check
    # Variant = 8
    # B19puvw_2_B2uvw_all[:,:,Variant]
    # testB19v=[1,2,3]
    # testB2v = B19puvw_2_B2uvw_all[:,:,Variant].dot(testB2v)
    # testB19v-np.linalg.inv(B19puvw_2_B2uvw_all[:,:,Variant]).dot(testB2v)
    # testB19p=[1,2,1]
    # testB2p = B19phkl_2_B2hkl_all[:,:,Variant].dot(testB19p)
    # testB19p-np.linalg.inv(B19phkl_2_B2hkl_all[:,:,Variant]).dot(testB2p)
    # testB2p=[1,-3,1]

    # correspondance matrix betwen uvw of B19p and that of B2 /B19[uvw]->B2[uvw]/
    B19puvw_2_B2uvw_all = np.empty((3, 3, 12))
    B19phkl_2_B2hkl_all = np.empty((3, 3, 12))
    B2uvw_2_B19puvw_all = np.empty((3, 3, 12))
    B2hkl_2_B19phkl_all = np.empty((3, 3, 12))

    for Variant in range(B19puvw_2_B2uvw_all.shape[2]):

        if (Variant + 1) == 1:

            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [1, 0, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [0, 1, 1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [0, -1, 1]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [1, 0, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [0, 1, -1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [0, 1, 1]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T

            #            B19puvw_2_B2uvw_all[:,0,Variant] = [-1, 0, 0]
        #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, 1,-1]
        #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1]

        #            B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0]
        #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, 1,1]
        #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1]
        #

        if (Variant + 1) == 2:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [-1, 0, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [0, -1, -1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [0, -1, 1]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [1, 0, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [0, -1, 1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [0, -1, -1]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T

        #            B19puvw_2_B2uvw_all[:,0,Variant] = [-1, 0, 0]
        #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, -1,-1]
        #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,1]
        #
        if (Variant + 1) == 3:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [1, 0, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [0, -1, 1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [0, -1, -1]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [1, 0, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [0, 1, 1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [0, -1, 1]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T

        #            B19puvw_2_B2uvw_all[:,0,Variant] = [1, 0, 0]
        #            B19puvw_2_B2uvw_all[:,1,Variant] = [ 0, -1,1]
        #            B19puvw_2_B2uvw_all[:,2,Variant] = [ 0,-1,-1]

        if (Variant + 1) == 4:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [-1, 0, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [0, 1, -1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [0, -1, -1]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [1, 0, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [0, -1, -1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [0, 1, -1]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T
        if (Variant + 1) == 5:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 1, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [1, 0, 1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [1, 0, -1]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, -1, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [-1, 0, 1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, 0, -1]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T
        if (Variant + 1) == 6:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, -1, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [-1, 0, -1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [1, 0, -1]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, -1, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [1, 0, -1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [1, 0, 1]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T
        if (Variant + 1) == 7:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 1, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [1, 0, -1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, 0, -1]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 1, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [1, 0, 1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [1, 0, -1]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T
        if (Variant + 1) == 8:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, -1, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [-1, 0, 1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, 0, -1]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 1, 0]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [-1, 0, -1]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, 0, 1]
            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T

        if (Variant + 1) == 9:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 0, 1]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [1, 1, 0]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, 1, 0]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 0, -1]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [1, -1, 0]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, -1, 0]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T
        if (Variant + 1) == 10:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 0, -1]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [-1, -1, 0]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, 1, 0]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 0, -1]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [-1, 1, 0]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [1, 1, 0]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T
        if (Variant + 1) == 11:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 0, 1]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [-1, 1, 0]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, -1, 0]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 0, 1]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [-1, -1, 0]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [1, -1, 0]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T
        if (Variant + 1) == 12:
            if notation == 'Miyazaki':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 0, -1]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [1, -1, 0]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, -1, 0]
            elif notation == 'Waitz':
                B19puvw_2_B2uvw_all[:, 0, Variant] = [0, 0, 1]
                B19puvw_2_B2uvw_all[:, 1, Variant] = [1, 1, 0]
                B19puvw_2_B2uvw_all[:, 2, Variant] = [-1, 1, 0]

            B19phkl_2_B2hkl_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant]).T

        B2uvw_2_B19puvw_all[:, :, Variant] = inv(B19puvw_2_B2uvw_all[:, :, Variant])
        B2hkl_2_B19phkl_all[:, :, Variant] = inv(B19phkl_2_B2hkl_all[:, :, Variant])

    return B19puvw_2_B2uvw_all, B2uvw_2_B19puvw_all, B19phkl_2_B2hkl_all, B2hkl_2_B19phkl_all


def def_gradient_stressfree(Cd, LA, LM, CId=None):
    T_MA = np.empty((3, 3, Cd.shape[2]))
    T_AM = np.empty((3, 3, Cd.shape[2]))
    for Variant in range(Cd.shape[2]):
        T_MA[:, :, Variant] = Cd[:, :, Variant].dot(inv(sqrtm((Cd[:, :, Variant].T.dot(Cd[:, :, Variant])))))
        T_AM[:, :, Variant] = inv(T_MA[:, :, Variant])

    # Bain strain calculation
    # Deformation gradient
    F_AM = np.empty((3, 3, Cd.shape[2]))

    # stretch - right Cauchy-Green deformation tensor
    U_AM = np.empty((3, 3, Cd.shape[2]))
    # rotation
    Q_M = np.empty((3, 3, Cd.shape[2]))
    for Variant in range(Cd.shape[2]):
        if CId is not None:
            F_AM[:, :, Variant] = T_MA[:, :, Variant].dot(LM.dot(CId[:, :, Variant].dot(inv(LA))))
            # print("Using Ci_d - correspondense for austenite directions")
        else:
            F_AM[:, :, Variant] = T_MA[:, :, Variant].dot(LM.dot(inv(Cd[:, :, Variant]).dot(inv(LA))))
            # print("Using C_d - correspondense for martensite directions")

        # stretch matrix
        U_AM[:, :, Variant] = sqrtm(F_AM[:, :, Variant].T.dot(F_AM[:, :, Variant]))

        # rotation matrix
        Q_M[:, :, Variant] = F_AM[:, :, Variant].dot(inv(U_AM[:, :, Variant]))

    return F_AM, U_AM, Q_M, T_MA, T_AM


def get_twinningdata(orim, eus, Ldir_css, twin_systems, twt, phase, tension=True):
    # inputs:
    # orim: list of orientations as list of orientation matrices csc=orim[gi]*css (css - coord.sys of sample, csc - coord sys of lattice)
    # Ldir_css - unit vector of loading
    # twin_systems: dictionary of twinning systems
    # twt: string with type of twinning to be considered - will be used as twin_systems[twt]
    # phase: 'a' for austenite twinning system, 'm' for martensite twinning system
    # outputs:
    # Dictionary containing under key ['n_csl'] list of the normal of the twinning system with higesthest probability to by activated
    # in individual grains (in the coordinate system of the lattice,
    # under key ['n_css'] list of normals the coordinate system of the sample,
    # under key ['angle_n_css'] list of angles between normals and loading direction
    # and under key ['SF'] list of Schmid factors ranging between -0.5 to 0.5, i.e. propensity to twinning
    # and under key ['twsimax'] index of twinning system with highest SF
    # and under key ['neworim'] list of orientations updated by rotation due to twinning
    # and under key ['neweus'] euler angles of updated orientations
    TwinnigData = {}
    TwinnigData['n_csl'] = []
    TwinnigData['n_css'] = []
    TwinnigData['angle_n_css'] = []
    TwinnigData['SF'] = []
    TwinnigData['neworim'] = []
    TwinnigData['neweus'] = []
    TwinnigData['eus'] = []
    TwinnigData['twsimax'] = []
    TwinnigData['n1'] = []
    TwinnigData['a1'] = []
    titles = 'StrainNonSym DefGrad StrainSymEng StrainSymGL DefGradTwin StrainSymEngTwin StrainSymGLTwin StrainLdirSymGl StrainLdirSymEng'.split()
    for title in titles:
        TwinnigData[title] = []
    # grain index
    gi = 0
    # Unit vector of the loading direction in coordinate system of the sample
    # Ldir_css=np.array([0,0,1])
    for gi in range(0, len(orim)):
        # Loading direction in coordinate system of the lattice
        Ldir_csl = orim[gi].dot(Ldir_css)
        SFgi = []
        # index over all the twinning systems within twt family and get all their Schmid factors
        for twsi in range(0, len(twin_systems[twt]['n1_' + phase])):
            # Unit vector of the normal to twinning plane (correspond to K1)
            n1 = twin_systems[twt]['n1_' + phase][twsi]
            # Unit vector of the shear direction (correspond to eta1)
            a1 = twin_systems[twt]['a1_' + phase][twsi]
            # Get propensity to twinning
            # SFgi.append(np.sign(a1.dot(Ldir_csl))*n1.dot(Ldir_csl)*a1.dot(Ldir_csl))
            SFgi.append(n1.dot(Ldir_csl) * a1.dot(Ldir_csl))

        SFgi = np.array(SFgi)
        if tension:
            twsimax = np.argmax(SFgi)
        else:
            twsimax = np.argmax(SFgi)
        TwinnigData['n_csl'].append(twin_systems[twt]['n1_' + phase][twsimax])
        TwinnigData['n_css'].append(orim[gi].T.dot(twin_systems[twt]['n1_' + phase][twsimax]))
        TwinnigData['angle_n_css'].append(np.arccos(TwinnigData['n_csl'][-1].dot(Ldir_csl)) * 180 / np.pi)
        TwinnigData['SF'].append(SFgi[twsimax])
        TwinnigData['neworim'].append(twin_systems[twt]['C_' + phase][twsimax].dot(orim[gi]))
        TwinnigData['neweus'].append(euler_angles_from_matrix(TwinnigData['neworim'][-1]))
        TwinnigData['eus'].append(eus[gi])
        TwinnigData['twsimax'].append(twsimax)
        TwinnigData['n1'].append(twin_systems[twt]['n1_' + phase][twsimax])
        TwinnigData['a1'].append(twin_systems[twt]['a1_' + phase][twsimax])
        a1 = twin_systems[twt]['a1_' + phase][twsimax]
        n1 = twin_systems[twt]['n1_' + phase][twsimax]
        s = twin_systems[twt]['s'][twsimax]
        z = np.cross(a1, n1)
        # transformation matrix into system aligned with a1, n1
        T = np.array([n1, a1, z])
        # transformation matrix into system aligned with basal directions of the twinned lattice
        Tij = 2 * np.outer(a1, a1) - np.eye(3)
        # Strains and def. gradient in the system aligned with basal directions of the matrix
        StrainNonSym = s * np.outer(a1, n1)
        DefGrad = np.eye(3) + StrainNonSym
        StrainSymEng = 0.5 * (StrainNonSym + StrainNonSym.T)  # 0.5*(DefGrad+DefGrad.T) - np.eye(3)
        # Green Lagrange takes into account the elongation of the unit side [010] of the reference square that is sheared along [100] by s
        # so that [010] becomes [s10] (relatively longer (sqrt(s**2+1)) compared to [010])
        StrainSymGL = 0.5 * (DefGrad.T.dot(DefGrad) - np.eye(3))
        StrainLdirSymGl = np.array(Ldir_csl).dot(StrainSymGL.dot(Ldir_csl))
        StrainLdirSymEng = np.array(Ldir_csl).dot(StrainSymEng.dot(Ldir_csl))
        # Strains and def. gradient in the system aligned with basal directions of the twin
        DefGradTwin = Tij.dot(DefGrad.dot(Tij.T))
        StrainSymEngTwin = 0.5 * (DefGradTwin + DefGradTwin.T) - np.eye(3)
        StrainSymGLTwin = 0.5 * (DefGradTwin.T.dot(DefGradTwin) - np.eye(3))
        for title in titles:
            exec(f'TwinnigData["{title}"].append({title})')

    return TwinnigData


def niti_twinning(B2, B19p, Uv, Parent_uvw2xyz, Parent_hkl2xyz, Product_uvw2xyz,
                  Product_hkl2xyz, Parent_uvw_2_Product_uvw_all, Parent_hkl_2_Product_hkl_all,
                  Parent_uvw_2_Product_uvw_all_norm, SymOps, SymOpsR, miller='greaterthanone', Qv=None):
    L_A = Parent_uvw2xyz
    L_M = Product_uvw2xyz
    # Meteric tensor ||[x,y,x]||^2=[uvw]*G_A*[uvw]^T
    G_A = np.matmul(L_A.T, L_A)
    G_M = np.matmul(L_M.T, L_M)
    # Reciprocal Meteric tensor ||[x,y,x]||^2=d_hkl^2=1/[hkl]*Gr_A*[hkl]^T
    Gr_A = np.linalg.inv(G_A)
    Gr_M = np.linalg.inv(G_M)

    i = 4
    j = 5
    if Qv is None:
        twindata = twin_equation_solution(Uv[:, :, i], Uv[:, :, j],
                                          Parent_uvw2xyz, Parent_hkl2xyz,
                                          Product_uvw2xyz, Product_hkl2xyz,
                                          Parent_uvw_2_Product_uvw_all_norm[:, :, j],
                                          Parent_uvw_2_Product_uvw_all[:, :, j],
                                          Parent_hkl_2_Product_hkl_all[:, :, j], tol=1e-10)
    else:
        twindata = twin_equation_solution(Uv[:, :, i], Uv[:, :, j],
                                          Parent_uvw2xyz, Parent_hkl2xyz,
                                          Product_uvw2xyz, Product_hkl2xyz,
                                          Parent_uvw_2_Product_uvw_all_norm[:, :, j],
                                          Parent_uvw_2_Product_uvw_all[:, :, j],
                                          Parent_hkl_2_Product_hkl_all[:, :, j], tol=1e-10, miller=miller,
                                          Qj=Qv[:, :, i], Qi=Qv[:, :, j])
        # print(list(twindata[0].keys()))

    varnotation = {0: '1', 1: '1\'', 2: '2', 3: '2\'', 4: '3', 5: '3\'', 6: '4', 7: '4\'', 8: '5', 9: '5\'', 10: '6',
                   11: '6\''}
    twin_systems = {}
    for twintype in ['100', '001', 'Type I', 'Type II']:
        twin_systems[twintype] = {}
        for key in list(twindata[0].keys()):
            twin_systems[twintype][key] = []
        twin_systems[twintype]['vars'] = []
        twin_systems[twintype]['uvw2xyz_m'] = []
        twin_systems[twintype]['uvw2xyz_a'] = []
        twin_systems[twintype]['StrainTensor_m'] = []
        twin_systems[twintype]['F_m'] = []
        twin_systems[twintype]['F_a'] = []
        twin_systems[twintype]['Tension_m'] = []
        twin_systems[twintype]['Compression_m'] = []
        twin_systems[twintype]['StrainTensor_a'] = []
        twin_systems[twintype]['Tension_a'] = []
        twin_systems[twintype]['Compression_a'] = []
        twin_systems[twintype]['Variant pairs'] = []
        twin_systems[twintype]['Variant pairs 2'] = []
        twin_systems[twintype]['dm_a'] = []
        twin_systems[twintype]['dm_m'] = []
    # print(twin_systems[twintype])
    i = 5
    j = 4
    i = 4
    j = 5

    # Trnaformation twins
    Twins = np.zeros((Parent_hkl_2_Product_hkl_all.shape[2], Parent_hkl_2_Product_hkl_all.shape[2]), dtype=object)
    numtype2 = 0
    numtype1 = 0
    num100 = 0
    num001 = 0
    numall = 0
    for j in range(0, Parent_hkl_2_Product_hkl_all.shape[2]):
        for i in range(0, Parent_hkl_2_Product_hkl_all.shape[2]):

            if i != j:
                # print((i,j))
                # twin_equation_solution(Uj,Ui,...) - Solves equation Q*Uj-Ui=axn
                if Qv is None:
                    twindata = twin_equation_solution(Uv[:, :, i], Uv[:, :, j],
                                                      Parent_uvw2xyz, Parent_hkl2xyz,
                                                      Product_uvw2xyz, Product_hkl2xyz,
                                                      Parent_uvw_2_Product_uvw_all_norm[:, :, j],
                                                      Parent_uvw_2_Product_uvw_all[:, :, j],
                                                      Parent_hkl_2_Product_hkl_all[:, :, j], tol=1e-10, miller=miller)
                else:
                    twindata = twin_equation_solution(Uv[:, :, i], Uv[:, :, j],
                                                      Parent_uvw2xyz, Parent_hkl2xyz,
                                                      Product_uvw2xyz, Product_hkl2xyz,
                                                      Parent_uvw_2_Product_uvw_all_norm[:, :, j],
                                                      Parent_uvw_2_Product_uvw_all[:, :, j],
                                                      Parent_hkl_2_Product_hkl_all[:, :, j], tol=1e-10, miller=miller,
                                                      Qj=Qv[:, :, i], Qi=Qv[:, :, j])

                    # print(twin_systems[twintype])
                # print(len(twindata))
                # print(i)
                if len(twindata) > 0:

                    K1_arounddigit = 30
                    eta1_arounddigit = 30
                    arounddigit = 0
                    for twin in twindata:
                        # if twintype == '001':
                        #    print(list(twin.keys()))

                        twintype = 'no'
                        if (twin['eta1_m'] == np.round(twin['eta1_m'])).all():
                            uvw2xyz_m = np.matmul((2 * np.outer(twin['a1_m'], twin['a1_m']) - np.eye(3)),
                                                  Product_uvw2xyz)
                            uvw2xyz_a = np.matmul((2 * np.outer(twin['a1_a'], twin['a1_a']) - np.eye(3)),
                                                  np.matmul(np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:, :, i]),
                                                            Product_uvw2xyz))
                            if (twin['K1_m'] == np.round(twin['K1_m'])).all():
                                Twins[i, j] = 'Compound'
                                if (np.abs(twin['K1_m']) - np.array(
                                        [0, 0, 1]) == 0).all():  # or (K_m[0]-np.array([0,0,1])==0).all():
                                    twintype = '001'
                                    num001 += 1
                                    # print('001: {}'.format(num001))
                                elif (np.abs(twin['K1_m']) - np.array([1, 0, 0]) == 0).all():
                                    twintype = '100'
                                    num100 += 1
                                    # print('100: {}'.format(num100))
                                else:
                                    numall += 1
                                    print('noncategorized twin')
                                    print('K1_m: {}, eta1_m:{}'.format(twin['K1_m'], twin['eta1_m']))
                                    print((twin['eta1_m'] == np.around(twin['eta1_m'], decimals=arounddigit)))
                                    print(np.around(twin['eta1_m'], decimals=arounddigit))
                            else:
                                # print(twin['eta1_m'])
                                Twins[i, j] = 'Type II'
                                twintype = 'Type II'
                                K1_arounddigit = 1
                                numtype2 += 1
                                # print('Type II: {}'.format(numtype2))
                                # print(twin['K1_a'])
                        elif (twin['K1_m'] == np.round(twin['K1_m'])).all():
                            uvw2xyz_m = np.matmul((-2 * np.outer(twin['n1_m'], twin['n1_m']) + np.eye(3)),
                                                  Product_uvw2xyz)
                            uvw2xyz_a = np.matmul((-2 * np.outer(twin['n1_a'], twin['n1_a']) + np.eye(3)),
                                                  np.matmul(np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:, :, i]),
                                                            Product_uvw2xyz))
                            twintype = 'Type I'
                            Twins[i, j] = 'Type I'
                            eta1_arounddigit = 1
                            numtype1 += 1
                            # print('Type I: {}'.format(numtype1))
                            # if (twin['K1_a']==[0.,0.,1.0]).all():
                            # print('{}-{}'.format(i,j))
                        # else:
                        # numall+=1
                        # print('numall: {}'.format(numall))

                        # if  twintype =='no':
                        #   numall+=1
                        #    print('numall: {}'.format(numall))
                        if twintype != 'no':
                            # isin=False
                            # print(twin_systems[twintype])
                            # if twintype=='Type II':
                            #    numtype2+=1
                            #    print(numtype2)
                            twels = ['K1_a_type', 'eta1_a_type', 'K2_a_type', 'eta2_a_type']
                            twels = ['K1_m_type', 'eta1_m_type']
                            twels = ['K1_a_type', 'eta1_a_type']
                            SymO = [SymOpsR, SymOps]

                            ISIN = False
                            isin = []
                            if len(twin_systems[twintype][twels[0]]) > 0:
                                isin = []
                                for twi in range(0, len(twin_systems[twintype][twels[0]])):
                                    isin.append(False)
                                    alldiff = []
                                    for twel, Sym in zip(twels, SymO):
                                        alldiff.append(0)
                                        for symops in Sym:
                                            twitwel = twin_systems[twintype][twel][twi]
                                            twintwel = twin[twel]
                                            if 'K1' in twel:
                                                twitwel = np.around(twitwel, decimals=K1_arounddigit)
                                                twintwel = np.around(twintwel, decimals=K1_arounddigit)
                                            elif 'eta1' in twel:
                                                twitwel = np.around(twitwel, decimals=eta1_arounddigit)
                                                twintwel = np.around(twintwel, decimals=eta1_arounddigit)

                                            if (abs(twitwel - symops.dot(twintwel)) > 1e-10).any():
                                                # if twintype=='Type I' and len(twin_systems[twintype][twels[0]])==1:
                                                #     print(twin_systems[twintype][twel][twi])
                                                #     print(symops.dot(twin[twel]))
                                                #     print('================================================================================')
                                                alldiff[-1] += 1
                                    if alldiff[0] == len(SymO[0]) or alldiff[1] == len(SymO[0]):
                                        isin[-1] = False
                                    else:
                                        isin[-1] = True
                                    # K1in=[1 for K1Mi in [twin_systems[twintype]['K1_m'][twi]] if (twin['K1_m'] == K1Mi).all() or (-1*twin['K1_m'] == K1Mi).all()]
                                    # E1in=[1 for E1Mi in [twin_systems[twintype]['eta1_m'][twi]] if (twin['eta1_m'] == E1Mi).all() or (-1*twin['eta1_m'] == E1Mi).all()]
                                    # K1E1in = [1 for K1Mi,E1Mi in zip([twin_systems[twintype]['K1_m'][twi]],[twin_systems[twintype]['eta1_m'][twi]])
                                    #                                 if ((twin['K1_m'] == K1Mi).all() and (twin['eta1_m'] == E1Mi).all()) or
                                    #                                 ((-1*twin['K1_m'] == K1Mi).all() and (-1*twin['eta1_m'] == E1Mi).all())]
                                    # print(twin_systems[twintype]['K1_m'][twi])
                                    # print('=================')
                                    # if len(K1in)!=0 and len(E1in)!=0:
                                    # if len(K1E1in)!=0:
                                    #    isin[-1]=True
                                    # if twintype=='Type II':
                                    #    print(twin_systems[twintype]['eta1_m'][twi])
                                    #    print('================================================')

                            keys2list = ['Q_a', 'b_a', 'Rij_a', 'a1_a', 'n1_a', 'a2_a', 'n2_a', 'K1_a', 'eta1_a',
                                         'eta2_a', 'K2_a', 'eta1_m', 'K1_m', 'eta2_m',
                                         'K2_m']  # ,'eta1_a','K1_a']
                            # keys2list=['Q_a','a1_a','n1_a','a2_a','n2_a']
                            try:
                                idx = isin.index(True)
                                # if twintype=='Type II':
                                #    print(twin[twels[0]])
                                #    print(twin_systems[twintype][twels[0]][idx])
                                #    print(twin[twels[1]])
                                #    print(twin_systems[twintype][twels[1]][idx])
                                #    print('=============================================')
                                ISIN = True
                                # twin_systems[twintype]['Variant pairs'][idx].append((i,j))
                                # twin_systems[twintype]['Variant pairs 2'][idx].append((varnotation[i],varnotation[j]))
                                twin_systems[twintype]['Variant pairs'][idx].append((j, i))
                                twin_systems[twintype]['Variant pairs 2'][idx].append((varnotation[j], varnotation[i]))

                                for key in keys2list:
                                    twin_systems[twintype][key][idx].append(twin[key])
                                # twin_systems[twintype]['Q_a'][idx].append(twin['Q_a'])
                                # twin_systems[twintype]['a1_a'][idx].append(twin['a1_a'])
                                # twin_systems[twintype]['n1_a'][idx].append(twin['n1_a'])
                                # twin_systems[twintype]['a2_a'][idx].append(twin['a2_a'])
                                # twin_systems[twintype]['n2_a'][idx].append(twin['n2_a'])
                                # twin_systems[twintype]['vars'].append((i,j))
                                twin_systems[twintype]['vars'].append((j, i))
                            except:
                                pass
                            # ISIN=False

                            if not ISIN:
                                for key in list(twin.keys()):
                                    if key in keys2list:
                                        twin_systems[twintype][key].append([twin[key]])
                                    else:
                                        twin_systems[twintype][key].append(twin[key])
                                # twin_systems[twintype]['vars'].append((i,j))
                                twin_systems[twintype]['vars'].append((j, i))
                                twin_systems[twintype]['uvw2xyz_m'].append(uvw2xyz_m)
                                # twin_systems[twintype]['uvw2xyz_a'].append(uvw2xyz_a)

                                n2eta1K1 = np.cross(twin_systems[twintype]['a1_m'][-1],
                                                    twin_systems[twintype]['n1_m'][-1]);
                                Qrot = np.vstack(
                                    (twin_systems[twintype]['a1_m'][-1], n2eta1K1, twin_systems[twintype]['n1_m'][-1]))
                                Qrot.dot(twin_systems[twintype]['a1_m'][-1])
                                Qri = np.linalg.inv(Qrot)
                                StrainTensor = np.array([[0, 0, twin_systems[twintype]['s'][-1] / 2], [0, 0, 0],
                                                         [twin_systems[twintype]['s'][-1] / 2, 0, 0]])
                                DefGrad = np.array([[1, 0, twin_systems[twintype]['s'][-1]], [0, 1, 0], [0, 0, 1]])

                                twin_systems[twintype]['StrainTensor_m'].append(
                                    np.matmul(np.matmul(Qri, StrainTensor), Qrot))
                                twin_systems[twintype]['F_m'].append(np.matmul(np.matmul(Qri, DefGrad), Qrot))
                                # if twintype=='001':
                                #    print(np.matmul(np.matmul(Qri,DefGrad),Qrot))
                                #    print(Qrot)
                                D, V = np.linalg.eig(twin_systems[twintype]['StrainTensor_m'][-1])
                                Idxs = np.argsort(D)[::-1]
                                Lambda = D[Idxs]
                                V = V[:, Idxs]
                                # if twintype == '001':
                                #    print('ok')
                                twin_systems[twintype]['Tension_m'].append(V[:, 0])
                                twin_systems[twintype]['Compression_m'].append(V[:, 2])
                                # Qa=np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:,:,i])
                                # twin_systems[twintype]['StrainTensor_a'].append(Qa.dot(twin_systems[twintype]['StrainTensor_m'][-1]).dot(Qa.T))
                                # twin_systems[twintype]['Tension_a'].append(Qa.dot(twin_systems[twintype]['Tension_m'][-1]))
                                # twin_systems[twintype]['Compression_a'].append(Qa.dot(twin_systems[twintype]['Compression_m'][-1]))

                                # twin_systems[twintype]['Variant pairs'].append([(i,j)])
                                # twin_systems[twintype]['Variant pairs 2'].append([(varnotation[i],varnotation[j])])
                                twin_systems[twintype]['Variant pairs'].append([(j, i)])
                                twin_systems[twintype]['Variant pairs 2'].append([(varnotation[j], varnotation[i])])
                                # print(twin_systems[twintype]['K1_m'][-1])

                                twin_systems[twintype]['dm_m'].append(
                                    get_twinning_dislocation(twin_systems[twintype]['K1_m'][-1][0],
                                                             twin_systems[twintype]['eta1_m'][-1][0],
                                                             twin_systems[twintype]['eta2_m'][-1][0],
                                                             L_M, G=G_M, Gr=Gr_M))

                            twin_systems[twintype]['uvw2xyz_a'].append(uvw2xyz_a)
                            Qa = np.linalg.inv(Parent_uvw_2_Product_uvw_all_norm[:, :, i])
                            twin_systems[twintype]['StrainTensor_a'].append(
                                Qa.dot(twin_systems[twintype]['StrainTensor_m'][-1]).dot(Qa.T))
                            twin_systems[twintype]['F_a'].append(Qa.dot(twin_systems[twintype]['F_m'][-1]).dot(Qa.T))
                            twin_systems[twintype]['Tension_a'].append(Qa.dot(twin_systems[twintype]['Tension_m'][-1]))
                            twin_systems[twintype]['Compression_a'].append(
                                Qa.dot(twin_systems[twintype]['Compression_m'][-1]))
                            twin_systems[twintype]['dm_a'].append(
                                get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1][0],
                                                         twin_systems[twintype]['eta1_a'][-1][0],
                                                         twin_systems[twintype]['eta2_a'][-1][0],
                                                         L_A, G=G_A, Gr=Gr_A))

                            # deformation twins in martensite
    # 20-1

    twintype = '20-1'
    twin_systems[twintype] = {}
    for key in list(twindata[0].keys()):
        twin_systems[twintype][key] = []
    twin_systems[twintype]['vars'] = []
    twin_systems[twintype]['uvw2xyz_m'] = []
    twin_systems[twintype]['uvw2xyz_a'] = []
    twin_systems[twintype]['StrainTensor_m'] = []
    twin_systems[twintype]['Tension_m'] = []
    twin_systems[twintype]['Compression_m'] = []
    twin_systems[twintype]['Variant pairs'] = []
    twin_systems[twintype]['dm_a'] = []
    twin_systems[twintype]['dm_m'] = []

    twin_systems[twintype]['eta1_m'].append(np.array([-1, 0, -2]))
    twin_systems[twintype]['K1_m'].append(np.array([2, 0, -1]))
    twin_systems[twintype]['eta2_m'].append(np.array([1, 0, 0]))
    twin_systems[twintype]['K2_m'].append(np.array([0, 0, -1]))
    uvw2xyz = Product_uvw2xyz
    hkl2xyz = Product_hkl2xyz
    a1 = uvw2xyz.dot(twin_systems[twintype]['eta1_m'][0])
    twin_systems[twintype]['a1_m'].append(a1 / np.linalg.norm(a1))
    a2 = uvw2xyz.dot(twin_systems[twintype]['eta2_m'][0])
    twin_systems[twintype]['a2_m'].append(a2 / np.linalg.norm(a2))
    n1 = hkl2xyz.dot(twin_systems[twintype]['K1_m'][0])
    twin_systems[twintype]['n1_m'].append(n1 / np.linalg.norm(n1))
    n2 = hkl2xyz.dot(twin_systems[twintype]['K2_m'][0])
    twin_systems[twintype]['n2_m'].append(n2 / np.linalg.norm(n2))

    twin_systems[twintype]['uvw2xyz_m'].append(
        np.matmul((2 * np.outer(twin_systems[twintype]['a1_m'][0], twin_systems[twintype]['a1_m'][0]) - np.eye(3)),
                  uvw2xyz))
    twin_systems[twintype]['R_m'].append(twin_systems[twintype]['uvw2xyz_m'][-1].dot(inv(uvw2xyz)))
    # C1_m=-np.eye(3)+2*np.outer(n1_m,n1_m)
    # twin_systems[twintype]['C_m'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_m'][0],twin_systems[twintype]['n1_m'][0]))
    twin_systems[twintype]['C_m'].append(
        2 * np.outer(twin_systems[twintype]['a1_m'][0], twin_systems[twintype]['a1_m'][0]) - np.eye(3))
    twin_systems[twintype]['shear_angle'].append(
        2 * abs(np.pi / 2 - np.arccos((twin_systems[twintype]['uvw2xyz_m'][0].dot(twin_systems[twintype]['eta2_m'][0])
                                       / norm(
                    twin_systems[twintype]['uvw2xyz_m'][0].dot(twin_systems[twintype]['eta2_m'][0]))).dot(
            twin_systems[twintype]['a1_m'][0]))))
    twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][0] / 2) * 2)

    n2eta1K1 = np.cross(twin_systems[twintype]['a1_m'][-1], twin_systems[twintype]['n1_m'][-1]);
    Qrot = np.vstack((twin_systems[twintype]['a1_m'][-1], n2eta1K1, twin_systems[twintype]['n1_m'][-1]))
    Qrot.dot(twin_systems[twintype]['a1_m'][-1])
    Qri = np.linalg.inv(Qrot)
    StrainTensor = np.array(
        [[0, 0, twin_systems[twintype]['s'][-1] / 2], [0, 0, 0], [twin_systems[twintype]['s'][-1] / 2, 0, 0]])
    twin_systems[twintype]['StrainTensor_m'].append(np.matmul(np.matmul(Qri, StrainTensor), Qrot))

    D, V = np.linalg.eig(twin_systems[twintype]['StrainTensor_m'][-1])
    Idxs = np.argsort(D)
    Lambda = D[Idxs]
    V = V[:, Idxs]
    twin_systems[twintype]['Tension_m'].append(V[:, 0])
    twin_systems[twintype]['Compression_m'].append(V[:, 2])
    twin_systems[twintype]['dm_m'].append(get_twinning_dislocation(twin_systems[twintype]['K1_m'][-1],
                                                                   twin_systems[twintype]['eta1_m'][-1],
                                                                   twin_systems[twintype]['eta2_m'][-1],
                                                                   L_M, G=G_M, Gr=Gr_M))

    for symhkl, symuvw in zip(B19p.reciprocal_symmetry_operations(), B19p.symmetry_operations()):
        K1 = np.round(symhkl[0:3, 0:3].dot(twin_systems[twintype]['K1_m'][0]), 5)
        K2 = np.round(symhkl[0:3, 0:3].dot(twin_systems[twintype]['K2_m'][0]), 5)
        eta1 = np.round(symuvw[0:3, 0:3].dot(twin_systems[twintype]['eta1_m'][0]), 5)
        eta2 = np.round(symuvw[0:3, 0:3].dot(twin_systems[twintype]['eta2_m'][0]), 5)
        isin = False
        for i in range(0, len(twin_systems[twintype]['eta1_m'])):
            if (twin_systems[twintype]['eta1_m'][i] == eta1).all():
                if (twin_systems[twintype]['eta2_m'][i] == eta2).all():
                    if (twin_systems[twintype]['K1_m'][i] == K1).all():
                        if (twin_systems[twintype]['K2_m'][i] == K2).all():
                            isin = True
        K1in = [1 for K1Mi in twin_systems[twintype]['K1_m'] if (K1 == K1Mi).all() or (-1 * K1 == K1Mi).all()]
        E1in = [1 for E1Mi in twin_systems[twintype]['eta1_m'] if (eta1 == E1Mi).all() or (-1 * eta1 == E1Mi).all()]
        if len(K1in) == 0 or len(E1in) == 0:
            #        if not isin:
            twin_systems[twintype]['eta1_m'].append(eta1)
            twin_systems[twintype]['K1_m'].append(K1)
            twin_systems[twintype]['eta2_m'].append(eta2)
            twin_systems[twintype]['K2_m'].append(K2)
            a1 = uvw2xyz.dot(twin_systems[twintype]['eta1_m'][-1])
            twin_systems[twintype]['a1_m'].append(a1 / np.linalg.norm(a1))
            a2 = uvw2xyz.dot(twin_systems[twintype]['eta2_m'][-1])
            twin_systems[twintype]['a2_m'].append(a2 / np.linalg.norm(a2))
            n1 = hkl2xyz.dot(twin_systems[twintype]['K1_m'][-1])
            twin_systems[twintype]['n1_m'].append(n1 / np.linalg.norm(n1))
            n2 = hkl2xyz.dot(twin_systems[twintype]['K2_m'][-1])
            twin_systems[twintype]['n2_m'].append(n2 / np.linalg.norm(n2))

            twin_systems[twintype]['uvw2xyz_m'].append(np.matmul(
                (2 * np.outer(twin_systems[twintype]['a1_m'][-1], twin_systems[twintype]['a1_m'][-1]) - np.eye(3)),
                uvw2xyz))
            twin_systems[twintype]['R_m'].append(twin_systems[twintype]['uvw2xyz_m'][-1].dot(inv(uvw2xyz)))

            # C1_m=-np.eye(3)+2*np.outer(n1_m,n1_m)
            # twin_systems[twintype]['C_m'].append(-np.eye(3)+2*np.outer(twin_systems[twintype]['n1_m'][0],twin_systems[twintype]['n1_m'][0]))
            twin_systems[twintype]['C_m'].append(
                2 * np.outer(twin_systems[twintype]['a1_m'][-1], twin_systems[twintype]['a1_m'][-1]) - np.eye(3))
            twin_systems[twintype]['shear_angle'].append(2 * abs(
                np.pi / 2 - np.arccos((twin_systems[twintype]['uvw2xyz_m'][-1].dot(twin_systems[twintype]['eta2_m'][-1])
                                       / norm(
                            twin_systems[twintype]['uvw2xyz_m'][-1].dot(twin_systems[twintype]['eta2_m'][-1]))).dot(
                    twin_systems[twintype]['a1_m'][-1]))))
            twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][-1] / 2) * 2)
            n2eta1K1 = np.cross(twin_systems[twintype]['a1_m'][-1], twin_systems[twintype]['n1_m'][-1]);
            Qrot = np.vstack((twin_systems[twintype]['a1_m'][-1], n2eta1K1, twin_systems[twintype]['n1_m'][-1]))
            Qrot.dot(twin_systems[twintype]['a1_m'][-1])
            Qri = np.linalg.inv(Qrot)
            StrainTensor = np.array(
                [[0, 0, twin_systems[twintype]['s'][-1] / 2], [0, 0, 0], [twin_systems[twintype]['s'][-1] / 2, 0, 0]])
            twin_systems[twintype]['StrainTensor_m'].append(np.matmul(np.matmul(Qri, StrainTensor), Qrot))

            D, V = np.linalg.eig(twin_systems[twintype]['StrainTensor_m'][-1])
            Idxs = np.argsort(D)
            Lambda = D[Idxs]
            V = V[:, Idxs]
            twin_systems[twintype]['Tension_m'].append(V[:, 0])
            twin_systems[twintype]['Compression_m'].append(V[:, 2])

            twin_systems[twintype]['dm_m'].append(get_twinning_dislocation(twin_systems[twintype]['K1_m'][-1],
                                                                           twin_systems[twintype]['eta1_m'][-1],
                                                                           twin_systems[twintype]['eta2_m'][-1],
                                                                           L_M, G=G_M, Gr=Gr_M))

    # deformation twins in austenite
    # 114
    twintype = '114'
    twin_systems[twintype] = {}
    for key in list(twindata[0].keys()):
        twin_systems[twintype][key] = []
    twin_systems[twintype]['vars'] = []
    twin_systems[twintype]['uvw2xyz_m'] = []
    twin_systems[twintype]['uvw2xyz_a'] = []
    twin_systems[twintype]['StrainTensor_a'] = []
    twin_systems[twintype]['Tension_a'] = []
    twin_systems[twintype]['Compression_a'] = []
    twin_systems[twintype]['Variant pairs'] = []
    twin_systems[twintype]['dm_a'] = []
    twin_systems[twintype]['dm_m'] = []

    # twin_systems[twintype]['eta1_a'].append([-1,-2,-2])
    # twin_systems[twintype]['K1_a'].append([-1,-1,4])
    # twin_systems[twintype]['eta2_a'].append([0,0,1])
    # twin_systems[twintype]['K2_a'].append([1,1,0])

    twin_systems[twintype]['eta1_a'].append([1, 2, 2])
    twin_systems[twintype]['K1_a'].append([-4, 1, 1])
    twin_systems[twintype]['eta2_a'].append([-1, 0, 0])
    twin_systems[twintype]['K2_a'].append([0, 1, 1])
    uvw2xyz = Parent_uvw2xyz
    hkl2xyz = Parent_hkl2xyz
    a1 = uvw2xyz.dot(twin_systems[twintype]['eta1_a'][0])
    twin_systems[twintype]['a1_a'].append(a1 / np.linalg.norm(a1))
    a2 = uvw2xyz.dot(twin_systems[twintype]['eta2_a'][0])
    twin_systems[twintype]['a2_a'].append(a2 / np.linalg.norm(a2))
    n1 = hkl2xyz.dot(twin_systems[twintype]['K1_a'][0])
    twin_systems[twintype]['n1_a'].append(n1 / np.linalg.norm(n1))
    n2 = hkl2xyz.dot(twin_systems[twintype]['K2_a'][0])
    twin_systems[twintype]['n2_a'].append(n2 / np.linalg.norm(n2))

    twin_systems[twintype]['uvw2xyz_a'].append(
        np.matmul((2 * np.outer(twin_systems[twintype]['a1_a'][0], twin_systems[twintype]['a1_a'][0]) - np.eye(3)),
                  uvw2xyz))

    # twin_systems[twintype]['R_a'].append(twin_systems[twintype]['uvw2xyz_a'][-1].dot(inv(uvw2xyz)))
    # C1_m=-np.eye(3)+2*np.outer(n1_m,n1_m)
    # np.eye(3)-2*np.outer(n1_m,n1_m)
    # twin_systems[twintype]['C_a'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_a'][0],twin_systems[twintype]['n1_a'][0]))
    twin_systems[twintype]['C_a'].append(
        2 * np.outer(twin_systems[twintype]['a1_a'][0], twin_systems[twintype]['a1_a'][0]) - np.eye(3))

    twin_systems[twintype]['shear_angle'].append(
        2 * abs(np.pi / 2 - np.arccos((twin_systems[twintype]['uvw2xyz_a'][0].dot(twin_systems[twintype]['eta2_a'][0])
                                       / norm(
                    twin_systems[twintype]['uvw2xyz_a'][0].dot(twin_systems[twintype]['eta2_a'][0]))).dot(
            twin_systems[twintype]['a1_a'][0]))))
    twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][0] / 2) * 2)

    n2eta1K1 = np.cross(twin_systems[twintype]['a1_a'][-1], twin_systems[twintype]['n1_a'][-1]);
    Qrot = np.vstack((twin_systems[twintype]['a1_a'][-1], n2eta1K1, twin_systems[twintype]['n1_a'][-1]))
    Qrot.dot(twin_systems[twintype]['a1_a'][-1])
    Qri = np.linalg.inv(Qrot)
    StrainTensor = np.array(
        [[0, 0, twin_systems[twintype]['s'][-1] / 2], [0, 0, 0], [twin_systems[twintype]['s'][-1] / 2, 0, 0]])
    twin_systems[twintype]['StrainTensor_a'].append(np.matmul(np.matmul(Qri, StrainTensor), Qrot))

    D, V = np.linalg.eig(twin_systems[twintype]['StrainTensor_a'][-1])
    Idxs = np.argsort(D)[::-1]
    Lambda = D[Idxs]
    V = V[:, Idxs]
    twin_systems[twintype]['Tension_a'].append(V[:, 0])
    twin_systems[twintype]['Compression_a'].append(V[:, 2])

    twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1],
                                                                   twin_systems[twintype]['eta1_a'][-1],
                                                                   twin_systems[twintype]['eta2_a'][-1],
                                                                   L_A, G=G_A, Gr=Gr_A))

    for symhkl, symuvw in zip(B2.reciprocal_symmetry_operations(), B2.symmetry_operations()):
        K1 = np.round(symhkl[0:3, 0:3].dot(twin_systems[twintype]['K1_a'][0]), 5)
        K2 = np.round(symhkl[0:3, 0:3].dot(twin_systems[twintype]['K2_a'][0]), 5)
        eta1 = np.round(symuvw[0:3, 0:3].dot(twin_systems[twintype]['eta1_a'][0]), 5)
        eta2 = np.round(symuvw[0:3, 0:3].dot(twin_systems[twintype]['eta2_a'][0]), 5)
        isin = False
        for i in range(0, len(twin_systems[twintype]['eta1_a'])):
            if (twin_systems[twintype]['eta1_a'][i] == eta1).all():
                if (twin_systems[twintype]['eta2_a'][i] == eta2).all():
                    if (twin_systems[twintype]['K1_a'][i] == K1).all():
                        if (twin_systems[twintype]['K2_a'][i] == K2).all():
                            isin = True
        if not isin:
            K1in = [1 for K1Mi in twin_systems[twintype]['K1_a'] if (K1 == K1Mi).all() or (-1 * K1 == K1Mi).all()]
            E1in = [1 for E1Mi in twin_systems[twintype]['eta1_a'] if (eta1 == E1Mi).all() or (-1 * eta1 == E1Mi).all()]
            if len(K1in) == 0 or len(E1in) == 0:
                twin_systems[twintype]['eta1_a'].append(eta1)
                twin_systems[twintype]['K1_a'].append(K1)
                twin_systems[twintype]['eta2_a'].append(eta2)
                twin_systems[twintype]['K2_a'].append(K2)
                a1 = uvw2xyz.dot(twin_systems[twintype]['eta1_a'][-1])
                twin_systems[twintype]['a1_a'].append(a1 / np.linalg.norm(a1))
                a2 = uvw2xyz.dot(twin_systems[twintype]['eta2_a'][-1])
                twin_systems[twintype]['a2_a'].append(a2 / np.linalg.norm(a2))
                n1 = uvw2xyz.dot(twin_systems[twintype]['K1_a'][-1])
                twin_systems[twintype]['n1_a'].append(n1 / np.linalg.norm(n1))
                n2 = uvw2xyz.dot(twin_systems[twintype]['K2_a'][-1])
                twin_systems[twintype]['n2_a'].append(n2 / np.linalg.norm(n2))

                twin_systems[twintype]['uvw2xyz_a'].append(np.matmul(
                    (2 * np.outer(twin_systems[twintype]['a1_a'][-1], twin_systems[twintype]['a1_a'][-1]) - np.eye(3)),
                    uvw2xyz))
                # twin_systems[twintype]['R_a'].append(twin_systems[twintype]['uvw2xyz_a'][-1].dot(inv(uvw2xyz)))
                # twin_systems[twintype]['C_a'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_a'][-1],twin_systems[twintype]['n1_a'][-1]))
                twin_systems[twintype]['C_a'].append(
                    2 * np.outer(twin_systems[twintype]['a1_a'][-1], twin_systems[twintype]['a1_a'][-1]) - np.eye(3))
                twin_systems[twintype]['shear_angle'].append(2 * abs(np.pi / 2 - np.arccos(
                    (twin_systems[twintype]['uvw2xyz_a'][-1].dot(twin_systems[twintype]['eta2_a'][-1])
                     / norm(twin_systems[twintype]['uvw2xyz_a'][-1].dot(twin_systems[twintype]['eta2_a'][-1]))).dot(
                        twin_systems[twintype]['a1_a'][-1]))))
                twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][-1] / 2) * 2)

                n2eta1K1 = np.cross(twin_systems[twintype]['a1_a'][-1], twin_systems[twintype]['n1_a'][-1]);
                Qrot = np.vstack((twin_systems[twintype]['a1_a'][-1], n2eta1K1, twin_systems[twintype]['n1_a'][-1]))
                Qrot.dot(twin_systems[twintype]['a1_a'][-1])
                Qri = np.linalg.inv(Qrot)
                StrainTensor = np.array([[0, 0, twin_systems[twintype]['s'][-1] / 2], [0, 0, 0],
                                         [twin_systems[twintype]['s'][-1] / 2, 0, 0]])
                twin_systems[twintype]['StrainTensor_a'].append(np.matmul(np.matmul(Qri, StrainTensor), Qrot))

                D, V = np.linalg.eig(twin_systems[twintype]['StrainTensor_a'][-1])
                Idxs = np.argsort(D)[::-1]
                Lambda = D[Idxs]
                V = V[:, Idxs]
                twin_systems[twintype]['Tension_a'].append(V[:, 0])
                twin_systems[twintype]['Compression_a'].append(V[:, 2])

                twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1],
                                                                               twin_systems[twintype]['eta1_a'][-1],
                                                                               twin_systems[twintype]['eta2_a'][-1],
                                                                               L_A, G=G_A, Gr=Gr_A))

    # 114
    twintypes = ['112']
    eta1s_a = [[1, 1, 1]]
    K1s_a = [[-2, 1, 1]]
    eta2s_a = [[-1, 0, 0]]
    K2s_a = [[0, 1, 1]]

    twintypes.append('115')
    eta1s_a.append([2, 5, 5])
    K1s_a.append([-5, 1, 1])
    eta2s_a.append([-1, 0, 0])
    K2s_a.append([0, 1, 1])

    twintypes.append('111')
    eta1s_a.append([-1, -1, -2])
    K1s_a.append([-1, -1, 1])
    eta2s_a.append([-1, -1, 2])
    K2s_a.append([-1, -1, -1])

    for twintype, eta1_a, K1_a, eta2_a, K2_a in zip(twintypes, eta1s_a, K1s_a, eta2s_a, K2s_a):
        twin_systems[twintype] = {}
        for key in list(twindata[0].keys()):
            twin_systems[twintype][key] = []
        twin_systems[twintype]['vars'] = []
        twin_systems[twintype]['uvw2xyz_m'] = []
        twin_systems[twintype]['uvw2xyz_a'] = []
        twin_systems[twintype]['StrainTensor_a'] = []
        twin_systems[twintype]['Tension_a'] = []
        twin_systems[twintype]['Compression_a'] = []
        twin_systems[twintype]['Variant pairs'] = []
        twin_systems[twintype]['dm_a'] = []
        twin_systems[twintype]['dm_m'] = []

        # twin_systems[twintype]['eta1_a'].append([-1,-2,-2])
        # twin_systems[twintype]['K1_a'].append([-1,-1,4])
        # twin_systems[twintype]['eta2_a'].append([0,0,1])
        # twin_systems[twintype]['K2_a'].append([1,1,0])

        twin_systems[twintype]['eta1_a'].append(eta1_a)
        twin_systems[twintype]['K1_a'].append(K1_a)
        twin_systems[twintype]['eta2_a'].append(eta2_a)
        twin_systems[twintype]['K2_a'].append(K2_a)
        uvw2xyz = Parent_uvw2xyz
        hkl2xyz = Parent_hkl2xyz
        a1 = uvw2xyz.dot(twin_systems[twintype]['eta1_a'][0])
        twin_systems[twintype]['a1_a'].append(a1 / np.linalg.norm(a1))
        a2 = uvw2xyz.dot(twin_systems[twintype]['eta2_a'][0])
        twin_systems[twintype]['a2_a'].append(a2 / np.linalg.norm(a2))
        n1 = hkl2xyz.dot(twin_systems[twintype]['K1_a'][0])
        twin_systems[twintype]['n1_a'].append(n1 / np.linalg.norm(n1))
        n2 = hkl2xyz.dot(twin_systems[twintype]['K2_a'][0])
        twin_systems[twintype]['n2_a'].append(n2 / np.linalg.norm(n2))

        twin_systems[twintype]['uvw2xyz_a'].append(
            np.matmul((2 * np.outer(twin_systems[twintype]['a1_a'][0], twin_systems[twintype]['a1_a'][0]) - np.eye(3)),
                      uvw2xyz))
        # twin_systems[twintype]['R_a'].append(twin_systems[twintype]['uvw2xyz_a'][-1].dot(inv(uvw2xyz)))
        # C1_m=-np.eye(3)+2*np.outer(n1_m,n1_m)
        # np.eye(3)-2*np.outer(n1_m,n1_m)
        # twin_systems[twintype]['C_a'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_a'][0],twin_systems[twintype]['n1_a'][0]))
        twin_systems[twintype]['C_a'].append(
            2 * np.outer(twin_systems[twintype]['a1_a'][0], twin_systems[twintype]['a1_a'][0]) - np.eye(3))
        twin_systems[twintype]['shear_angle'].append(2 * abs(
            np.pi / 2 - np.arccos((twin_systems[twintype]['uvw2xyz_a'][0].dot(twin_systems[twintype]['eta2_a'][0])
                                   / norm(
                        twin_systems[twintype]['uvw2xyz_a'][0].dot(twin_systems[twintype]['eta2_a'][0]))).dot(
                twin_systems[twintype]['a1_a'][0]))))
        twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][0] / 2) * 2)

        n2eta1K1 = np.cross(twin_systems[twintype]['a1_a'][-1], twin_systems[twintype]['n1_a'][-1]);
        Qrot = np.vstack((twin_systems[twintype]['a1_a'][-1], n2eta1K1, twin_systems[twintype]['n1_a'][-1]))
        Qrot.dot(twin_systems[twintype]['a1_a'][-1])
        Qri = np.linalg.inv(Qrot)
        StrainTensor = np.array(
            [[0, 0, twin_systems[twintype]['s'][-1] / 2], [0, 0, 0], [twin_systems[twintype]['s'][-1] / 2, 0, 0]])
        twin_systems[twintype]['StrainTensor_a'].append(np.matmul(np.matmul(Qri, StrainTensor), Qrot))

        D, V = np.linalg.eig(twin_systems[twintype]['StrainTensor_a'][-1])
        Idxs = np.argsort(D)[::-1]
        Lambda = D[Idxs]
        V = V[:, Idxs]
        twin_systems[twintype]['Tension_a'].append(V[:, 0])
        twin_systems[twintype]['Compression_a'].append(V[:, 2])

        twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1],
                                                                       twin_systems[twintype]['eta1_a'][-1],
                                                                       twin_systems[twintype]['eta2_a'][-1],
                                                                       L_A, G=G_A, Gr=Gr_A))

        # print(twin_systems[twintype]['dm_a'])
        # print(twin_systems[twintype]['eta2_a'][-1])
        # print(G_A)

        for symhkl, symuvw in zip(B2.reciprocal_symmetry_operations(), B2.symmetry_operations()):
            K1 = np.round(symhkl[0:3, 0:3].dot(twin_systems[twintype]['K1_a'][0]), 5)
            K2 = np.round(symhkl[0:3, 0:3].dot(twin_systems[twintype]['K2_a'][0]), 5)
            eta1 = np.round(symuvw[0:3, 0:3].dot(twin_systems[twintype]['eta1_a'][0]), 5)
            eta2 = np.round(symuvw[0:3, 0:3].dot(twin_systems[twintype]['eta2_a'][0]), 5)
            isin = False
            for i in range(0, len(twin_systems[twintype]['eta1_a'])):
                if (twin_systems[twintype]['eta1_a'][i] == eta1).all():
                    if (twin_systems[twintype]['eta2_a'][i] == eta2).all():
                        if (twin_systems[twintype]['K1_a'][i] == K1).all():
                            if (twin_systems[twintype]['K2_a'][i] == K2).all():
                                isin = True
            if not isin:
                K1in = [1 for K1Mi in twin_systems[twintype]['K1_a'] if (K1 == K1Mi).all() or (-1 * K1 == K1Mi).all()]
                E1in = [1 for E1Mi in twin_systems[twintype]['eta1_a'] if
                        (eta1 == E1Mi).all() or (-1 * eta1 == E1Mi).all()]
                if len(K1in) == 0 or len(E1in) == 0:
                    twin_systems[twintype]['eta1_a'].append(eta1)
                    twin_systems[twintype]['K1_a'].append(K1)
                    twin_systems[twintype]['eta2_a'].append(eta2)
                    twin_systems[twintype]['K2_a'].append(K2)
                    a1 = uvw2xyz.dot(twin_systems[twintype]['eta1_a'][-1])
                    twin_systems[twintype]['a1_a'].append(a1 / np.linalg.norm(a1))
                    a2 = uvw2xyz.dot(twin_systems[twintype]['eta2_a'][-1])
                    twin_systems[twintype]['a2_a'].append(a2 / np.linalg.norm(a2))
                    n1 = uvw2xyz.dot(twin_systems[twintype]['K1_a'][-1])
                    twin_systems[twintype]['n1_a'].append(n1 / np.linalg.norm(n1))
                    n2 = uvw2xyz.dot(twin_systems[twintype]['K2_a'][-1])
                    twin_systems[twintype]['n2_a'].append(n2 / np.linalg.norm(n2))

                    twin_systems[twintype]['uvw2xyz_a'].append(np.matmul((2 * np.outer(
                        twin_systems[twintype]['a1_a'][-1], twin_systems[twintype]['a1_a'][-1]) - np.eye(3)), uvw2xyz))
                    # twin_systems[twintype]['R_a'].append(twin_systems[twintype]['uvw2xyz_a'][-1].dot(inv(uvw2xyz)))
                    # twin_systems[twintype]['C_a'].append(np.eye(3)-2*np.outer(twin_systems[twintype]['n1_a'][-1],twin_systems[twintype]['n1_a'][-1]))
                    twin_systems[twintype]['C_a'].append(
                        2 * np.outer(twin_systems[twintype]['a1_a'][-1], twin_systems[twintype]['a1_a'][-1]) - np.eye(
                            3))
                    twin_systems[twintype]['shear_angle'].append(2 * abs(np.pi / 2 - np.arccos(
                        (twin_systems[twintype]['uvw2xyz_a'][-1].dot(twin_systems[twintype]['eta2_a'][-1])
                         / norm(twin_systems[twintype]['uvw2xyz_a'][-1].dot(twin_systems[twintype]['eta2_a'][-1]))).dot(
                            twin_systems[twintype]['a1_a'][-1]))))
                    twin_systems[twintype]['s'].append(np.tan(twin_systems[twintype]['shear_angle'][-1] / 2) * 2)

                    n2eta1K1 = np.cross(twin_systems[twintype]['a1_a'][-1], twin_systems[twintype]['n1_a'][-1]);
                    Qrot = np.vstack((twin_systems[twintype]['a1_a'][-1], n2eta1K1, twin_systems[twintype]['n1_a'][-1]))
                    Qrot.dot(twin_systems[twintype]['a1_a'][-1])
                    Qri = np.linalg.inv(Qrot)
                    StrainTensor = np.array([[0, 0, twin_systems[twintype]['s'][-1] / 2], [0, 0, 0],
                                             [twin_systems[twintype]['s'][-1] / 2, 0, 0]])
                    twin_systems[twintype]['StrainTensor_a'].append(np.matmul(np.matmul(Qri, StrainTensor), Qrot))

                    D, V = np.linalg.eig(twin_systems[twintype]['StrainTensor_a'][-1])
                    Idxs = np.argsort(D)[::-1]
                    Lambda = D[Idxs]
                    V = V[:, Idxs]
                    twin_systems[twintype]['Tension_a'].append(V[:, 0])
                    twin_systems[twintype]['Compression_a'].append(V[:, 2])

                    twin_systems[twintype]['dm_a'].append(get_twinning_dislocation(twin_systems[twintype]['K1_a'][-1],
                                                                                   twin_systems[twintype]['eta1_a'][-1],
                                                                                   twin_systems[twintype]['eta2_a'][-1],
                                                                                   L_A, G=G_A, Gr=Gr_A))

    # print('test')
    return twin_systems


def get_twinning_dislocation(K1, eta1, eta2, L, G=None, Gr=None):
    # Checked for cubic lattice and K1=-1-11,eta1=-1-1-2,eta2=-1-12 =1/6
    # https://www.sciencedirect.com/science/article/pii/S0966979514000892?via%3Dihub
    # L -latice tensor converting uvw->xyz
    # if type(K1) is list:
    # K1=K1[0]
    #    print(K1)
    # if type(eta1) is list:
    #    eta1=eta1[0]
    # if type(eta2) is list:
    #    eta2=eta2[0]
    # print(K1)
    # print(eta1)
    # print(eta2)
    if True:
        if G is None:
            # Meteric tensor ||[x,y,x]||^2=[uvw]*G_A*[uvw]^T
            G = np.matmul(L.T, L)
        if Gr is None:
            # Reciprocal Meteric tensor ||[x,y,x]||^2=d_hkl^2=1/[hkl]*Gr_A*[hkl]^T
            Gr = np.linalg.inv(G)
        if type(K1) is not np.ndarray:
            K1 = np.array(K1)
        if type(eta1) is not np.ndarray:
            eta1 = np.array(eta1)
        if type(eta2) is not np.ndarray:
            eta2 = np.array(eta2)
            # rotation angle of the twinning = shear is defined as 2* tan(shear angle/2)
        shear_angle = 2 * abs(np.pi / 2 - np.arccos(
            (L.dot(eta2) / np.linalg.norm(L.dot(eta2))).dot(L.dot(eta1) / np.linalg.norm(L.dot(eta1)))))
        # print(shear_angle
        # d-spacing of twinning plane K1
        dK1 = 1 / np.sqrt(K1.dot(Gr).dot(K1))
        # length of eta1
        deta1 = np.sqrt(eta1.dot(G).dot(eta1))
        # slip of the firs K1 plane above the twinning plane
        s = 2 * np.tan(shear_angle / 2) * dK1
        # twinning dislocation magnitude as fraction of eta1 (dm*eta1)
        dm = s / deta1
    return dm


def twin_equation_solution(Uj, Ui, L_A, Lr_A, L_M, Lr_M,
                           R_AM, Ci_d, Ci_p, tol=1e-10, miller='greaterthanone', printlambda=False,
                           Qj=None, Qi=None):
    # Uj transformation stretches of a possible twinned variant related to corresponding transformation gradient Fj=Qj*Uj
    # Ui transformation matrix of the matrix variant related to corresponding transformation gradient Fi=Qi*Ui
    # Qj rotational part of the transformation gradien Fj=Qj*Uj
    # Qi rotational part of the transformation gradien Fi=Qi*Ui
    # L_A/L_M matrix converting UVW->XYZ for parent phase, austenite/product phase, martensite
    # Lr_A/Lr_M matrix converting HKL->XYZ for parent phase, austenite/product phase, martensite
    # R_AM rotation matrix rotating the reference space of the austenite phase aligned with basal directions x=[1,0,0],y=[0,1,0]
    # into the space where x,y,z are aligned with austenite directions corresponding to martensite directions [1,0,0]m,[0,1,0]m,[0,0,1]m of variant i.
    # Basically R_AM.dot(inv(Ci_d))= diagonal matrix
    # Ci_d lattice correspondence for directions converting UVW of austenite with lattice corresponding UVW of martensite
    # Ci_p lattice correspondence for planes converting HKL of austenite with lattice corresponding HKL of martensite

    from numpy import matmul
    from numpy import sqrt
    from numpy.linalg import inv
    from scipy.linalg import sqrtm
    from numpy.linalg import norm
    if miller == 'greaterthanone':
        MIN = True
    else:
        MIN = False
    # Solves equation Q*Uj-Ui=axn for stretches so the reference space Ui rotated Qi^(-1) with respect to real parent space
    # C=matmul(matmul(inv(Ui).T,Uj.T),matmul(Uj,inv(Ui)))

    # C=matmul(matmul(Uj,inv(Ui)).T,matmul(Uj,inv(Ui)))
    # print('ok')
    C = np.linalg.multi_dot([inv(Ui.T), Uj.T, Uj, inv(Ui)])

    D, V = np.linalg.eig(C)
    # V[:,1]=-1*V[:,1]

    matmul(C, V) - matmul(V, D * np.eye(3))

    # sorting according to eigenvalues Lambda[0]<Lambda[1]<Lambda[2]
    Idxs = np.argsort(D)

    Lambda = D[Idxs]
    # print(Lambda)
    if printlambda:
        print(Lambda)
    V = V[:, Idxs]

    twindata = {}
    keys = ['shear_angle', 's', 'b_a', 'Rij_a', 'k', 'Type', 'n_a', 'a_a', 'Q_a', 'eta_a', 'K_a', 'eta_a_type',
            'K_a_type', 'n_m', 'a_m', 'Q_m', 'R_m', 'eta_m', 'K_m', 'eta_m_type', 'K_m_type', 'C_m', 'C_a']
    for key in keys:
        twindata[key] = []

    TWINDATA = []
    # print(Lambda[2]-Lambda[1])
    # solution exists if Lambda[0]<1,Lambda[1]=1,Lambda[2]>1
    if Lambda[0] < 1 and np.abs(Lambda[1] - 1) < tol and Lambda[2] > 1:

        # solution in the frame where Ui Uj are defioned
        # solutions are in terms of
        # twinning shear direction a
        # twinning plane normal n
        # rotation Q so that Q*Uj-Ui=axn
        # 180 twinning rotation R so that  R1_a*Ui*R1_a.T=Uj, R1_a*Ui*R1_a.T=Uj

        for k in [-1, 1]:
            twind = {}

            n1_a = (sqrt(Lambda[2]) - sqrt(Lambda[0])) / sqrt(Lambda[2] - Lambda[0]) * (
                    -1 * sqrt(1 - Lambda[0]) * Ui.T.dot(V[:, 0]) + k * sqrt(Lambda[2] - 1) * Ui.T.dot(V[:, 2]));
            rho = 1 * norm(n1_a)
            n1_a = 1 / rho * n1_a
            a1_a = rho * (sqrt(Lambda[2] * (1 - Lambda[0]) / (Lambda[2] - Lambda[0])) * V[:, 0] + k * sqrt(
                Lambda[0] * (Lambda[2] - 1) / (Lambda[2] - Lambda[0])) * V[:, 2]);

            # n1_a is twinning normal undistorted by transformation!! It is real normal but only lattice corresponding to martensite twinning normal
            n1_a, c = flipvector(n1_a);

            # a1_a is real twinning direction in parent phase but with subtracted
            a1_a = c * a1_a;

            twind[
                'n_a'] = n1_a  # this is twinning normal undistorted by transformation!! It is real normal lattice corresponding to martensite twinning normal
            twind['a_a'] = a1_a  # This is real twinning direction rotated Qi^(-1) with respect to real parent space
            twind['k'] = k  # auxiliary

            # np.linalg.inv(Ui).dot(a1_a)
            eta1_a = vector2miller(np.linalg.inv(L_A).dot(np.linalg.inv(Ui).dot(a1_a)),
                                   MIN=MIN)  ##this is twinning direction undistorted by
            # transformation!! See np.linalg.inv(Ui).dot(a1_a). It is real twinning shear direction lattice corresponding to martensite twinning direction

            # eta1_a=vector2miller(np.linalg.inv(L_A).dot(a1_a))
            twind['eta_a'] = eta1_a
            twind['eta_a_type'] = eta1_a
            K1_a = vector2miller(inv(Lr_A).dot(n1_a),
                                 MIN=MIN)  # Miller indexes of twinning normal undistorted by transformation!! It is HKL lattice corresponding to martensite twinning normal
            twind['K_a'] = K1_a
            twind['K_a_type'] = K1_a

            # now for martensite
            # variant A based on lattice correspondence
            c = 1.0
            eta1_m = vector2miller(Ci_d.dot(eta1_a),
                                   MIN=MIN)  # UVW of twinning direction in the martensite system
            K1_m = vector2miller(Ci_p.dot(K1_a),
                                 MIN=MIN)  # HKL of twinning direction in the martensite system
            twind['eta_m'] = c * eta1_m
            twind['K_m'] = c * K1_m

            twind['eta_m_type'] = c * eta1_m
            twind['K_m_type'] = c * K1_m

            n1_m = Lr_M.dot(K1_m)  # Real space of twinning normal in the martensite system
            a1_m = L_M.dot(eta1_m)  # Real space twinning direction in the martensite system
            n1_m = n1_m / norm(n1_m)
            a1_m = a1_m / norm(a1_m)
            twind['n_m'] = n1_m
            twind['a_m'] = a1_m

            # shear of the twinning
            s1 = norm(a1_a) * norm(inv(Ui).dot(n1_a))
            twind['s'] = s1
            twind['shear_angle'] = np.arctan(s1 / 2) * 2

            # rigid body rotation necessary to apply to streteches of j variant to make twin with variant i
            # however there are other rotations Qj Qi already subtracted from deformation gradients Fj, Fi.
            # therefore complete rotation Rij_a is calculated below if Qj, Qi provided
            Q1_a = matmul(np.outer(a1_a, n1_a) + Ui, inv(Uj))

            # (matmul(Q1_a,Uj)-Ui)-np.outer(a1_a,n1_a)
            # R1_a=-np.eye(3)+2*np.outer(a1_a,a1_a)

            twind['Q_a'] = Q1_a
            # twind['R_a']=R1_a
            if Qj is not None and Qi is not None:
                # this is real twinning shear direction accounting for the subtracted rotation
                twind['b_a'] = Qi.dot(twind['a_a'])
                twind['Rij_a'] = Qi.dot(twind['Q_a']).dot(Qj.T)
                # print(twind['Rij_a'])
                # the other way to reach twinning elements in martensite
                # get real twinning normal (distorted by transformation)
                n_m = twind['n_a'].dot(np.linalg.inv(Qi.dot(Ui)))
                # rotate it into martensite frame, normalize, convert to HKL and make miller
                n_m = R_AM.dot(n_m)
                n_m = n_m / norm(n_m)
                twind['n_m'] = n_m
                K_m = vector2miller(inv(Lr_M).dot(n_m), MIN=MIN)
                twind['K_m'] = K_m
                twind['K_m_type'] = K_m
                # eta_m corresponds to real twinning shear direction
                a_m = Qi.dot(twind['a_a'])
                a_m = R_AM.dot(a_m)
                a_m /= norm(a_m)
                twind['a_m'] = a_m
                eta_m = vector2miller(inv(L_M).dot(a_m), MIN=MIN)
                twind['eta_m'] = eta_m
                twind['eta_m_type'] = eta_m
            else:
                twind['b_a'] = None
                twind['Rij_a'] = None

            # change of lattice directions in the twin
            # this is passive and active, it is symmetric since it is 180 deg rotation
            # if vector v lies in the twinning plane np.outer(n1_m,n1_m) is zeros and directions change signs C1_m*v=-I*v+0, the signes along n1 dop not change sign
            if (K1_m != np.round(K1_m)).any() and (eta1_m == np.round(eta1_m)).all():
                # for Type II twins - eta1 is preserved - rotation around eta1
                C1_m = -np.eye(3) + 2 * np.outer(a1_m, a1_m)
                Type = 'Type II'
                # print(K1_a)
            else:
                # for other K1 and eta1 is preserved - rotation 180 around K1
                # C1_m=np.eye(3)-2*np.outer(n1_m,n1_m)
                if (twind['eta_m'] == np.round(twind['eta_m'])).all():
                    Type = 'compound'
                else:
                    Type = 'Type I'
                C1_m = -np.eye(3) + 2 * np.outer(a1_m, a1_m)

            Q1_m = R_AM.dot(Q1_a.dot(inv(R_AM)))

            twind['Q_m'] = Q1_m
            twind['R_m'] = C1_m
            twind['C_m'] = C1_m
            twind['C_a'] = C1_m
            twind['Type'] = Type

            for key in keys:
                twindata[key].append(twind[key])

        KEYS = ['shear_angle', 's', 'b_a', 'Rij_a', 'k', 'Type', 'n1_a', 'a1_a', 'n2_a', 'a2_a', 'Q_a', 'eta1_a',
                'K1_a', 'eta1_a_type',
                'K1_a_type', 'eta2_a', 'K2_a', 'eta2_a_type', 'K2_a_type', 'n1_m', 'a1_m', 'n2_m', 'a2_m', 'Q_m', 'R_m',
                'eta1_m', 'K1_m', 'eta2_m', 'K2_m', 'C_m', 'C_a']

        for ij in [[1, 0], [0, 1]]:
            TWIN = {}
            for key in KEYS:
                if '1' in key:
                    TWIN[key] = twindata[key.replace('1', '')][ij[0]]
                elif '2' in key:
                    TWIN[key] = twindata[key.replace('2', '')][ij[1]]
                else:
                    TWIN[key] = twindata[key][ij[0]]
            TWINDATA.append(TWIN)

    return TWINDATA
