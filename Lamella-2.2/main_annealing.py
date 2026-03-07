#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:44:56 2025

@author: lheller
"""

# Import libraries
import sys
sys.path.append("/home/lheller/Jupyter/Lamella-2.2/utils/")
sys.path.append("/home/lheller/Jupyter/Lamella-2.2/")
#sys.path.append('/usr/local/msc/2024/mentat2024.2/shlib/linux64/')
#sys.path.append('/usr/local/msc/2024/mentat2024.2/python/LX8664/extra-lib/libimf.so')
sys.path.append('/usr/local/msc/2022/mentat2022.1/shlib/linux64/')
#sys.path.append('/usr/local/msc/2022/mentat2022.1/python/LX8664/extra-lib/libimf.so')
#import Twin
#import json
#import Classes
import osamp
import main
import lhfun
from read_tessfile import *
from mscmarc_functions import *
import numpy as np
import os
#import Process
import gmsh
import subprocess
import glob
import pickle
import shutil
import json
import matplotlib.pyplot as plt
import pickle


#Path to MSC mentat prepocessor
mentatpath='/usr/local/msc/2024/mentat2024.2/bin/mentat'
mentatpath='/usr/local/msc/2022/mentat2022.1/bin/mentat'
#Path to folder with source code and data folder from which the resulat are compied to Simulation subfolder inside simulationset folder
BASEPATH='/home/lheller/Jupyter/Lamella-2.2/'
#path to the folder where files with parameters of individual simulations will be stored
paramfolder='/home/lheller/Jupyter/Lamella-2.2/params/'
if BASEPATH[-1]!='/':
    BASEPATH+='/'
#Main simulation set folder
simulationsetfoldername='simulationset004'
#Simulation subfolder without numbering
simfold='simulation'
#Simulation set subfolder name to which 000X numbers will be added for each particular simulation in the set
SIMULBASEPATH='/home/lheller/data/marc/Lamella-2.2/'
simulfoldername=f'{simulationsetfoldername}/{simfold}'
#If take the initial tessellation/orientations from the data currently being in ./data folder
#TakeTessFromDataFolder=True
#TakeOriFromDataFolder=True



#periodicity - does not work will multiscale in neper anyway
periodicity=False
#mesh size parameter
rcl='default'
rcl='0.5'
#Plasticity on/off
plasticity=True
#Definition of the yield stress evolution with the accumulation of plastic strain, i.e. hardening behaviour
#initial yield stress
sigma_y0=800
#yield stress when plastic strain = 1, i.e. 100% elongation
sigma_y1=1200
#exponential hardening assumed: sigma_y(pl_strain) = sigma_y0+(sigma_y1-sigma_y0)*(1-np.exp(-coeff*pl_strain))
#hardening coefficient
coeff=5
C11_num= 169e3
C12_num= 141e3
C44_num= 33e3

AdditionalInfo=f'meshing: nondefault size -rcl {rcl} and regularization (-reg 1)'
AdditionalInfo+='10 realizaci'
AdditionalInfo+=f'\nElastic constants C11/C12/C44 [MPa] {C11_num}/{C12_num}/{C44_num}'
AdditionalInfo+=f'\nPlasticity {plasticity}'
if plasticity:
    AdditionalInfo+=f'\nsigma_y0 [MPa] {sigma_y0}'
    AdditionalInfo+=f'\nsigma_y1 [MPa] {sigma_y1}'
    AdditionalInfo+=f'\ncoeff {coeff}'


if True:
    #kappas = [[0], [30]]
    kappas = [[0], [10], [20], [30]]
    epsilons = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    simnum=0
    simulfoldernames=[]
    simuldirs=[]
    for kappa in kappas:
        for epsilon in epsilons:
            simnum+=1
            simulfoldernames.append(f'{simulfoldername}{str(simnum).zfill(4)}')
            simuldirs.append(os.path.join(SIMULBASEPATH,simulfoldernames[-1]))
simuldirs = sorted(simuldirs)





os.chdir(BASEPATH)
errors=[]
mean_precision=[]
std_precision=[]
mesh=True
mscmodel=True
#for jsonfile in sorted(glob.glob('config*.json')):
for simuldir in sorted(simuldirs)[11:]:#simuldirsfailed:#sorted(simuldirs)[12:13]:
    #print('====================================================================')
    print(simuldir.split('/')[-1],end=':')
    os.chdir(BASEPATH)
    with open(os.path.join(simuldir,'config.json'), 'r') as file:
        config = json.load(file)
    REDOTess=True
    while REDOTess:     
        print(f"Tess",end='/')
        cells, segments=main.main(os.path.join(simuldir,'config.json'),twinning_strain='StrainSymGLTwin', get_results=True)
        file = open('./data/cells.pckl', 'wb')
        pickle.dump({'cells':cells,'segments':segments}, file)
        file.close()        
        
        lhfun.copy_data_to_myresult(config['target_path'])
        file = open(os.path.join(simuldir,'TwinningDataAll.pckl'), 'rb')
        TwinningDataAll = pickle.load(file)
        with open(os.path.join(simuldir,'results.json'), 'r') as file:
            results = json.load(file)
        #get grains ID in the [0,1]
        CID=[res['cid'] for res in results]
        TwinningData={}
        for key in TwinningDataAll.keys():
            try:
                TwinningData[key] = [TwinningDataAll[key][idx-1] for idx in CID]
            except:
                TwinningData[key] = TwinningDataAll[key]
        fileTD = open(os.path.join(simuldir,'TwinningData.pckl'), 'wb')
        pickle.dump(TwinningData, fileTD)
        fileTD.close()
        
        TessFileSmall= os.path.join(simuldir,'small.tess')
        TessDataSmall=read_tess(TessFileSmall)
        TessFile= os.path.join(simuldir,'2scale.tess')
        #TessFileSmall= os.path.join(simuldir,'small.tess')
        TessData=read_tess(TessFile)
        #eigenstrain,grainsID, grainsIDList=get_eigenstrains(simuldir,TwinningData,SetName='POLY')
        polyid=0
        for cellid in range(len(TwinningData['StrainSymGL'])):
            cellf = open(os.path.join(simuldir, f'cell{cellid+1}'), "r")
            lines=cellf.readlines()
            for li,line in enumerate(lines):
                polyid+=1
        if polyid==len(TessData['polyhedron']['ID']):
            if mesh:
                TessFile= os.path.join(simuldir,'2scale.tess')
                gmshfile=TessFile.replace('.tess','.msh4')
                print(f"Mesh",end='/')
                # Generating mesh by Neper: parameter -rcl 0.5 defines the mesh size - the smaler number the finner mesh, -reg 0: no regularization
                #'-rcl','0.8' increases the num. of elements from ~90000 to ~140000
                if rcl=='default':
                    command1 = [
                        "neper",
                        "-M",
                        TessFile,
                        '-format',
                         "msh4",
                        '-o',
                        gmshfile]
                else:
                    command1 = [
                        "neper",
                        "-M",
                        TessFile, '-rcl',rcl,                     
                        '-format',
                         "msh4",
                        '-o',
                        gmshfile]
                
                result = subprocess.run(command1, capture_output=True, text=True)
                for f in glob.glob(os.path.join(BASEPATH,"*.geo")):
                    os.remove(f)
                for f in glob.glob(os.path.join(BASEPATH,"*.msh")):
                    os.remove(f)
                for f in glob.glob(os.path.join(BASEPATH,"*.proc")):
                    os.remove(f)

                # Print the output and error (if any)
                #print("stdout:", result.stdout)
                #print("stderr:", result.stderr)
                #print("Return code:", result.returncode)
                if result.returncode != 0:
                    #print(f"FEM meshing failed:\n{result.stderr}")
                    print('Mesh Faild',end='/')
                else:
                    #print(f"Finished generating FEM mesh for {TessFile} file.")
                    #rc = run_command(command1)
                    # Export the mesh to MSC Marc readable input file
                    gmsh.initialize()
                    gmsh.open(gmshfile)
                    inputfile=gmshfile.replace('.msh4','.inp')
                    gmsh.write(inputfile)
                    gmsh.finalize()
                    #delete temp files of gmsh
                    for file in os.listdir(simuldir):
                        if "tmp" in file: 
                            #print("Duplicate file found: " + file)
                            os.remove(os.path.join(simuldir,file))
                        else:
                            continue
                    #print('=======================================================================================================================')
                    REDOTess=False
                    print('OK',end='||')
            else:
                REDOTess=False
                print('OK',end='||')
        else:
            REDOTess=True
            print('Tess Failed',end='/')
    if mscmodel:
        
        print('==============================================================')
        print(simuldir)
        print('==============================================================')
        procfile,MarcInputName = gen_importproc(simuldir)
        command1=[mentatpath, '-bg', procfile]
        print('Importing Abaqus input file, verifying, renumbering mesh, and writing mesh into Marc input file')
        result = subprocess.run(command1, capture_output=True, text=True,env={'DISPLAY': ':0','MSC_LICENSE_FILE':'27500@pc015c.fzu.cz'})
        
        procfile = gen_mentatproc(simuldir,MarcInputName,[C11_num,C12_num,C44_num],matname='austenite',plasticity=True,
                                                sigma_y0=sigma_y0,sigma_y1=sigma_y1, coeff=coeff)
        command1=[mentatpath, '-bg', procfile]
        print('Assigning material, BCs, orientations, inherent strains, load case, job, and writing final Marc input file')
        result = subprocess.run(command1, capture_output=True, text=True,env={'DISPLAY': ':0','MSC_LICENSE_FILE':'27500@pc015c.fzu.cz'})











