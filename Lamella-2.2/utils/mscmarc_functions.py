import numpy as np
import sys
import os
import pickle
#from utils.crystallography_functions import *
from read_tessfile import *
sys.path.append('/usr/local/msc/2024/mentat2024.2/shlib/linux64/')
#sys.path.append('/usr/local/msc/marc/2022/mentat2022.1/shlib/linux64/')
from py_mentat import *


def gen_importproc(simuldir,Sets2Rem=[], SetName='POLY'):
    lines=[]
    lines.append('*new_model yes')


    import_file=os.path.join(simuldir,'2scale.inp')
    ModelName = import_file.replace('.inp','.mud');
    MarcInputName=import_file.replace('.inp','.dat');

    lines.append('*import abaqus "'+import_file+'"')

    lines.append('*remove_unused_nodes')
    lines.append('*check_zero')
    lines.append('*select_clear')
    lines.append('*check_zero')
    lines.append('*remove_elements all_selected')
    lines.append('*select_clear')
    lines.append("*renumber_all")


    lines.append('*write_marc "'+MarcInputName+'" yes')
    lines.append('*quit yes')
    procfile=os.path.join(simuldir,'proc.proc')
    fid=open(procfile,'wb')
    for line in lines:
        fid.write((line+'\n').encode())
    fid.close()
    return procfile,MarcInputName

def remsets(simuldir,MarcInputName,Sets2Rem=[], SetName='POLY'):
    lines=[]
    lines.append('*import marc_read "'+MarcInputName+'"')
    lines.append('*select_elements')
    for Set in Sets2Rem:
        lines.append(f"{SetName}{Set}")
    lines.append(f"*remove_elements")
    lines.append(f"all_selected")
    lines.append('*remove_unused_nodes')
    lines.append('*check_zero')
    lines.append('*select_clear')
    lines.append('*check_zero')
    lines.append('*remove_elements all_selected')
    lines.append('*select_clear')
    lines.append("*renumber_all")

    lines.append('*write_marc "'+MarcInputName+'" yes')
    lines.append('*quit yes')
    procfile=os.path.join(simuldir,'proc.proc')
    fid=open(procfile,'wb')
    for line in lines:
        fid.write((line+'\n').encode())
    fid.close()
    return procfile,MarcInputName


def gen_mentatproc(simuldir,MarcInputName,C,matname='austenite',plasticity=False,sigma_y0=800,sigma_y1=1200, coeff=5,SetName='POLY',CenralNodes=None):

    Xcoords, Ycoords, Zcoords, NodeIds = read_nodes(MarcInputName)
    lines=[]
    lines.append('*import marc_read "'+MarcInputName+'"')

    # open a file, where you stored the pickled data
    file = open(os.path.join(simuldir,'TwinningData.pckl'), 'rb')
    # dump information to that file
    TwinningData = pickle.load(file)
    # close the file
    file.close()
    TessFile= os.path.join(simuldir,'2scale.tess')
    TessData=read_tess(TessFile)

    file = open(os.path.join(simuldir,'cells.pckl'), 'rb')
    data = pickle.load(file)
    cells = data['cells']
    segments = data['segments']



    C11=str(C[0])
    C12=str(C[1])
    C44=str(C[2])
    matname='austenite'

    lines.append('*new_mater standard *mater_option general:state:solid *mater_option general:skip_structural:off')

    lines.append('*mater_option structural:type:elast_plast_aniso')

    lines.append(f'*mater_name {matname}')

    lines.append('*mater_param structural:c11 '+C11)

    lines.append('*mater_param structural:c22 '+C11)

    lines.append('*mater_param structural:c33 '+C11)

    lines.append('*mater_param structural:c12 '+C12)

    lines.append('*mater_param structural:c13 '+C12)

    lines.append('*mater_param structural:c23 '+C12)

    lines.append('*mater_param structural:c44 '+C44)

    lines.append('*mater_param structural:c55 '+C44)

    lines.append('*mater_param structural:c66 '+C44)


    lines.append('*add_mater_elements')

    lines.append('all_existing')
    if plasticity:
        #Definition of the yield stress evolution with the accumulation of plastic strain, i.e. hardening behaviour
        #initial yield stress
        #sigma_y0=sigma_y0word
        #yield stress when plastic strain = 1, i.e. 100% elongation
        #sigma_y1=sigma_y1word
        #exponential hardening assumed: sigma_y(pl_strain) = sigma_y0+(sigma_y1-sigma_y0)*(1-np.exp(-coeff*pl_strain))
        #hardening coefficient
        #coeff=coeffword
        #Hardening function
        pl_strain=np.linspace(0,1,101)
        stress_pl=sigma_y0+(sigma_y1-sigma_y0)*(1-np.exp(-coeff*pl_strain))

        #print("Assigning Plastic Deformation Behaviour.")
        tabname2='hardening'
        cmds=["*new_md_table 1 1","*set_md_table_type 1","eq_plastic_strain",
                f"*set_md_table_max_v 1 1","*table_name",f'{tabname2}']
        for cmd in cmds:
            lines.append(cmd)
        cmds=[f"*edit_table {tabname2}",f'*set_md_table_min_f 1 {sigma_y0}',f'*set_md_table_max_f 1 {sigma_y1}',"*table_add"]
        for cmd in cmds:
            lines.append(cmd)
        cmds=[(str(strain),str(stress)) for strain, stress in zip(pl_strain[0::10], stress_pl[0::10])]
        for cmdsi in cmds:
            for cmd in cmdsi:
                lines.append(cmd)
        lines.append(" ")
        cmds=[f"*edit_mater {matname}","*mater_option structural:plasticity:on",
                "*mater_param structural:yield_stress 1",f'*mater_param_table structural:yield_stress',
                f"{tabname2}"," "]
        for cmd in cmds:
            lines.append(cmd)

    ### Boundary conditions on the central nodes of the upper (Z=1) and bottom face (Z=0)
    if CenralNodes is None:
        CenralNodes=[]
        XC=(Xcoords.max()+Xcoords.min())/2
        YC=(Ycoords.max()+Ycoords.min())/2
        for Zc in [Zcoords.min(),Zcoords.max()]:
            SelNodesIdxs=np.where(Zcoords==Zc)[0]
            R=(Xcoords[SelNodesIdxs]-XC)**2+(Ycoords[SelNodesIdxs]-YC)**2
            Idx=SelNodesIdxs[np.argmin(R)]
            #print(f'X={Xcoords[Idx]},Y={Ycoords[Idx]},Z={Zcoords[Idx]}')
            CenralNodes.append(NodeIds[Idx])
    else:
        print("central nodes taken")
    CenterFix=[['x','y','z'],['x','y']]
    Names=['CenterBotom','CenterUp']
    for cnodes,Name,fix in zip(CenralNodes,Names,CenterFix):
        #Name=f'Corner{np.round(coords[0])}{np.round(coords[1])}{np.round(coords[2])}'
        #print(Name)
        lines.extend(nodeIds_displacement_proc(Name,[cnodes],fix))

    lines.extend(update_undo_proc('off'))
    # Assign orientations to cells
    #print("Assigning orientations to grains.")
    #OriData=[[phi1,Phi,phi2]  for phi1,Phi,phi2 in zip(TessData['bunge']['phi1'],TessData['bunge']['Phi'],TessData['bunge']['phi2'])]
    polyid=0
    #print('new')
    for i, cell in enumerate(cells):
        for seg in segments[i]:
            polyid+=1
            sname=f'{SetName}{polyid}'
            if seg[1] == "gap":
                lines.extend(Orient_for_set_proc(sname,cell.orientation))
            else:
                lines.extend(Orient_for_set_proc(sname,cell.lamella_orientation))



    #for ni,polyid in enumerate(TessData['polyhedron']['ID']):
    #    #print(f'{SetName}{polyid}')
    #    sname=f'{SetName}{polyid}'
    #    lines.extend(Orient_for_set_proc(sname,OriData[ni]))


    #assigned twinning strain to lamellea
    ## Twinning strains into a dictionary for each lamella
    #print("Assigning twinning strains to lamellea.")
    #eigenstrain,grainsID, grainsIDList=get_eigenstrains(simuldir,TwinningData,SetName=SetName)

    tablename='linramp1'
    lines.append("*new_pre_defined_table linear_ramp_time")
    lines.append(f"*table_name {tablename}")
    lines.append(" ")

    #lines.extend(set_inherent_strains_proc(eigenstrain,tablename))


    polyid=0
    for i, cell in enumerate(cells):
        for seg in segments[i]:
            polyid+=1
            sname=f'{SetName}{polyid}'
            if seg[1] != "gap":
                lines.extend(set_inherent_strain(sname,cell.twinning_strain,tablename))


    loadcasename='LCASE'
    lines.append('*new_loadcase *loadcase_type struc:static')
    lines.append('*loadcase_name '+loadcasename)
    lines.append('*loadcase_value time 1')
    lines.append('*loadcase_option stepping:fixed')
    lines.append('*loadcase_value nsteps 10')
    lines.append("*loadcase_option converge:displacements")
    #lines.append('*loadcase_option time_cut:off')
    lines.append('*loadcase_option time_cut:automatic')
    cmds=[f"*edit_loadcase {loadcasename}"," *loadcase_option nonpos:on"," "]
    for cmd in cmds:
        lines.append(cmd)

    #Define Simulation job
    #print("Defining simulation job.")
    lines.append('*edit_job job1')
    lines.append('*job_option strain:large')
    lines.append('*add_job_applys CenterBotom')
    lines.append('*add_job_applys CenterUp')
    lines.append('*add_job_loadcases '+loadcasename)




    #Define Results as output of hdf5 file
    lines.append(f"*job_option hdf_post:on @set($post_variables,hdf)")
    for result in 'von_mises epl_strain eel_strain te_energy ee_energy eq_inhr_strain'.split():
        lines.append(f"*add_post_hdf_var {result}")

    for result in 'stress stress_p back_stress strain el_strain el_strain_p pl_strain pl_strain_g inhr_strain inhr_strain_p'.split():
        lines.append(f"*add_post_hdf_tensor {result}")


    lines.extend(update_undo_proc('on'))
    lines.append('*write_marc "'+MarcInputName+'" yes')
    lines.append('*quit yes')
    procfile = os.path.join(simuldir,'proc.proc')
    fid=open(procfile,'wb')
    for line in lines:
        fid.write((line+'\n').encode())
    fid.close()
    return procfile
def nodeIds_displacement_proc(Name,NodeIds,directions, values=[0,0,0]):
    lines=[]
    lines.append("*new_apply")
    lines.append("*apply_type fixed_displacement")
    lines.append("*apply_name "+Name)


    for direction,value in zip(directions,values):
        lines.append("*apply_dof "+direction)
        lines.append('*apply_dof_value '+direction+' '+str(value))

    lines.append("*add_apply_nodes ")

    strids = ""
    N=100;
    n=0
    ni=0;
    for Node in NodeIds:
        ni+=1;
        #print('%d/%d' % (ni, len(NodeIds)))
        n+=1;
        strids += "%d " % Node
        if n==N:
            lines.append(strids)
            lines.append('all_selected')
            #py_send(' ')
            #py_send(' *edit_apply '+Name)
            #py_send('*add_apply_nodes ')
            n=0;
            strids=""
    lines.append(strids)
    lines.append('all_selected')
    lines.append(' ')
    return lines
def Orient_for_set_proc(name,OSdata):
#   select elements in set
    lines = []
    lines.append("*select_sets " + name)

#   open new orientation definition
    lines.append("*new_orient *orient_type 3d_aniso")

#   asign name to orientation
    lines.append("*orient_name " + name)


#   basic vectors definition
    bv_1=[1., 0., 0.]
    bv_2=[0., 1., 0.]
#   transformed vectors - inicialization
    v1=[1., 0., 0.]
    v2=[0., 1., 0.]


#   extract Euler Angles for the set



    # transf. matrix
#    tm=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    #tm = euler_matrix(OSdata[0], OSdata[1], OSdata[2])
    #print("===============================================================")
    #print("inverse taken")
    #print("===============================================================")
    tm = np_inverse_euler_matrix(OSdata[0], OSdata[1], OSdata[2]);


    # vector transformation: v1=tm*bv_1, v2=tm*bv_2
    for i in range(0,3):
       tmr=tm[i]
       v1[i]=tmr[0]*bv_1[0]+tmr[1]*bv_1[1]+tmr[2]*bv_1[2]
       v2[i]=tmr[0]*bv_2[0]+tmr[1]*bv_2[1]+tmr[2]*bv_2[2]

#   asign transformed vectors to the orientation
    lines.append("set_orient_vector1 "+"%.8f"%(v1[0])+" "+"%.8f"%(v1[1])+" "+"%.8f"%(v1[2]))

    lines.append("set_orient_vector2 "+"%.8f"%(v2[0])+" "+"%.8f"%(v2[1])+" "+"%.8f"%(v2[2]))


#   asign selected elements (from set "name")
    lines.append("*add_orient_elements all_selected")


#   clear select
    lines.append("*select_clear_elements")

    return lines

def set_inherent_strain(SetName,twinning_strain,tablename):
    line=[]
    line.append("*select_sets " + SetName)
    line.append("*new_apply *apply_type inherent_strain")
    line.append("*apply_name " + SetName+'eig')
    line.append("*apply_dof e *apply_dof_value e")

    for key in twinning_strain.keys():
        line.append(f"*apply_param_value inh_strain_base_{key} {np.round(twinning_strain[key],decimals=10)}")
    line.append("*apply_option inh_strain_base_csys:material")
    line.append("*add_apply_elements all_selected")
    #   clear select
    line.append("*select_clear_elements")
    line.append("*edit_apply " + SetName+'eig')
    for dirs in 'xx yy zz xy yz zx'.split():
        line.append(f"*apply_param_table inh_strain_base_{dirs} {tablename}")
    return line


def set_inherent_strains_proc(eigenstrain,tablename):
    lines=[]
    for key in list(eigenstrain.keys()):
        line=[]
        line.append("*select_sets " + key)
        line.append("*new_apply *apply_type inherent_strain")
        line.append("*apply_name " + key+'eig')
        line.append("*apply_dof e *apply_dof_value e")

        for dirs,eigidxs in zip('xx yy zz xy yz zx'.split(),[(0,0),(1,1),(2,2),(0,1),(1,2),(2,0)]):
            #line.append(f"*apply_param_value inh_strain_base_{dirs} "+repr(np.round(eigenstrain[key][eigidxs],decimals=10)))
            line.append(f"*apply_param_value inh_strain_base_{dirs} {np.round(eigenstrain[key][eigidxs],decimals=10)}")
        line.append("*apply_option inh_strain_base_csys:material")
        line.append("*add_apply_elements all_selected")
        #   clear select
        line.append("*select_clear_elements")
        line.append("*edit_apply " + key+'eig')
        for dirs in 'xx yy zz xy yz zx'.split():
            line.append(f"*apply_param_table inh_strain_base_{dirs} {tablename}")
        lines.extend(line)
    return lines


#    for i, cell in enumerate(small_cells):
#        file_name = f"./data/cell{i + 1}"
#        with open(file_name, "w", encoding='utf-8') as file:
#            segments = small_segments[i]
#            for seg in segments:
#                if seg[1] == "gap":
#                    file.write(f"{cell.orientation[0] } {cell.orientation[1] } {cell.orientation[2] }\n")
#                else:
#                    file.write(
#                        f"{cell.lamella_orientation[0] } {cell.lamella_orientation[1] } {cell.lamella_orientation[2] }\n")


def get_eigenstrains(simuldir,TwinningData,polyid=0,numlam=0,SetName='POLY'):
    eigenstrain={}
    grainsID={}
    grainsIDList={}
    SF={}
    for cellid in range(len(TwinningData['StrainSymGLTwin'])):
        #print(f'cell{cellid+1}')
        cellf = open(os.path.join(simuldir, f'cell{cellid+1}'), "r")
        lines=cellf.readlines()
        grainsID[cellid+1]=[]
        for li,line in enumerate(lines):
            polyid+=1
            grainsID[cellid+1].append(polyid)
            grainsIDList[polyid]=cellid+1
            #if li%2!=0:#(np.abs(np.array(Cells[cellid].lam_ori)-np.array(TwinningData['neweus'][cellid]))<1e-5).all():
            if (np.abs(np.array([float(li)for li in line.split()])-np.array(TwinningData['neweus'][cellid]))<1e-5).all():
                eigenstrain[f'{SetName}{polyid}']=TwinningData['StrainSymGLTwin'][cellid]
                SF[f'{SetName}{polyid}']=TwinningData['SF'][cellid]
        cellf.close()
    
    if False:
        eigenstrain={}
        SF={}
        for cellid in range(len(TwinningData['StrainSymGLTwin'])):
            #print(f'cell{cellid+1}')
            cellf = open(os.path.join(simuldir, f'cell{cellid+1}'), "r")
            lines=cellf.readlines()
            for line in lines:
                polyid+=1
                if (np.abs(np.array([float(li)for li in line.split()])-np.array(TwinningData['neweus'][cellid]))<1e-5).all():
                    numlam+=1
                    eigenstrain[f'{SetName}{polyid}']=TwinningData['StrainSymGLTwin'][cellid]
                    SF[f'{SetName}{polyid}']=TwinningData['SF'][cellid]
            cellf.close()

    return eigenstrain,grainsID, grainsIDList
def update_undo_proc(onoff):
    lines=[]
    lines.append(f"*py_echo {onoff}")
    lines.append(f"*set_undo {onoff}")
    lines.append(f"*set_update {onoff}")
    return lines


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
def read_nodes(marcinputfile):
    file = open(marcinputfile,'r') 
    file1 = file.readlines()  
    file.close()
    #time.sleep(35)

    coordinates_start = 0
    for key in ['coordinates']:
        for line in file1:
            if key in line:
                break
            else:
                coordinates_start += 1;

    NumberOfNodes = file1[coordinates_start+1].split()[1]
    Nodes = extract_nodes(file1,coordinates_start,'.')

    Zcoords = [Node[3] for Node in Nodes];
    Ycoords = [Node[2] for Node in Nodes]
    Xcoords = [Node[1] for Node in Nodes]
    NodeIds = [Node[0] for Node in Nodes]

    MinZ = min(Zcoords)
    MaxZ = max(Zcoords)
    MinY = min(Ycoords)
    MaxY = max(Ycoords)
    MeanY = np.mean(Ycoords)
    MinX = min(Xcoords)
    MaxX = max(Xcoords)
    MeanX = np.mean(Xcoords)
    Zcoords = np.array(Zcoords)
    Ycoords = np.array(Ycoords)
    Xcoords = np.array(Xcoords)
    NodeIds = np.array(NodeIds)
    return Xcoords, Ycoords, Zcoords, NodeIds

def import_inp(WorkingDir,import_file, ModelName, quadratic=False):
    ModelNameBack = ModelName.replace('.mud','_back.mud')
    if False:
        f = open(WorkingDir+'/'+import_file, 'r')
        linelist = f.readlines()
        f.close
        
        corrected_import_file = import_file.replace('.inp','_corr.inp');
        
        # Re-open file here
        f2 = open(WorkingDir+'/'+corrected_import_file, 'w')
        nexline2modif = False;
        modifiedsets=[]
        for line in linelist:
            if nexline2modif:
                nexline2modif = False;
                if not(',' in line) and line!='\n':
                    setid = int(keyline.split()[-1].replace('elset=poly',''));
                    modifiedsets.append(setid);
                    print("Replacement in "+keyline)
                    if setid==30:
                        line30=line
                    line = line.replace('\n',',\n');
            if ('*Elset' in line):        
                keyline = line;
                nexline2modif = True;
            f2.write(line)
        f2.close()
        ImportFile = corrected_import_file;
    else:
        ImportFile=import_file
    
    #import
    
    cmd = '*import abaqus "'+ImportFile+'"';
    py_send(cmd)
    
    
    #show model
    cmd = '*model_orientation bottom';
    py_send(cmd)
    cmd = '*fill_view';
    py_send(cmd)
    py_send('*py_update')
    #Number of nodes
    n = py_get_int("nnodes()")
    print("Number of nodes - "+str(n))

    #Number of elements
    m = py_get_int("nelements()")
    print("Number of Elements - "+str(n))
    
    py_send('*remove_unused_nodes')
    py_send('*check_zero')
    
    ##Save Zero volume elements
    #py_send('*store_elements '+'ZeroVolElements')
    #py_send('all_selected')
    #py_send(' ')
    #n = py_get_int("nsets()")                       # get number of sets
    #i=n
    #ids = py_get_int("set_id(%d)" % i) 
    #sname = py_get_string("set_name(%d)" %ids) 
    #print "Name of set: ", sname
    #cmd ="nset_entries(%d)" % ids
    #n = py_get_int(cmd)            
    #print "No of zero volume elements in the set:", n
    #for j in range(1,n+1):
    #    k = py_get_int("set_entry(%d,%d)" % (ids, j))
    #    print "Zero volume element id:", k
    #
    
    ##Remove Zero volume elements
    py_send('*select_clear')
    py_send('*check_zero')
    py_send('*remove_elements')
    py_send('all_selected')
    py_send(' ')
    py_send('*select_clear')
    py_send("*renumber_all")
    
    #
    if quadratic:
        py_send('*change_elements_quadratic')
        py_send('all_existing')
        py_send('*sweep_elements')
        py_send('all_existing')
        py_send('*sweep_nodes')
        py_send('all_existing')
        py_send("*renumber_all")
        py_send(' ')
    
    cmd = '*model_orientation bottom';
    py_send(cmd)
    cmd = '*fill_view';
    py_send(cmd)
    py_send('*py_update')

    #save mud file
    cmd = 'save_as_model '+ModelNameBack+' yes';
    py_send(cmd)

    cmd = 'save_as_model '+ModelName+' yes';
    py_send(cmd)
    
    max_node_id = py_get_int("max_node_id()")
    n = py_get_int("nnodes()")
    if n==max_node_id:
        print("Nodes are renumbered correctly");
    else:
        print("Nodes are not renumbered");
    
    max_element_id = py_get_int("max_element_id()")
    m = py_get_int("nelements()")
    if m==max_element_id:
        print("Elements are renumbered correctly");
    else:
        print("Elements are not renumbered");
    #Elems = []
    #for i in range(1,m+1):
    #    if i%10000==0:
    #        print str(i)+'/'+str(m+1);        
    #
    #    cmd = "element_id(%d)" % i
    #    ide = py_get_int(cmd);
    #    Elems.append(ide)
    #
    #Check if all elements assigned to a grain
    #ElInSets, NoElSets = numel_in_sets('POLY')
    
    #if m==ElInSets:
    #    print("All elements are assigned to a grain set");
    #else:
    #    print("Some elements are not assigned to any grain set");

def extract_nodes(file1,coordinates_start,decimal=','):
    idstart = coordinates_start+2
    line = file1[idstart];
    Nodes = [];
    #YC=[];
    while line.startswith((' ', '\t')):
        line1 = line.strip();
        #line1=lineini;
        #print(line1)
        idxNodeNumber = len(line1)

        idx1 = line1.find(decimal);
        idx1 = line1[0:idx1].find('-')
        if idx1!=-1:
            idxNodeNumber = idx1;

        idx1 = line1.find(decimal);
        idx1 = line1[0:idx1].find('+')
        if idx1!=-1:
            idxNodeNumber = idx1;

        NodeNumber = line1[0:idxNodeNumber].split()[0]
        #NodeNumber = line1.split()[0];
        
        line1 = line1[line1.find(NodeNumber)+len(NodeNumber):]
        
        Xcoord = extract_coord(line1,decimal);
        
        #line1 = line1.rsplit(Xcoord)[1].strip()
        line1 = line1[line1.find(Xcoord)+len(Xcoord):];
        #print(line1)
        Ycoord = extract_coord(line1,decimal);
        #print(Ycoord)
        Zcoord = line1[line1.find(Ycoord)+len(Ycoord):];
        
    
        if Zcoord[0]=='-':
            Zsign = -1.
        else:
            Zsign = 1.;
        if Ycoord[0]=='-':
            Ysign = -1.
        else:
            Ysign = 1.;
        if Xcoord[0]=='-':
            Xsign = -1.
        else:
            Xsign = 1.;
            
        if Zcoord[0]=='-' or Zcoord[0]=='+':
            Zcoord = Zcoord[1:]
        if Ycoord[0]=='-' or Ycoord[0]=='+':
            Ycoord = Ycoord[1:]
        if Xcoord[0]=='-' or Xcoord[0]=='+':
            Xcoord = Xcoord[1:]
        Nodes.append([int(NodeNumber), Xsign*float(Xcoord.replace(',','.').replace('+','E+').replace('-','E-')),
                      Ysign*float(Ycoord.replace(',','.').replace('+','E+').replace('-','E-')),
                      Zsign*float(Zcoord.replace(',','.').replace('+','E+').replace('-','E-'))])
    
        idstart+=1;
        line = file1[idstart];
        #YC.append(Ysign*float(Ycoord.replace(',','.').replace('+','E+').replace('-','E-')))
    return Nodes        

def extract_coord(line1,decimal=','):
    idx1 = line1.find(decimal);
    idx2 = line1.find(decimal,idx1+1);
    if line1[idx1:idx2].find(' ')>-1:
        
        Coord = line1.split()[0]
    else:
        idxplus = line1[idx1:idx2].find('+');
        idxminus = line1[idx1:idx2].find('-');
        if idxminus==-1:
            strf = '+'
            idxini = idxplus
        elif idxplus==-1:
            strf = '-'  
            idxini = idxminus
        else:
            if idxplus<idxminus:
                strf = '-';
                idxini = idxplus
            else:
                strf = '+'
                idxini = idxminus
                
        idxplus2 = line1[idx1:idx2].find(strf,idxini+1)
        Coord = line1[0:idxplus2+idx1]
    return Coord
def nodeIds_displacement(Name,NodeIds,directions, values=[0,0,0]):
    py_send("*new_apply")
    py_send("*apply_type fixed_displacement")
    py_send("*apply_name "+Name)
    

    for direction,value in zip(directions,values):
        py_send("*apply_dof "+direction)
        py_send('*apply_dof_value '+direction+' '+str(value))
    
    py_send("*add_apply_nodes ")
    
    strids = ""
    N=100;
    n=0
    ni=0;
    for Node in NodeIds:
        ni+=1;
        print('%d/%d' % (ni, len(NodeIds)))
        n+=1;
        strids += "%d " % Node
        if n==N:
            py_send(strids)
            py_send('all_selected')
            #py_send(' ')
            #py_send(' *edit_apply '+Name)
            #py_send('*add_apply_nodes ')
            n=0;
            strids=""
    py_send(strids)
    py_send('all_selected')
    py_send(' ')
def Orient_for_set(name,OSdata): 
#   select elements in set
    cmd="*select_sets " + name
    py_send (cmd)
    
#   open new orientation definition
    cmd="*new_orient *orient_type 3d_aniso"
    py_send (cmd)
    
#   asign name to orientation
    cmd="*orient_name " + name
    py_send (cmd)
    
    
#   basic vectors definition
    bv_1=[1., 0., 0.]
    bv_2=[0., 1., 0.]
#   transformed vectors - inicialization
    v1=[1., 0., 0.]
    v2=[0., 1., 0.]


#   extract Euler Angles for the set
    
 

    # transf. matrix
#    tm=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    #tm = euler_matrix(OSdata[0], OSdata[1], OSdata[2])
    #print("===============================================================")
    #print("inverse taken")
    #print("===============================================================")
    tm = np_inverse_euler_matrix(OSdata[0], OSdata[1], OSdata[2]);
 
    
    # vector transformation: v1=tm*bv_1, v2=tm*bv_2
    for i in range(0,3):
       tmr=tm[i]  
       v1[i]=tmr[0]*bv_1[0]+tmr[1]*bv_1[1]+tmr[2]*bv_1[2]
       v2[i]=tmr[0]*bv_2[0]+tmr[1]*bv_2[1]+tmr[2]*bv_2[2]

#   asign transformed vectors to the orientation   
    cmd="set_orient_vector1 "+"%.8f"%(v1[0])+" "+"%.8f"%(v1[1])+" "+"%.8f"%(v1[2])
    py_send (cmd)
    cmd="set_orient_vector2 "+"%.8f"%(v2[0])+" "+"%.8f"%(v2[1])+" "+"%.8f"%(v2[2])
    py_send (cmd)
     
    
#   asign selected elements (from set "name")   
    cmd="*add_orient_elements all_selected"
    py_send (cmd)

#   clear select    
    cmd="*select_clear_elements"
    py_send (cmd)


def set_ori(SetName,TessData,Odata):
    print("================================================")
    print("============= Solid Orient Definition ==========")
    print("================================================")

    for ni,polyid in enumerate(TessData['polyhedron']['ID']):
        #print(f'{SetName}{polyid}')
        sname=f'{SetName}{polyid}'
        Orient_for_set(sname,Odata[ni])

def update_undo(onoff):
    # echo, update and undo off
    if onoff == 'off':
        cmd="*py_echo off"
        py_send (cmd)    
        cmd="*set_undo off"
        py_send (cmd)
        cmd="*set_update off"
        py_send (cmd)
    if onoff == 'on':
        cmd="*py_echo on"
        py_send (cmd)    
        cmd="*set_undo on"
        py_send (cmd)
        cmd="*set_update on"
        py_send (cmd)
def create_nodeset(Name,Nodes):
    import time
    #print('ahoj')
    py_send('*store_nodes')
    py_send(Name)
    cmd = ""
    N =100;
    n=0
    ni=0;
    for i in Nodes:
        ni+=1;
        #print('%d/%d' % (ni, len(Nodes)))
        n+=1;
        cmd += "%d " % i
        if n==N:
            py_send(cmd)
            py_send('all_selected')
            py_send('*store_nodes '+Name)
            n=0;
            cmd=""
            #time.sleep(5)
    py_send(cmd)
    py_send('all_selected')
    py_send(' ')
