# Read orientations from .tess file
def read_tess(TessFileName):
    f=open(TessFileName, "r")
    Lines = f.readlines()
    f.close()        
    reading=False
    
    TessData={}
    
    reading='no'
    for line in Lines:
        if '*' in line:
            inc = 0
            if "**vertex" in line:
                reading='vertex'
                TessData['vertex'] = {'number':0,'ID':[],'vertices_xyz':[],'vertex_state':[]}
            elif "**edge" in line:
                reading='edge'
                TessData['edge'] = {'number':0,'ID':[],'vertices_id':[],'edge_state':[]}
            elif "**face" in line:
                reading='face'
                TessData['face'] = {'number':0, 'ID':[], 'vertices_id':[],'edges_id':[],'face_eq':[],'face_state':[],
                        'face_point':[],'face_xyz':[]}
            elif "*face" in line:
                reading='domainface'
                TessData['domainface'] = {'number':0, 'ID':[], 'vertices_id':[],'edges_id':[],'face_eq':[],'face_type':[],
                        'number_tess_faces':[],'tess_faces':[],'all_tess_faces':[],'face_label':[]}
            elif "**polyhedron" in line:
                reading='polyhedron'
                TessData['polyhedron'] = {'number':0,'ID':[],'face_id':[]}
            else:
                reading='no'
        elif "euler-bunge:passive" in line:
                reading='bunge'
                TessData['bunge'] = {'phi1':[],'Phi':[],'phi2':[]}
        else:
            
            if reading == 'polyhedron':
                if inc==0:
                    TessData['polyhedron']['number']=int(line)
                else:
                    dat=line.split()
                    TessData['polyhedron']['ID'].append(int(dat[0]))
                    TessData['polyhedron']['face_id'].append([int(di) for di in dat[2:]])
                    #break    
                inc+=1
            if reading == 'bunge':
                dat=line.split()
                TessData['bunge']['phi1'].append(float(dat[0]))
                TessData['bunge']['Phi'].append(float(dat[1]))
                TessData['bunge']['phi2'].append(float(dat[2]))
                
            if reading == 'face':
                if inc==0:
                    TessData['face']['number']=int(line)
                else:
                    if inc==1:
                        dat=line.split()
                        TessData['face']['ID'].append(int(dat[0]))
                        TessData['face']['vertices_id'].append([int(di) for di in dat[2:]])
                    elif inc==2:
                        dat=line.split()
                        TessData['face']['edges_id'].append([int(di) for di in dat[1:]])
                    elif inc==3:
                        dat=line.split()
                        #break
                        TessData['face']['face_eq'].append([float(di) for di in dat])
                    elif inc==4:
                        dat=line.split()
                        TessData['face']['face_state'].append(int(dat[0]))
                        TessData['face']['face_point'].append(int(dat[1]))
                        face_xyz = []
                        for di in dat[2:]:   
                            face_xyz.append(float(di))
                        TessData['face']['face_xyz'].append(face_xyz)
                        inc=0
                        #break    
                inc+=1
            if reading == 'domainface':
                #print(line)
                if inc==0:
                    TessData['domainface']['number']=int(line)
                else:
                    if inc==1:
                        dat=line.split()
                        TessData['domainface']['ID'].append(int(dat[0]))
                        TessData['domainface']['vertices_id'].append([int(di) for di in dat[2:]])
                    elif inc==2:
                        dat=line.split()
                        TessData['domainface']['edges_id'].append([int(di) for di in dat[1:]])
                    elif inc==3:
                        #break
                        TessData['domainface']['face_type'].append(line)
                    elif inc==4:
                        dat=line.split()
                        #break
                        TessData['domainface']['face_eq'].append([float(di) for di in dat])
                    elif inc==5:
                        #break
                        TessData['domainface']['face_label'].append(line)
                    elif inc==6:
                        dat=line.split()
                        TessData['domainface']['number_tess_faces'].append(int(dat[0]))
                        TessData['domainface']['tess_faces'].append([int(di) for di in dat[1:]])
                        TessData['domainface']['all_tess_faces'].extend([int(di) for di in dat[1:]])
                        inc=0
                       #break    
                inc+=1
            if reading == 'edge':
                if inc==0:
                    TessData['edge']['number']=int(line)
                else:
                    dat=line.split()
                    TessData['edge']['ID'].append(int(dat[0]))
                    TessData['edge']['vertices_id'].append([int(di) for di in dat[1:-1]])
                    TessData['edge']['edge_state'].append(int(dat[-1]))
                    #break    
                inc+=1
            if reading == 'vertex':
                if inc==0:
                    TessData['vertex']['number']=int(line)
                else:
                    dat=line.split()
                    TessData['vertex']['ID'].append(int(dat[0]))
                    TessData['vertex']['vertices_xyz'].append([float(di) for di in dat[1:-1]])
                    TessData['vertex']['vertex_state'].append(int(dat[-1]))
                    #break    
                inc+=1
    
    return  TessData         
