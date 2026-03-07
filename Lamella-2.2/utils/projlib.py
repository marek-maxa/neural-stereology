#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:13:57 2019

@author: lheller
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from orilib import *
from matplotlib.patches import Wedge
try:
    import random
except:
    pass
try:
    import scipy
    from scipy.interpolate import griddata

except:
    pass
try:
    from orix.quaternion import Rotation
except:
    pass
try:
    from orix.vector import Vector3d
except:
    pass
try:
    from orix.projections import StereographicProjection
except:
    pass
try:
    from diffsims.generators.rotation_list_generators import get_beam_directions_grid
except:
    pass
try:
    from matplotlib.colors import ListedColormap
except:
    pass
from spherical_kde import SphericalKDE

try:
    from wand.image import Image
except:
    pass
def genoritri(resolution=1.0, mesh="spherified_cube_edge"):
    grid_cub = get_beam_directions_grid("cubic", resolution, mesh=mesh)
    trioris = Rotation.from_euler(np.deg2rad(grid_cub))*Vector3d.zvector()
    return trioris.data.T

#Set of all orientations in space defined by rotation around 2 perpendicular axis and an angle resolution
def genori(dangle=1.0,hemi='both', tol=1e-2, rot=np.eye(3), half='no'):
    #dangle=1.
    Phi1=np.linspace(0.,360.-dangle,int(360./dangle))
    Phi2=np.linspace(0.,180.-dangle,int(180./dangle))
    GridX=[];
    GridY=[];
    Dc=np.array([1,0,0]);
    oris=[]
    for phi1 in Phi1:
        for phi2 in Phi2:        
            RotZ = active_rotation(phi1, 'z', deg=True) 
            RotY = active_rotation(phi2, 'y', deg=True)
            oris.append(RotY.dot(RotZ.dot(Dc)))
    oris=rot.dot(np.asarray(oris).T)
    if half=='upper':
        poris=equalarea_directions(oris)
        oris=oris[:,poris[1,:]>-tol]
    elif half=='lower':
        poris=equalarea_directions(oris)
        oris=oris[:,poris[1,:]<tol]
    if hemi=='upper':
        return oris[:,oris[2,:]>-tol]
    elif hemi=='lower':
        return oris[:,oris[2,:]<tol]
    else:    
        return oris

#generate regular grid as masked array npxnp covering the whole range of projected oris
def xyz2spher(xyz,deg=False):
    polar_angle=np.arctan2(np.sqrt(xyz[:,0]**2+xyz[:,1]**2),xyz[:,2])
    azimuth_angle=np.arctan2(xyz[:,1],xyz[:,0])        
    if deg:
        polar_angle*=180/np.pi
        azimuth_angle*=180/np.pi    
    return polar_angle,azimuth_angle 
def spher2xyz(polar_angle,azimuth_angle,deg=False):
    if deg:
        #polar_angle*=np.pi/180
        #azimuth_angle*=np.pi/180   
        z=np.cos(polar_angle*np.pi/180)
        xy=np.sin(polar_angle*np.pi/180)
        y=np.sin(azimuth_angle*np.pi/180)*xy
        x=np.cos(azimuth_angle*np.pi/180)*xy
    else:
        z=np.cos(polar_angle)
        xy=np.sin(polar_angle)
        y=np.sin(azimuth_angle)*xy
        x=np.cos(azimuth_angle)*xy

    return np.vstack((x,y,z)).T

def genprojgrid(oris,gdata=None,nump=1001,proj='equalarea',method2='linear',gdout=False,poris=None,minmax='notfull'):
    #project


    if poris is None:
        if proj=='stereo':
            poris=stereoprojection_directions(oris)
        elif proj=='equalarea':
            poris=equalarea_directions(oris)
        
    
    if minmax=='full':
        if proj=='stereo':
            minv=-1
            maxv=1
            minv2=-1
            maxv2=1        
        elif proj=='equalarea':
            minv=-np.sqrt(2)
            maxv=np.sqrt(2)
            minv2=-np.sqrt(2)
            maxv2=np.sqrt(2)
    else:
        minv=min(poris[0,:])
        maxv=max(poris[0,:])
        minv2=min(poris[1,:])
        maxv2=max(poris[1,:])

    
    #make regular grid
    grid_x, grid_y = np.mgrid[minv:maxv:nump*1j, minv2:maxv2:nump*1j]
    #Grided data
    gdout=True
    if gdata is None:
        gdata=poris[1,:].flatten()*0+1
        gdout=False
    gdata=griddata((poris[0,:].flatten(),poris[1,:].flatten()), gdata, (grid_x, grid_y), method=method2)
    #numerical mask of the hemisphere
    nummask = np.nan_to_num(gdata*0+1,nan=0.0)
    #boolean mask of the hemisphere
    mask = np.nan_to_num(gdata*0,nan=1.0).astype(bool)
    #masked gridded data
    gdata=np.nan_to_num(gdata,nan=0.0)
    grid_z =np.ma.array(gdata,mask=mask,fill_value=0)
    #grid_z1 = gdata*nummask
    if gdout:
        return grid_x,grid_y, grid_z, nummask, mask
    else:
        return grid_x,grid_y, nummask, mask


def gen_dirs_norms(L, Lr, uvws,hkls, R2Proj=np.eye(3),symops=None,recsymops=None,hemisphere = "upper", **kwargs):
    #generates all symmetry equivalent directions and normals and their projections
    normals=[]
    
    for hkl in hkls:
            nv=R2Proj.dot(Lr.dot(hkl))
            nv/=np.linalg.norm(nv)
            isin=False
            for n in normals:
                if list(n['hkl']) == list(hkl):#np.linalg.norm(n['vector']-nvs)<1e-10:
                    isin=True
                    break
            if not isin:
                
                normals.append({'vector':nv,'hkl':hkl,'hklf':hkl,'label':str(hkl).replace('[','(').replace(']',')').replace(' ',''),
                                'equalarea':equalarea_directions(nv),'stereo':stereoprojection_directions(nv),
                               'equalarea plane':equalarea_planes(nv,hemisphere=hemisphere),
                                'stereo plane':stereoprojection_planes(nv,hemisphere=hemisphere),'textshift':[0,0]})
                if not recsymops is None:
                    for rs in recsymops:
                        
                        hklsym=np.round(rs.dot(hkl))
                        isin=False
                        for n in normals:
                            if True:#n['hklf']==hkl:
                                if list(hklsym) == list(n['hkl']):#np.linalg.norm(n['vector']-nvs)<1e-10:
                                    isin=True
                                    break
                        if not isin:
                            #print(list(hklsym))
                            #print(hkl)
                            #print('------------------------------')
                            nvs=R2Proj.dot(rs.dot(Lr.dot(hkl)))
                            nvs/=np.linalg.norm(nvs)
                            normals.append({'vector':nvs,'hkl':list(hklsym),'hklf':hkl,'label':str(hklsym).replace('[','(').replace(']',')').replace(' ',''),
                                            'equalarea':equalarea_directions(nvs),'stereo':stereoprojection_directions(nvs),
                                           'equalarea plane':equalarea_planes(nvs,hemisphere=hemisphere),
                                           'stereo plane':stereoprojection_planes(nv,hemisphere=hemisphere),'textshift':[0,0]})
            #print('=============================================')  
    dirs=[]
    for uvw in uvws:
            dv=R2Proj.dot(L.dot(uvw))
            dv/=np.linalg.norm(dv)
            isin=False
            for d in dirs:
                if list(d['uvw']) == list(uvw):#np.linalg.norm(n['vector']-nvs)<1e-10:
                    isin=True
                    break
            if not isin:
                dirs.append({'vector':dv,'uvw':uvw,'uvwf':uvw,'label':str(uvw),
                                'equalarea':equalarea_directions(dv),'stereo':stereoprojection_directions(dv),'textshift':[0,0]})
                if not symops is None:
                    for rs in symops:
                        #dvs=rs.dot(dv)
                        uvwsym=np.round(rs.dot(uvw))
                        isin=False
                        for d in dirs:
                            if True:#d['uvwf']==uvw:
                                if list(uvwsym) == list(d['uvw']):#np.linalg.norm(d['vector']-dvs)<1e-10:
                                    isin=True
                                    break
                        if not isin:
                            dvs=R2Proj.dot(rs.dot(L.dot(uvw)))
                            dvs/=np.linalg.norm(dvs)
                            dirs.append({'vector':dvs,'uvw':list(uvwsym),'uvwf':uvw,'label':str(uvwsym).replace('[','(').replace(']',')').replace(' ',''),
                                            'equalarea':equalarea_directions(dvs),'stereo':stereoprojection_directions(dvs),'textshift':[0,0]})
                        
    return dirs,normals


def fullcirc_hist(Mats, Dr=[0,0,1], symops=None, equalarea=False, scale='sqrt', nlevels=10, lvls=None,bins=128, ax=None, title=None, ret=False, 
                  kernel=False,  weights=None,Lim=None,interp=True,interpn=1000, smooth=False, vmin=None, 
                  bandwidth=None, vmax=None,colorbar=True,ticks=None, R2Proj=None,contour=True, mrd=False, **kwargs):
    #generating inverse poles of Dr from orientation matrices Mats
    Dr=np.array(Dr)/np.linalg.norm(Dr)
    if symops is None:
        Mr = np.reshape(Mats, (Mats.shape[0]*Mats.shape[1],Mats.shape[2]))
        data = Mr.dot(Dr)
        data = np.reshape(data,(int(data.shape[0]/3),3)).T
    else:
        Drs = set([tuple(v) for v in np.dot(symops, Dr)])
        Drs=np.asarray(list(Drs))
        data=np.tensordot(Mats, Drs.T, axes=[[-1], [-2]]).transpose([0, 2, 1])
        data = data.reshape(-1,3).T
        
    #matrix Rmat time aray of matrices Mats[N,3,3]: (to be verified!)
    #Mr=np.reshape(Mats, (Mats.shape[0]*Mats.shape[1],Mats.shape[2])).T
    #Mr=Rmat.T.dot(Mr)
    #Mats=Mr.T.reshape(-1,3,3)
    
    
    
    #print(data.shape)
    #Changing the coordinate system to align projection x,y axes as we want
    if R2Proj is not None:
        data = R2Proj.dot(data)
    dataini=data.copy() 
    #inverting vectors projecting to lower hemisphere
    data[:,data[2,:]<0]=-1*data[:,data[2,:]<0]
    data = data[:,data[2,:]>=0]
    #plotting projection circles and setting corresponding radius of the circe (lim)
    #calculating projection of data (data->spsel)
    if equalarea:
        proj='equalarea'
        lim=np.sqrt(2)
        spsel = equalarea_directions(data)
        if Lim=='half':
            fig,ax = schmidtnet_half(ax=ax,basedirs=False)     
        elif Lim=='tri':
            fig,ax = stereotriangle(ax=ax,basedirs=False,equalarea=True)   
        else:
            fig,ax = schmidtnet(ax=ax,basedirs=False)
    else:
        proj='stereo'
        lim=1.
        spsel = stereoprojection_directions(data)
        if Lim=='half':
            fig,ax = wulffnet_half(ax=ax,basedirs=False)       
        elif Lim=='tri':
            fig,ax = stereotriangle(ax=ax,basedirs=False,equalarea=False)
        else:
            fig,ax = wulffnet(ax=ax,basedirs=False)
    if title !='':
        ax.title.set_text(title)       
        
    #calculating weighted histogram of the projected data withing the limit of the circle 
    hist, yedges, xedges = np.histogram2d(spsel[1,:], spsel[0,:], bins=bins,range=[[-lim, lim], [-lim, lim]], weights= weights)
    #histraw=hist.copy()
    
#    if scale=='sqrt':
#        hist = hist**0.5
#    elif scale=='log':
#        hist = np.log(hist)
        
    #nlevels=nlevels
    #lvls = np.linspace(0, np.max(hist), nlevels)
    
    #generating X, Y grid aligned with bins
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2.,
                       (yedges[:-1] + yedges[1:])/2.)
    
    #circle = X**2 + Y**2 >= 2.0
    #histraw[circle] = np.nan  
    #c=ax.contourf(histraw, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], **kwargs)
    if kernel:
        #spherical coordinates theta (elevation), phi of inverse poles
        xy = dataini[0,:]**2 + dataini[1,:]**2
        theta = np.arctan2(np.sqrt(xy), dataini[2,:])#elevation
        phi=np.arctan2(dataini[1,:], dataini[0,:])

        if bandwidth is None:
            bandwidth=0.15
        #calculation of the logarithm of kernel density function of inverse poles using corresponding spherical coordinates theta (elevation), phi
        #I do not know why SphericalKDE provides logarithm
        logkde = SphericalKDE(phi, theta, bandwidth=bandwidth,weights=weights)
        
        #generation of orientation whithin whole space where we will evaluate logkde
        dangle=5.0
        oris=genori(dangle=dangle,hemi='both', tol=1e-2, rot=np.eye(3), half='no')
        xy = oris[0,:]**2 + oris[1,:]**2
        thtg = np.arctan2(np.sqrt(xy), oris[2,:])#elevation==polar angle
        phig=np.arctan2(oris[1,:], oris[0,:])#azimuth angle
        #evaluation of kde for all orientations
        H=np.exp(logkde(phig, thtg))    
        
        if mrd:
            #Normalization of H for MRD
            #normalizing data to multiples of random distribution
            #hist/(sum(hist)*pxarea)*numpixels*pxarea  
            #integral Hds=1
            Norm=(H*np.sin(thtg)*(dangle*np.pi/180)**2/4/np.pi).sum()
            H/=Norm
            #H=H/H.sum()*H.shape[0]
            #check normalization
            #print((H*np.sin(thtg)*(dangle*np.pi/180)**2/4/np.pi).sum())
            
        #interpolation of kde over regular square grid on the projection circle - it is masked or nan outside the circle
        X,Y, hist, nummask, mask=genprojgrid(oris[:,oris[2,:]>=0],gdata=H[oris[2,:]>=0],nump=1001,proj=proj,method2='linear',gdout=True)
        #if mrd:
        #    hist=hist/np.nansum(hist)*np.where(~np.isnan(hist.data))[0].shape[0]
        bins=1001
        #if True:
        #hist=hist.data
        #k = scipy.stats.gaussian_kde([spsel[0,:], spsel[1,:]],weights= weights)
        #hist = k(np.vstack([X.flatten(), Y.flatten()]))
        #hist = hist.reshape(X.shape)
        #print(hist)
    #print(hist)
    #refine histogram by interpolation
    if interp and not kernel:
        #interpolation of histogram data to increase spatial resolution thus making nicer. It is faster and smoother than increasing number of bins 
        xedges=np.linspace(((xedges[:-1] + xedges[1:])/2.)[0],((xedges[:-1] + xedges[1:])/2.)[-1],interpn)
        yedges=np.linspace(((yedges[:-1] + yedges[1:])/2.)[0],((yedges[:-1] + yedges[1:])/2.)[-1],interpn)
        grid_x, grid_y = np.meshgrid(xedges,yedges)    
    
        hist = scipy.interpolate.griddata((X.flatten(), Y.flatten()), hist.flatten(), (grid_x, grid_y), method='linear')
        bins=interpn
        X=grid_x
        Y=grid_y

    #print(hist)
    #hist[np.where((hist<0) & (np.abs(hist)<1e-10))]=0
    #histogram scaling - should not be used if MRD required
    hist[np.where(hist<0)]=0
    if scale=='sqrt':
        hist = hist**0.5
    elif scale=='log':
        hist = np.log(hist)
    #masking data on the grid points outside the projection circle
    if equalarea:
        circle = X**2 + Y**2 >= 2.0
        xlim=[-np.sqrt(2)*1.05,np.sqrt(2)*1.05]
        ylim=[-np.sqrt(2)*1.05,np.sqrt(2)*1.05]
    else:
        circle = X**2 + Y**2 >= 1
        xlim=[-1.05,1.05]
        ylim=[-1.05,1.05]
    hist[circle] = np.nan  
    
    #normalizing data to multiples of random distribution
    #hist/(sum(hist)*pxarea)*numpixels*pxarea
    if mrd and not kernel:
        hist=hist/np.nansum(hist)*np.where(~np.isnan(hist.data))[0].shape[0]

    #calculating levels for isolines and contourse    
    nlevels=nlevels
    if vmin is None:
        vmin=0
    if vmax is None:
        vmax=np.max(hist[~np.isnan(hist)])
    if lvls is None:
        lvls = np.linspace(vmin, vmax, nlevels)
    else:
        nlevels=lvls.shape[0]
    kwargs = {}
    kwargs['levels'] = lvls[1:]
    
    #cutting projection area to half circle of stereotriangle
    if Lim is not None:
        if Lim=='half':
            ylim[0]=-1e-5
            cut=Y<-1e-5
            hist[cut] = np.nan 
        if Lim=='tri':
            oristri=genoritri(resolution=0.1,mesh="spherified_cube_edge")
            grid_x,grid_y, nummask, mask=genprojgrid(oristri,nump=bins,proj=proj,method2='linear',gdout=False)
            if kernel:
                hist[nummask==0]=np.nan
            else:
                hist[nummask.T==0]=np.nan
            
            
    #print(hist)
    #Only visualization below
    if kernel:
        if smooth:
            sc=ax.pcolor(X, Y,hist,vmin=vmin,vmax=vmax)
        else:
            #print('ok')
            #sc=ax.contourf(histraw, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], **kwargs)
            sc=ax.contourf(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], **kwargs)
        if contour:
            CS=ax.contour(X, Y,hist,levels=lvls,colors='k')
            ax.clabel(CS, fontsize=9, inline=1,colors='k')

    else:
        if smooth:
            sc=ax.pcolor(X, Y,hist,vmin=vmin,vmax=vmax)
        else:
            sc=ax.contourf(hist, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], **kwargs)        
        if contour:
            CS=ax.contour(X, Y,hist,levels=nlevels,colors='k')
            ax.clabel(CS, fontsize=9, inline=1,colors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if colorbar:
        pos = ax.get_position()
        fac=0.75
        cbarh=0.04
        cbar_ax = fig.add_axes([pos.width*(1-fac)/2+pos.x0, pos.y0+cbarh-0.1, pos.width*fac,cbarh])
        #cbar_ax =AX[1]
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')   
        if mrd:
            cbar.ax.set_xlabel("MRD")
        else:
            cbar.ax.set_xlabel("Probability")
        #cbar.ax.set_xlim([0,30])
        xticks=(cbar.ax.get_xticks()[0:-1]+cbar.ax.get_xticks()[1:])/2
        #cbar.ax.set_xticks(np.round(xticks*100)/100)
        if ticks is not None:
            cbar.ax.set_xticks(ticks)

    if ret:
        return hist, xedges, yedges, fig, ax
#convert projected points into xyz
def rp2xyz(r,p):
    npatan2d = lambda x,y: 180.*np.arctan2(x,y)/np.pi
    z = npcosd(r)
    xy = np.sqrt(1.-z**2)
    return xy*npsind(p),xy*npcosd(p),z

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


def equalarea_planes(normals,arclength=360.,iniangle=0.,hemisphere="both"):
    #%normals = [x1,x2,...,xn;y1,y2,...,yn;z1,z2,...,zn];
    #%varargin{1} arclength in deg
    #normals = np.transpose(np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]))
    #
    if len(normals.shape)==1:
        normals = np.expand_dims(normals,axis=1)

    normals = normals.astype(float)
    normals /= np.sqrt((normals ** 2).sum(0))

    proj_normals = equalarea_directions(normals)

    idxs = np.where(abs(normals[0,:])+abs(normals[1,:])==0)[0]
    
    inplanedirs = np.vstack((-normals[1,:],normals[0,:],np.zeros(normals[0,:].shape)));
    inplanedirs[:,idxs] = np.vstack((np.zeros(normals[0,idxs].shape), -normals[2,idxs],normals[1,idxs]));
    
    inplanedirs /= np.sqrt((inplanedirs ** 2).sum(0))

    thirdaxis=np.cross(normals,inplanedirs,axisa=0,axisb=0,axisc=0)
#    thirdaxis = np.vstack((normals[1,:]*inplanedirs[2,:]-normals[2,:]*inplanedirs[1,:],
#                           -1*(normals[0,:]*inplanedirs[2,:]-normals[2,:]*inplanedirs[0,:]),
#                           normals[0,:]*inplanedirs[1,:]-normals[1,:]*inplanedirs[0,:]));
    t=np.linspace(iniangle,iniangle+arclength,180*2+1)*np.pi/180;
    basicarc = np.vstack((np.cos(t),np.sin(t),np.zeros(t.shape)));
    
    proj_planes=[];
    Zdir=[]
    #print(hemisphere)
    for i in range(0,normals.shape[1]):
        Rot2Global = np.transpose(np.vstack((inplanedirs[:,i],thirdaxis[:,i],normals[:,i])));
        Ccp = np.matmul(Rot2Global,basicarc);
        Zdir=Ccp[2]
        if hemisphere == "both":            
            Ds = equalarea_directions(Ccp)
        elif hemisphere == "triangle":  
            #idxs = np.where(Ccp[2,:]>=0)[0]
            Ds=equalarea_intotriangle(Ccp)#[:,idxs])
        else:
            if hemisphere == "upper":
                idxs = np.where(Ccp[2,:]>=0)[0]
                Ds = equalarea_directions(Ccp[:,idxs])
            elif hemisphere == "lower":
                idxs = np.where(Ccp[2,:]<=0)[0]
                Ds = equalarea_directions(Ccp[:,idxs])
        proj_planes.append(Ds)
    
    if len(proj_planes)==1:
        return proj_planes[0]
    else:          
        return proj_planes
def stereo2xyz(projdir):
    r,p = 2.*np.arctan(np.sqrt(projdir[0]**2+projdir[1]**2)),np.arctan2(projdir[1],projdir[0])
    z = np.cos(r)
    xy = np.sqrt(1.-z**2)
    return xy*np.cos(p),xy*np.sin(p),z

def equalarea2xyz(projdir):
    if projdir[0]==0:
        an=np.pi/2
    else:
        an=np.arctan(projdir[1]/projdir[0])
    dirsxy1=np.cos(an)
    dirsxy2=np.sin(an)
    if projdir[0]==0:
        alpha=np.arcsin(projdir[1]/dirsxy2/2)*2
    else:
        alpha=np.arcsin(projdir[0]/dirsxy1/2)*2
    z=np.cos(alpha)
    if projdir[0]==0:
        x=0
        y=np.sin(alpha)
    elif projdir[1]==0:
        y=0
        x=np.sin(alpha)
    else:
        x=dirsxy1*np.sin(alpha)
        y=dirsxy2*np.sin(alpha)
    print(alpha*180/np.pi)
    return x,y,z


def equalarea_directions(dirs):
    #dirs = [x1,x2,...,xn;y1,y2,...,yn;z1,z2,...,zn];
    #example: dirs = [0,1,2,3;1,2,3,0;0,3,2,1]
    
    #normalize and project
       
    if len(dirs.shape)==1:
        dirs = np.expand_dims(dirs,axis=1)

    dirs = dirs.astype(float)
    #normalizing dirs
    dirs /= np.sqrt((dirs ** 2).sum(0))
    dirsxy = dirs[0:2,:];
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



def wulffnet(ax=None,basedirs=False,facecolor=(210./255.,235./255.,255./255.)):
    if ax==None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()
    if basedirs:
        basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]);
        basicdirections = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,1,1],[-1,1,1],[1,-1,1],[-1,-1,1],[0,1,1],[1,0,1]]);
        #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
        basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]);
        basicdirectionstext = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,1,1],[-1,1,1],[1,-1,1],[-1,-1,1],[0,1,1],[1,0,1]]);
        #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];



    #longitude lines
    #fig, ax = plt.subplots()
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax.plot(0, 0, 'k+')
    circ = plt.Circle((0, 0), 1.0, facecolor=facecolor, edgecolor='black')
    ax.add_patch(circ)

    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
  # equal aspect ratio
    ax.axis('off')  # remove the box
    ##plt.show()
    
    t=np.linspace(0,180,180*2+1)*np.pi/180;
    xc = np.sin(t);
    yc = np.cos(t);
    AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
    for an in AltitudeAngle:
        RotY = passive_rotation(an, 'y')
        Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

        proj_dirs = stereoprojection_directions(Ccp)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

    #Latitude Lines
    LatitudeAngle = np.linspace(-90,90,37)*np.pi/180; 
    #t=np.linspace(0,180,360*2+1)*np.pi/180;
    zc = np.sin(t);
    xc = np.cos(t);
    for an in LatitudeAngle:#[0.]:#LatitudeAngle:
        Rmeridian = np.cos(an);
        px = Rmeridian*xc;
        py = np.sin(an)*np.ones(t.shape);
        pz = Rmeridian*zc;

        proj_dirs = stereoprojection_directions(np.vstack((px,py,pz)))

        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
    
    if basedirs:
        an=45.*np.pi/180.;
        an=0.*np.pi/180.;
        Rotz = active_rotation(an, 'z')
        an=np.arccos(1/np.sqrt(3));
        an=0.*np.pi/180.;
        Rotx = active_rotation(an, 'x')
        dirs = np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections))
        proj_dirs = stereoprojection_directions(dirs)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='b',marker='o',linestyle='')
        for diri,proj_diri in zip(basicdirectionstext,np.transpose(proj_dirs)):
            ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))

    ax.set_xlim((-1.05,1.05))
    ax.set_ylim((-1.05,1.05))

    return fig,ax

def wulffnet_half(ax=None,basedirs=False,facecolor=(210./255.,235./255.,255./255.)):
    if ax==None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()
    if basedirs:
        basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
        #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
        basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
        #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];



    #longitude lines
    #fig, ax = plt.subplots()
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax.plot(0, 0, 'k+')
    
    
    
    
    w1 = Wedge((0,0), 1.0, 0, 180, fc=facecolor, edgecolor='black')
    ax.add_artist(w1)
    
            
    
    
#    circ = plt.Circle((0, 0), 1.0, facecolor=(210./255.,235./255.,255./255.), edgecolor='black')
#    ax.add_patch(circ)

    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
  # equal aspect ratio
    ax.axis('off')  # remove the box
   ##plt.show()
    
    t=np.linspace(0,90,180*2+1)*np.pi/180;
    xc = np.sin(t);
    yc = np.cos(t);
    AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
    for an in AltitudeAngle:
        RotY = passive_rotation(an, 'y')
        Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

        proj_dirs = stereoprojection_directions(Ccp)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

    #Latitude Lines
    LatitudeAngle = np.linspace(0,90,37)*np.pi/90; 
    #t=np.linspace(0,180,360*2+1)*np.pi/180;
    zc = np.sin(t);
    xc = np.cos(t);
    for an in LatitudeAngle:#[0.]:#LatitudeAngle:
        Rmeridian = np.cos(an);
        px = Rmeridian*xc;
        py = np.sin(an)*np.ones(t.shape);
        pz = Rmeridian*zc;

        proj_dirs = stereoprojection_directions(np.vstack((px,py,pz)))

        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
    
    if basedirs:
        an=45.*np.pi/180.;
        an=0.*np.pi/180.;
        Rotz = active_rotation(an, 'z')
        an=np.arccos(1/np.sqrt(3));
        an=0.*np.pi/180.;
        Rotx = active_rotation(an, 'x')
        dirs = np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections))
        proj_dirs = stereoprojection_directions(dirs)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='b',marker='o',linestyle='')
        for diri,proj_diri in zip(basicdirectionstext,np.transpose(proj_dirs)):
            ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))

    #ax.set_xlim((-1.05,1.05))
    #ax.set_ylim((-1.05,1.05))
    dd=0.05
    RR=1
    ax.set_xlim([-RR-dd,RR+dd])
    ax.set_ylim([0-dd,RR+dd])    


    return fig,ax
def wulffnet_quarter(ax=None,basedirs=False):
    if ax==None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()
    if basedirs:
        basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
        #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
        basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
        #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];



    #longitude lines
    #fig, ax = plt.subplots()
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax.plot(0, 0, 'k+')
    
    
    
    
    w1 = Wedge((0,0), 1.0, 0, 90, fc=(210./255.,235./255.,255./255.), edgecolor='black')
    ax.add_artist(w1)
    
            
    
    
#    circ = plt.Circle((0, 0), 1.0, facecolor=(210./255.,235./255.,255./255.), edgecolor='black')
#    ax.add_patch(circ)

    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
  # equal aspect ratio
    ax.axis('off')  # remove the box
    #plt.show()
    
    t=np.linspace(0,90,180*2+1)*np.pi/180;
    xc = np.sin(t);
    yc = np.cos(t);
    AltitudeAngle = np.linspace(0,90,19)*np.pi/180;
    for an in AltitudeAngle:
        RotY = passive_rotation(an, 'y')
        Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

        proj_dirs = stereoprojection_directions(Ccp)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

    #Latitude Lines
    LatitudeAngle = np.linspace(0,45,19)*np.pi/90; 
    t=np.linspace(0,90,180*2+1)*np.pi/180;
    zc = np.sin(t);
    xc = np.cos(t);
    for an in LatitudeAngle:#[0.]:#LatitudeAngle:
        Rmeridian = np.cos(an);
        px = Rmeridian*xc;
        py = np.sin(an)*np.ones(t.shape);
        pz = Rmeridian*zc;

        proj_dirs = stereoprojection_directions(np.vstack((px,py,pz)))

        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
    
    if basedirs:
        an=45.*np.pi/180.;
        an=0.*np.pi/180.;
        Rotz = active_rotation(an, 'z')
        an=np.arccos(1/np.sqrt(3));
        an=0.*np.pi/180.;
        Rotx = active_rotation(an, 'x')
        dirs = np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections))
        proj_dirs = stereoprojection_directions(dirs)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='b',marker='o',linestyle='')
        for diri,proj_diri in zip(basicdirectionstext,np.transpose(proj_dirs)):
            ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))


    return fig,ax

def schmidtnet(ax=None,basedirs=False,facecolor=(210./255.,235./255.,255./255.)):
    if ax==None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()

    if basedirs:
        basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
        #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
        basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
        #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];



    #longitude lines
    #fig, ax = plt.subplots()
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax.plot(0, 0, 'k+')
    equaarea_factor = 2./np.sqrt(2)
    circ = plt.Circle((0, 0), equaarea_factor*1.0, facecolor=facecolor, edgecolor='black')
    ax.add_patch(circ)

    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
    ax.axis('off')  # remove the box
    #plt.show()
    
    t=np.linspace(0,180,180*2+1)*np.pi/180;
    xc = np.sin(t);
    yc = np.cos(t);
    AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
    for an in AltitudeAngle:
        RotY = passive_rotation(an, 'y')
        Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

        proj_dirs = equalarea_directions(Ccp)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

    #Latitude Lines
    LatitudeAngle = np.linspace(-90,90,37)*np.pi/180; 
    #t=np.linspace(0,180,360*2+1)*np.pi/180;
    zc = np.sin(t);
    xc = np.cos(t);
    for an in LatitudeAngle:#[0.]:#LatitudeAngle:
        Rmeridian = np.cos(an);
        px = Rmeridian*xc;
        py = np.sin(an)*np.ones(t.shape);
        pz = Rmeridian*zc;

        proj_dirs = equalarea_directions(np.vstack((px,py,pz)))

        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
    
    if basedirs:
        an=45.*np.pi/180.;
        an=0.*np.pi/180.;
        Rotz = active_rotation(an, 'z')
        an=np.arccos(1/np.sqrt(3));
        an=0.*np.pi/180.;
        Rotx = active_rotation(an, 'x')
        dirs = np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections))
        proj_dirs = equalarea_directions(dirs)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='b',marker='o',linestyle='')
        for diri,proj_diri in zip(basicdirectionstext,np.transpose(proj_dirs)):
            ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))

    return fig,ax


def wulffnet_regular_grid(ax,dangle,dirout=False, plot=True):
    #dphi=10.deg
    #dtheta=10.deg
    #dangle = 10.
    
    Phi1=np.linspace(0.,360.-dangle,int(360./dangle))
    Phi2=np.linspace(0.,180.-dangle,int(180./dangle))
    GridX=[];
    GridY=[];
    Dc=[1.,0.,0.];
    dirs=[]
    for phi1 in Phi1:
        for phi2 in Phi2:        
            RotZ = active_rotation(phi1, 'z', deg=True) 
            RotY = active_rotation(phi2, 'y', deg=True)
            Ds = np.matmul(RotY,RotZ).dot(Dc)
            dirs.append(Ds)
            proj_Ds = stereoprojection_directions(Ds)
            GridX.append(proj_Ds[0,0])
            GridY.append(proj_Ds[1,0])
                
    

    #fig,ax = wulffnet()
    if plot:
        ax.plot(GridX,GridY,'.',color='r',markersize=1)
    if dirout:
        return GridX,GridY,dirs
    else:
        return GridX,GridY

def schmidtnet_half(ax=None,basedirs=False,facecolor=(210./255.,235./255.,255./255.)):
    if ax==None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()
    if basedirs:
        basicdirections = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
        #basicdirections = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];
        basicdirectionstext = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1],[1,0,2]]);
        #basicdirectionstext = [1,0,0;0,1,0;0,0,1;1,1,0;1,1,1;0,1,1;1,0,1;1,1,-2;-1,-1,2;1,-1,0;-1,1,0];



    #longitude lines
    #fig, ax = plt.subplots()
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax.plot(0, 0, 'k+')
    equaarea_factor = 2./np.sqrt(2)

    w1 = Wedge((0,0), equaarea_factor*1.0, 0, 180, fc=facecolor, edgecolor='black')
    ax.add_artist(w1)

    #circ = plt.Circle((0, 0), equaarea_factor*1.0, facecolor=facecolor, edgecolor='black')
    #ax.add_patch(circ)

    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
    ax.axis('off')  # remove the box
    #plt.show()
    
    t=np.linspace(0,90,180*2+1)*np.pi/180;
    xc = np.sin(t);
    yc = np.cos(t);
    AltitudeAngle = np.linspace(0,180,37)*np.pi/180;
    for an in AltitudeAngle:
        RotY = passive_rotation(an, 'y')
        Ccp = np.matmul(RotY,np.vstack((xc,yc,np.zeros(yc.shape))))

        proj_dirs = equalarea_directions(Ccp)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')

    #Latitude Lines
    LatitudeAngle = np.linspace(0,180,37)*np.pi/180; 
    #t=np.linspace(0,180,360*2+1)*np.pi/180;
    zc = np.sin(t);
    xc = np.cos(t);
    for an in LatitudeAngle:#[0.]:#LatitudeAngle:
        Rmeridian = np.cos(an);
        px = Rmeridian*xc;
        py = np.sin(an)*np.ones(t.shape);
        pz = Rmeridian*zc;

        proj_dirs = equalarea_directions(np.vstack((px,py,pz)))

        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--')
    
    if basedirs:
        an=45.*np.pi/180.;
        an=0.*np.pi/180.;
        Rotz = active_rotation(an, 'z')
        an=np.arccos(1/np.sqrt(3));
        an=0.*np.pi/180.;
        Rotx = active_rotation(an, 'x')
        dirs = np.matmul(np.matmul(Rotx,Rotz),np.transpose(basicdirections))
        proj_dirs = equalarea_directions(dirs)
        ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='b',marker='o',linestyle='')
        for diri,proj_diri in zip(basicdirectionstext,np.transpose(proj_dirs)):
            ax.text(0.03+proj_diri[0],0.03+proj_diri[1],str(diri))

    return fig,ax


def wulffnet_regular_grid(ax,dangle):
    #dphi=10.deg
    #dtheta=10.deg
    #dangle = 10.
    
    Phi1=np.linspace(0.,360.-dangle,int(360./dangle))
    Phi2=np.linspace(0.,180.-dangle,int(180./dangle))
    GridX=[];
    GridY=[];
    Dc=[1.,0.,0.];
    for phi1 in Phi1:
        for phi2 in Phi2:        
            RotZ = active_rotation(phi1, 'z', deg=True) 
            RotY = active_rotation(phi2, 'y', deg=True)
            Ds = np.matmul(RotY,RotZ).dot(Dc)
            proj_Ds = stereoprojection_directions(Ds)
            GridX.append(proj_Ds[0,0])
            GridY.append(proj_Ds[1,0])
                
    

    #fig,ax = wulffnet()
    ax.plot(GridX,GridY,'.',color='r',markersize=1)
    
    return GridX,GridY
def schmidt_regular_grid(ax,Na=72,Nr=20,plot=True):
    dphi1=360/Na
    phi1=np.linspace(0,360-dphi1,int(Na))
    R=equalarea_directions(np.array([1,0,0]))[0,0]
    TotalArea=np.pi*R**2
    dr=R/(Nr+0.5)
    r=np.linspace(0,R-dr/2,int(Nr+1))
#    GridX=[0.];
#    GridY=[0.];
#    Weight= np.pi*(r[1]/2)**2/TotalArea
    AreaRatio=[]
    for ri in r:
        Nari=int(Na*(ri/r[-1]))
        
        if Nari<8:
            Nari=8
        else:
            Nari=Nari-Nari%8    
        #print(Nari)
        #Nari=8
        dphi1=360./(Nari)
        phi1=np.linspace(0,360-dphi1,Nari)
        phi1=np.linspace(dphi1/2,360-dphi1/2,Nari)
        #phi1=phi1[0:-1]
        #print(2*np.pi*((ri+dr/2)**2/2-(ri-dr/2)**2/2))
        #tot=0
        for phi1i in phi1:        
            if ri==0:
                GridX=[0.];
                GridPhi=[0.]
                GridY=[0.];
                GridR=[r[1]-dr/2]
                GridR=[ri]
                AreaRatio= [np.pi*(r[1]-dr/2.)**2/TotalArea]
            else:
                GridX.append(ri*np.cos(phi1i*np.pi/180.))
                GridY.append(ri*np.sin(phi1i*np.pi/180.))
                AreaRatio.append(dphi1*np.pi/180.*((ri+dr/2)**2/2.-(ri-dr/2)**2/2.)/TotalArea)
                GridR.append(ri)
                phi=np.arctan2(GridY[-1],GridX[-1])*180./np.pi
                GridPhi.append(phi)
                #tot+=dphi1*np.pi/180.*((ri+dr/2)**2/2-(ri-dr/2)**2/2)
        #print(tot)
    #sum(AreaRatio)
    if plot:
        ax.plot(GridX,GridY,'.',color='r',markersize=1)
    
    return GridX,GridY,GridR,GridPhi,AreaRatio

def pf_cmap02(GridX,GridY,GridR,GridPhi,AreaRatio,Intensity,NoCont=10,GridSize=1000,cmap='jet',method='cubic'):
    
    
#    r = [np.sqrt(x**2 + y**2) for x,y in zip(GridX,GridY)]
#    theta = [np.arctan2(y,x) for x,y in zip(GridX,GridY)]
#    theta = theta-min(theta)
#    equaarea_factor = 2./np.sqrt(2)
#    
#    ri = np.linspace(0,equaarea_factor,100)
#    thetai = np.linspace(0,2.0*np.pi,100)
#    thetai,ri = np.mgrid[0:2*np.pi:GridSize, 0:equaarea_factor:GridSize]
    
    equaarea_factor = 2./np.sqrt(2)
    xi,yi = np.mgrid[-equaarea_factor:equaarea_factor:GridSize, -equaarea_factor:equaarea_factor:GridSize]
    Ii = scipy.interpolate.griddata((GridX,GridY),Intensity,(xi,yi),method=method)
    
    
    #Ii=np.nan_to_num(Ii)
    fig = plt.figure()
    #fig,ax = schmidtnet(basedirs=False,facecolor='white')
    ax = fig.add_subplot(111)
    
    palette = plt.cm.jet
    palette.set_bad ('w',1.0) # Bad values (i.e., masked, set to grey 0.8
    A = np.ma.array ( Ii, mask=np.isnan(Ii))
    
    CS=plt.contourf(xi,yi,A,NoCont,cmap=cmap)
    theta = np.linspace(0,2.0*np.pi,100)
    xc=equaarea_factor*np.cos(theta)
    yc=equaarea_factor*np.sin(theta)
    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
  # equal aspect ratio

    plt.plot(xc,yc, color='black',linewidth=2)
    
    
    
    
    
    
    ax.axis('off')  # remove the box

#    cb = plt.colorbar(p2, cax=ax)
    cb=plt.colorbar(CS)
    ax.set_xlim([-equaarea_factor,1*equaarea_factor])
    ax.set_ylim([-equaarea_factor,1*equaarea_factor])
    #plt.show()
    cb.remove()
    cmin=np.nanmin(Ii)
    cmax=np.nanmax(Ii)
    
    ax2= fig.add_axes(ax.get_position())
    ax2.set_position([0.85, 0.1,0.03, 0.8])
    #ax2.axis('off')  # remove the box
    
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    cmap = mpl.cm.ScalarMappable(
          norm = norm, 
          cmap = cmap)
    cmap.set_array([])
    cb=fig.colorbar(cmap,cax=ax2)
    cb.ax.set_ylabel('Multiple of random distribution')

    textvar1=ax.text(0., equaarea_factor, "Y", size=15, rotation=0.,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(0., 0, 0),
                       fc=(1., 1, 1),
                       ))
    textvar2=ax.text(equaarea_factor, 0.,"X", size=15, rotation=0.,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(0., 0, 0),
                       fc=(1., 1, 1),
                       ))

    plt.tight_layout()
    #plt.show()
    return fig,ax,ax2,cb,xi,yi,A,CS,cmap
    


def pf(gPhi1,gPHI,gPhi2,Dc,lattice,Na=72,Nr=20,syms=True,s=50,facecolor=(210./255.,235./255.,255./255.),plot=True):
    #fig,ax = schmidtnet()
    GridX,GridY,GridR,GridPhi,AreaRatio=schmidt_regular_grid([],Na=Na,Nr=Nr,plot=False)
    
    Intensity = np.array(GridX)*0
    inc=0
    #Dc=[1,0,0]
    Symsall = symmetry_elements(lattice)
    if syms:
        Syms=Symsall
    
    for Phi1,PHI,Phi2 in zip(gPhi1,gPHI,gPhi2):
        inc+=1
        print(str(inc)+'/'+str(len(gPhi1)))
        #R = np.array(np_euler_matrix(Phi1, PHI,Phi2))
        #Phi1,PHI,Phi2 = euler_angles_reduction(Phi1,PHI,Phi2)
        U = np_inverse_euler_matrix(Phi1, PHI,Phi2)
        if not syms:
            Syms = [Symsall[random.randint(0,len(Symsall)-1)]]
        for Sym in Syms:
            #Ri=np.matmul(Sym,R)
            #Phi1,PHI,Phi2=euler_angles_from_matrix(Ri)
            #Phi1,PHI,Phi2 = euler_angles_reduction(Phi1,PHI,Phi2)
            Ds = np.array(U).dot(Sym.dot(Dc))
            proj_Ds = equalarea_directions(Ds)
            phi = np.arctan2(proj_Ds[1],proj_Ds[0])[0]*180./np.pi
            r=np.sqrt(proj_Ds[:,0].dot(proj_Ds[:,0]))
    
            dr=np.array(GridR)-r
            idxmin=np.where(abs(dr)==min(abs(dr)))
            ri=GridR[idxmin[0][0]]
    
            idxr=np.where(np.array(GridR)==ri)[0]
            dtan = np.array(GridPhi)[idxr]-phi
            idxtan = np.where(abs(dtan)==min(abs(dtan)))
            idxmin3=idxr[idxtan[0]][0]
            
    #        dx = np.array(GridX)-proj_Ds[0];
    #        dy = np.array(GridY)-proj_Ds[1];
    #        dr = dx**2+dy**2
    #        idxmin = np.where(dr==min(dr))[0][0]
            #Intensity[idxmin3]+=(1./AreaRatio[idxmin3])
            Intensity[idxmin3]+=(1./(len(Syms)*len(gPhi1)*AreaRatio[idxmin3]))
#            Intensity[idxmin3]+=1./len(gPhi1)
    #        GridX[idxmin]
    #        GridY[idxmin]
    
    
#    Intensity2 = np.array(GridX)*0
#    for Int,Int2,AR in zip()   
    #fig, ax = plt.subplots()
    if plot:
        fig,ax = schmidtnet(basedirs=False,facecolor=facecolor)
        GridX,GridY,GridR,GridPhi,AreaRatio=schmidt_regular_grid(ax,Na=Na,Nr=Nr,plot=True)
    
        plt.scatter(GridX,GridY, c=Intensity, s=s, edgecolor='',zorder=10,cmap='jet')
        cb=plt.colorbar()
        #plt.show()
        return fig,ax,cb,Intensity
    else:
        return GridX,GridY,GridR,GridPhi,AreaRatio,Intensity




def pf_cmap(gPhi1,gPHI,gPhi2,Dc,lattice,Na=72,Nr=20,syms=True,s=50,NoCont=10,GridSize=1000,cmap='jet',method='cubic'):
    
    GridX,GridY,GridR,GridPhi,AreaRatio,Intensity=pf(gPhi1,gPHI,gPhi2,Dc,lattice,Na=Na,Nr=Nr,syms=syms,s=s,facecolor="None",plot=False)
    
    r = [np.sqrt(x**2 + y**2) for x,y in zip(GridX,GridY)]
    theta = [np.arctan2(y,x) for x,y in zip(GridX,GridY)]
    theta = theta-min(theta)
    equaarea_factor = 2./np.sqrt(2)
    
    ri = np.linspace(0,equaarea_factor,100)
    thetai = np.linspace(0,2.0*np.pi,100)
    thetai,ri = np.mgrid[0:2*np.pi:GridSize, 0:equaarea_factor:GridSize]
    # grid the data.
    Ii = scipy.interpolate.griddata((np.concatenate((theta-2*np.pi,theta,theta+2*np.pi), axis=0),np.concatenate((r,r,r), axis=0)),
                                    np.concatenate((Intensity,Intensity,Intensity), axis=0),(thetai,ri),method=method)
    
    
    Ii=np.nan_to_num(Ii)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    
        
    CS=plt.contourf(thetai,ri,Ii,NoCont,cmap=cmap)
    plt.plot(np.linspace(0,2.0*np.pi,1000),np.linspace(0,2.0*np.pi,1000)*0+equaarea_factor, color='black',linewidth=2)
    ax.axis('off')  # remove the box

    #cb = plt.colorbar(p2, cax=ax)
    cb=plt.colorbar(CS)
    ax.set_xlim([0,2*np.pi])
    ax.set_ylim([0,1*equaarea_factor])
    #plt.show()
    cb.remove()
    cmin=np.nanmin(Ii)
    cmax=np.nanmax(Ii)
    
    ax2= fig.add_axes(ax.get_position())
    ax2.set_position([0.85, 0.1,0.03, 0.8])
    #ax2.axis('off')  # remove the box
    
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    cmap = mpl.cm.ScalarMappable(
          norm = norm, 
          cmap = cmap)
    cmap.set_array([])
    cb=fig.colorbar(cmap,cax=ax2)
    cb.ax.set_ylabel('Multiple of random distribution')

    textvar1=ax.text(0., 2./np.sqrt(2), "X", size=15, rotation=0.,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(0., 0, 0),
                       fc=(1., 1, 1),
                       ))
    textvar2=ax.text(np.pi/2, 2./np.sqrt(2), "Y", size=15, rotation=0.,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(0., 0, 0),
                       fc=(1., 1, 1),
                       ))

    plt.tight_layout()
    #plt.show()
    return fig,ax,ax2,cb,thetai,ri,Ii,CS,cmap
    

def pf_cmap_cscale(fig,ax2,cmin,cmax,cmap):
    #cmin=0
    #cmax=5
    plt.clim(cmin,cmax)
    
    cmap = mpl.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax),
          cmap = cmap)
    cmap.set_array([])
    cb=fig.colorbar(cmap,cax=ax2)
    cb.ax.set_ylabel('Multiple of random distribution')


    
def ipf(gPhi1,gPHI,gPhi2,Dc,lattice,Na=72,Nr=20,syms=True):
    #fig,ax = schmidtnet()
    GridX,GridY,GridR,GridPhi,AreaRatio=schmidt_regular_grid([],Na=Na,Nr=Nr,plot=False)
    
    Intensity = np.array(GridX)*0
    inc=0
    #Dc=[1,0,0]
    Symsall = symmetry_elements(lattice)
    if syms:
        Syms=Symsall
    
    for Phi1,PHI,Phi2 in zip(gPhi1,gPHI,gPhi2):
        inc+=1
        print(str(inc)+'/'+str(len(gPhi1)))
        #R = np.array(np_euler_matrix(Phi1, PHI,Phi2))
        #Phi1,PHI,Phi2 = euler_angles_reduction(Phi1,PHI,Phi2)
        U = np_euler_matrix(Phi1, PHI,Phi2)
        if not syms:
            Syms = [Symsall[random.randint(0,len(Symsall)-1)]]
        for Sym in Syms:
            #Ri=np.matmul(Sym,R)
            #Phi1,PHI,Phi2=euler_angles_from_matrix(Ri)
            #Phi1,PHI,Phi2 = euler_angles_reduction(Phi1,PHI,Phi2)
            Ds = np.array(U).dot(Sym.dot(Dc))
            proj_Ds = equalarea_directions(Ds)
            phi = np.arctan2(proj_Ds[1],proj_Ds[0])[0]*180./np.pi
            r=np.sqrt(proj_Ds[:,0].dot(proj_Ds[:,0]))
    
            dr=np.array(GridR)-r
            idxmin=np.where(abs(dr)==min(abs(dr)))
            ri=GridR[idxmin[0][0]]
    
            idxr=np.where(np.array(GridR)==ri)[0]
            dtan = np.array(GridPhi)[idxr]-phi
            idxtan = np.where(abs(dtan)==min(abs(dtan)))
            idxmin3=idxr[idxtan[0]][0]
            
    #        dx = np.array(GridX)-proj_Ds[0];
    #        dy = np.array(GridY)-proj_Ds[1];
    #        dr = dx**2+dy**2
    #        idxmin = np.where(dr==min(dr))[0][0]
            #Intensity[idxmin3]+=(1./AreaRatio[idxmin3])
            Intensity[idxmin3]+=(1./(len(Syms)*len(gPhi1)*AreaRatio[idxmin3]))
#            Intensity[idxmin3]+=1./len(gPhi1)
    #        GridX[idxmin]
    #        GridY[idxmin]
    
    
#    Intensity2 = np.array(GridX)*0
#    for Int,Int2,AR in zip()   
    #fig, ax = plt.subplots()
    fig,ax = schmidtnet()
    GridX,GridY,GridR,GridPhi,AreaRatio=schmidt_regular_grid(ax,Na=Na,Nr=Nr,plot=True)

    plt.scatter(GridX,GridY, c=Intensity, s=50, edgecolor='',zorder=10,cmap='jet')
    cb=plt.colorbar()
    #plt.show()
    return fig,ax,cb,Intensity
        
def stereotriangle(ax=None,basedirs=False,equalarea=False,grid=False,resolution=None,gridmarkersize=None,gridmarkercol=None,gridzorder=None,mesh=False):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()
        
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    
    if resolution is None:
        resolution=5
    if gridmarkersize is None:
        gridmarkersize=5
    if gridmarkercol is None:
        gridmarkercol='k'
    if gridzorder is None:
        gridzorder=50000
    
    ax.plot(0, 0, 'k+')
    #circ = plt.Circle((0, 0), 1.0, facecolor=(210./255.,235./255.,255./255.), edgecolor='black')
    #ax.add_patch(circ)

    ax.set_aspect('equal',adjustable='box')  # equal aspect ratio
    ax.axis('off')  # remove the box
    #plt.show()


    #print(max(proj_normals[1,:])/max(proj_normals[0,:]))
    #testing = generate grid of crystallographic direction = integer Miller indexes
    if False:
        an1=np.arctan(1/np.array(list(range(1,8))))
        for ii,a1 in enumerate(an1):
            vertical=[]
            an2=np.arctan(np.linspace(0,1,ii+2)*np.tan(a1))
            an2max=an2[-1]
            an22=np.arctan(1/np.linspace(1,15-ii,15-ii))[::-1]
            an22=np.append([0],an22)
            an22=an22[an22<an2max]
            an22=np.append(an22,an2max)
            for a2 in an22:                
                vertical.append(np.array([np.tan(a1),np.tan(a2),1]))
        
            vertical=np.array(vertical).T
            if equalarea:
                proj_dirs=equalarea_directions(vertical)   
            else:
                proj_dirs=stereoprojection_directions(vertical)   
            ax.plot(proj_dirs[0,:],proj_dirs[1,:],color='r',markeredgecolor='None',marker="o",markersize=10,zorder=50000)

    if mesh:
        dan=resolution
        dan2=resolution
        an1=[dan*(ii+1)/180*np.pi for ii in range(int(45/dan))]
        for ii,a1 in enumerate(an1):
            vertical=[]
            an2=[dan2*(jj)/180*np.pi for jj in range(int((2+ii)*dan/dan2))]
            for a2 in an2:
                vertical.append(np.array([np.tan(a1),np.tan(a2),1]))
            vertical=np.array(vertical).T
            if equalarea:
                proj_dirs=equalarea_directions(vertical)   
            else:
                proj_dirs=stereoprojection_directions(vertical)   
            ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--',zorder=gridzorder)
        
        for jj in range(int(45/dan)+1):
            horizontal=[]
            for ii in range(int(45/dan)-jj+1):
                horizontal.append(np.array([np.tan(dan*(ii+jj)/180*np.pi),np.tan(dan*(jj)/180*np.pi),1]))
            horizontal=np.array(horizontal).T
            if equalarea:
                proj_dirs=equalarea_directions(horizontal)   
            else:
                proj_dirs=stereoprojection_directions(horizontal)   
            ax.plot(proj_dirs[0,:],proj_dirs[1,:],color=(0.5,0.5,0.5),linewidth=0.5,linestyle='--',zorder=gridzorder)
        
     
    normals = np.array([1,-1,0]);
#    normals = np.array([1,1,1]);
    if equalarea:
        proj_111=equalarea_directions(np.array([1,1,1]))
    else:
        proj_111=stereoprojection_directions(np.array([1,1,1]))
    arclength = 55#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
    
    if equalarea:
        proj_normals = equalarea_planes(normals,arclength=arclength,iniangle=35)
    else:
        proj_normals = stereoprojection_planes(normals,arclength=arclength,iniangle=35)
        
    ax.plot(proj_normals[0,:], proj_normals[1,:], 'k')
        
    
    
    normals = np.array([-1,0,1]);
    arclength = 35#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
    if equalarea:
        proj_normals = equalarea_planes(normals,arclength=arclength,iniangle=90)
    else:
        proj_normals = stereoprojection_planes(normals,arclength=arclength,iniangle=90)
    ax.plot(proj_normals[0,:], proj_normals[1,:], 'k')
    
    if equalarea:
        R = equalarea_directions(np.array([1,0,1]))[0]
    else:
        R = stereoprojection_directions(np.array([1,0,1]))[0]
    
    ax.plot([0,R[0]],[0,0], 'k')
#    dirs = np.column_stack([[0,0,1],[1,1,0],[1,0,0]]);

    dirs = np.column_stack([[0,0,1],[1,1,1],[1,0,1]]);
    
    
    if equalarea:
        proj_dirs = equalarea_directions(dirs)
    else:
        proj_dirs = stereoprojection_directions(dirs)
    
    ax.plot(proj_dirs[0,:], proj_dirs[1,:], 'ko',zorder=50)
    if basedirs:
        #ax.plot(proj_dirs[0,:], proj_dirs[1,:], 'ko',zorder=50)
        for diri,proj_diri in zip(np.transpose(dirs),np.transpose(proj_dirs)):
            if (diri==[0,0,1]).all():
                ax.text(-0.1+proj_diri[0],0.01+proj_diri[1],str(diri))
            else:
                ax.text(0.01+proj_diri[0],0.01+proj_diri[1],str(diri))
    
    if grid:
        if resolution is None:
            resolution=5
        if gridmarkersize is None:
            gridmarkersize=5
        if gridmarkercol is None:
            gridmarkercol='k'
        if gridzorder is None:
            gridzorder=50000

        grid_cub = get_beam_directions_grid("cubic", resolution, mesh="spherified_cube_edge")
        grid_stereo = Rotation.from_euler(np.deg2rad(grid_cub))*Vector3d.zvector()
    
        if equalarea:
            proj_Ds = equalarea_directions(grid_stereo.data.T)
        else:
            proj_Ds = stereoprojection_directions(grid_stereo.data.T)
        
        # #Colors=stereotriangle_colors(proj_Ds)
        # #fig,ax=stereotriangle(ax=None,basedirs=basedirs)
        ax.scatter(proj_Ds[0,:],proj_Ds[1,:],c=gridmarkercol,s=gridmarkersize,zorder=gridzorder)#,'.',color='r',markersize=1)
        ax.scatter([0],[0],c=gridmarkercol,s=gridmarkersize,zorder=gridzorder)#,'.',color='r',markersize=1)
    
    dirs= np.column_stack([[1,1,1],[1,0,1]]);
    if equalarea:
        proj_dirs = equalarea_directions(dirs)
    else:
        proj_dirs = stereoprojection_directions(dirs)
    
    dd=0.02
    #ax.set_xlim([0-dd,proj_dirs[0,1]+dd])
    #ax.set_ylim([0-dd,proj_dirs[1,0]+dd])    
    
    return fig,ax

def colored_stereotriangle(basedirs=False,resolution = 1, markersize=1):
#resolution = 1 
    grid_cub = get_beam_directions_grid("cubic", resolution, mesh="spherified_cube_edge")
    grid_stereo = Rotation.from_euler(np.deg2rad(grid_cub))*Vector3d.zvector()

    proj_Ds = stereoprojection_directions(grid_stereo.data.T)
    Colors=stereotriangle_colors(proj_Ds)
    fig,ax=stereotriangle(ax=None,basedirs=basedirs)
    ax.scatter(proj_Ds[0,:],proj_Ds[1,:],c=Colors,s=markersize)#,'.',color='r',markersize=1)
    #plt.show()
    return fig,ax
def filled_colored_stereotriangle(basedirs=False,resolution = 1, markersize=1,ax=None,**kwargs):
#resolution = 1 
    #import matplotlib.tri as tri
    
    grid_cub = get_beam_directions_grid("cubic", resolution, mesh="spherified_cube_edge")
    grid_stereo = Rotation.from_euler(np.deg2rad(grid_cub))*Vector3d.zvector()

    proj_Ds = stereoprojection_directions(grid_stereo.data.T)
    Colors=stereotriangle_colors(proj_Ds)
    cmap=ListedColormap(Colors)
    fig,ax=stereotriangle(ax=ax,basedirs=basedirs)
    #ax.tricontourf(proj_Ds[0,:],proj_Ds[1,:], range(0,len(Z)),vmin=0,vmax=len(Z),cmap=ListedColormap(Colors))
    ax.tripcolor(proj_Ds[0,:],proj_Ds[1,:], list(range(0,Colors.shape[0])),cmap=cmap, shading='gouraud',**kwargs)
    #ax.scatter(proj_Ds[0,:],proj_Ds[1,:],c=Colors,s=markersize)#,'.',color='r',markersize=1)  
    
    
    return fig,ax
def colors4stereotriangle(resolution = 1):
#resolution = 1 
    grid_cub = get_beam_directions_grid("cubic", resolution, mesh="spherified_cube_edge")
    grid_stereo = Rotation.from_euler(np.deg2rad(grid_cub))*Vector3d.zvector()

    proj_Ds = stereoprojection_directions(grid_stereo.data.T)
    Colors=stereotriangle_colors(proj_Ds)
    return proj_Ds,Colors

def stereotriangle_colors(proj_Ds):
    #adapted from https://mathematica.stackexchange.com/questions/47492/how-to-create-an-inverse-pole-figure-color-map
    #Red point
    Rp=stereoprojection_directions(np.array([0,0,1]))
    XR=Rp[0]
    YR=Rp[1]
    #Green point
    Gp=stereoprojection_directions(np.array([1,0,1]))
    XG=Gp[0]
    YG=Gp[1]
    #Blue point
    Bp=stereoprojection_directions(np.array([1,1,1]))
    XB=Bp[0]
    YB=Bp[1]
    
    #Point O to be colored
    try:
        
        if len(proj_Ds.shape)==2:
            XO=proj_Ds[0,:]
            YO=proj_Ds[1,:]
        else:
            XO=proj_Ds[0]
            YO=proj_Ds[1]
    except:
            XO=proj_Ds[0]
            YO=proj_Ds[1]
            
    #XO=XB
    #YO=YB
    
    #Intersection Gx of GO with RB
    K1GO=(XG-XO)*(YR-YB)-(YG-YO)*(XR-XB)
    #O==G
    OeG=np.where(K1GO==0)
    K1GO[OeG]=1
    XGx=((XG*YO-YG*XO)*(XR-XB)-(XG-XO)*(XR*YB-YR*XB))/K1GO
    YGx=((XG*YO-YG*XO)*(YR-YB)-(YG-YO)*(XR*YB-YR*XB))/K1GO
    XGx[OeG]=XR
    YGx[OeG]=YR
    #Intersection Bx of BO with RG
    K1BO=(XB-XO)*(YR-YG)-(YB-YO)*(XR-XG)
    #O==B
    OeB=np.where(K1BO==0)
    K1BO[OeB]=1
    XBx=((XB*YO-YB*XO)*(XR-XG)-(XB-XO)*(XR*YG-YR*XG))/K1BO
    YBx=((XB*YO-YB*XO)*(YR-YG)-(YB-YO)*(XR*YG-YR*XG))/K1BO
    XBx[OeB]=XR
    YBx[OeB]=YR
    
    
    #Intersection Rx of RO with arc [101]-[111]
    #O==R
    OeR=np.where(XO==0)
    XO2=XO.copy()
    XO2[OeR]=1.0
    K1=(YO/XO2)**2
    XRx=(np.sqrt(K1+2)-1)/(K1+1)
    YRx=YO/XO2*XRx
    XRx[OeR]=XB
    YRx[OeR]=YB
    
    #ratios |ORx|//|RRx| |OGx|//|GGx| |OBx|//|BBx|
    RED=np.sqrt((XO-XRx)**2+(YO-YRx)**2)/np.sqrt((XR-XRx)**2+(YR-YRx)**2)
    GREEN=np.sqrt((XO-XGx)**2+(YO-YGx)**2)/np.sqrt((XG-XGx)**2+(YG-YGx)**2)
    BLUE=np.sqrt((XO-XBx)**2+(YO-YBx)**2)/np.sqrt((XB-XBx)**2+(YB-YBx)**2)
    
    
    
    Colors=np.vstack((RED,GREEN,BLUE)).T
    #Colors=Colors/np.max(Colors,axis=1)
    
    Colors=np.vstack((RED/np.max(Colors,axis=1),GREEN/np.max(Colors,axis=1),BLUE/np.max(Colors,axis=1))).T
    
    return Colors


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

def stereoprojection_intotriangle_ini(dirs,eps=1.0e-5):
    normals = np.array([-1,0,1]);
    arclength = 40.#-np.arccos(np.sqrt(2)/np.sqrt(3))*180/np.pi;
    proj_normals = stereoprojection_planes(normals,arclength=arclength,iniangle=90)
    proj_tans = np.arctan(proj_normals[1,:]/proj_normals[0,:])
    
    if len(dirs.shape)==1:
        dirs = np.expand_dims(dirs,axis=1)

    proj_dirs = np.zeros(dirs.shape)
    inc=-1
    for diri in dirs.T:
#        print('===================================================')
#        print(diri)
#        print('===================================================')
        inc+=1
        el=equivalent_elements(diri,'cubic')
        #print(el)
        for eli in el:
            proj_eli = stereoprojection_directions(eli)
#            print(eli)
#            print(np.arctan(proj_eli[1,0]/proj_eli[0,0])-np.arccos(1./np.sqrt(3.)))
#            print(np.arctan(proj_eli[1,0]/proj_eli[0,0]))#-np.pi/4)
#            print((np.arccos(abs(eli[2])/np.sqrt(eli.dot(eli)))-np.arccos(1./np.sqrt(3.))))
#            if (eli>=-eps).all() and (np.arccos(abs(eli[2])/np.sqrt(eli.dot(eli)))-np.arccos(1./np.sqrt(3.)))<eps:
#            if (proj_eli[:,0]>=-eps).all() and (np.arccos(abs(eli[2])/np.sqrt(eli.dot(eli)))-np.arccos(1./np.sqrt(3.)))<eps:
            if ((proj_eli[:,0])>=-eps).all():                #proj_eli = stereoprojection_directions(eli)
                atan=np.arctan2(proj_eli[1,0],proj_eli[0,0])
                if (atan-np.pi/4)<eps:
                    idx=np.where(abs(proj_tans-atan)==min(abs(proj_tans-atan)))[0][0]
#                    print(proj_eli[:,0].dot(proj_eli[:,0])) 
#                    print(proj_normals[:,idx].dot(proj_normals[:,idx])) 
#                    print((proj_eli[:,0].dot(proj_eli[:,0])-proj_normals[:,idx].dot(proj_normals[:,idx])))
#                    print((proj_eli[:,0].dot(proj_eli[:,0])-proj_normals[:,idx].dot(proj_normals[:,idx]))<eps)
                    if (proj_eli[:,0].dot(proj_eli[:,0])-proj_normals[:,idx].dot(proj_normals[:,idx]))<eps:
                        proj_dirs[:,inc]=proj_eli[:,0]
#                        print('OK')
#                        print(proj_dirs[:,inc])
                        #break
    return proj_dirs
def stereoprojection_intotriangle_fast(dirs,eps=1.0e-5,geteqdirs=False,geteqmats=False,Rin=None,symops=None):
    etamax=np.arctan2(1,1)*180./np.pi
    if len(dirs.shape)==1:
        dirs = np.expand_dims(dirs,axis=1)
    if symops is None:
        symops = symmetry_elements('cubic')
    eqmats=np.array([np.eye(3) for ii in range(dirs.shape[1])])
    eqdirs = np.zeros(dirs.shape)
    Rout=np.zeros(eqmats.shape)
    RTout=np.zeros(eqmats.shape)
    
    for sym in symops:
	#could be faster if  if we remove directions found in each itteration
        #sym=np.eye(3)
        datas=sym.dot(dirs)
        idxs = np.where((datas[0,:]>=0 ) & (datas[1,:]>=0) & (datas[2,:]>=0))[0]
        eta=np.arctan2(datas[0,idxs],np.abs(datas[2,idxs]))*180./np.pi
        chi=np.arctan2(datas[1,idxs],datas[0,idxs])*180./np.pi
        idxs=idxs[np.where((eta<=etamax) & (chi<=etamax))[0]]
        eqmats[idxs,:,:]=sym
        eqdirs[:,idxs]=datas[:,idxs]
        #dirs=np.delete(dirs,idxs,1)
        if Rin is not None:
            for idxsi in idxs:
                Rout[idxsi,:,:]=sym.dot(Rin[idxsi])
                RTout[idxsi,:,:]=Rout[idxsi,:,:].T
        
        if dirs.shape[1]==0:
            break    #print('test')
    proj_dirs=stereoprojection_directions(eqdirs)
    out={}
    if Rin is not None:
        out['Rout']=Rout  
        out['RTout']=RTout  
    if geteqdirs:
        out['eqdirs']=eqdirs
    if geteqmats:
        out['eqmats']=eqmats
    if len(out)>0:
        return proj_dirs, out
    else:
        return proj_dirs
def stereoprojection_intotriangle(dirs,eps=1.0e-5,geteqdirs=False,geteqmats=False,Rin=None,symops=None):
    etamax=np.arctan2(1,1)*180./np.pi
    if len(dirs.shape)==1:
        dirs = np.expand_dims(dirs,axis=1)
    if symops is None:
        symops = symmetry_elements('cubic')
    proj_dirs = np.zeros(dirs.shape)
    eqdirs = np.zeros(dirs.shape)
    eqmats=[]
    inc=-1
    Rout=[]
    RTout=[]
    idx=-1
    for diri in dirs.T:
        idx+=1
        inc+=1
        br=False
        for sym in symops:
            Ds = sym.dot(diri)
            
            if Ds[0]>=0 and Ds[1]>=0 and Ds[2]>=0:
                eta=np.arctan2(Ds[0],np.abs(Ds[2]))*180./np.pi
                chi=np.arctan2(Ds[1],Ds[0])*180./np.pi#np.arcsin(Ds[2])*180./np.pi
                if eta<=etamax and chi<=etamax:# and np.abs(chi<=etamax)<=1.0e-5: #and eta<=np.pi/2 and chi<=etamax:# and chi>=chimax:   
                    break
                    br=True
        #if not br:
        #    print("sdddddddddddddddddd")
        proj_dirs[:,inc] = stereoprojection_directions(Ds)[:,0]
        eqmats.append(sym)
        if Rin is not None:
            Rout.append(sym.dot(Rin[idx]))
            #if np.linalg.det(Rout[-1])<0:
                #Rout[-1]=-1*Rout[-1]
            RTout.append(Rout[-1].T)
        eqdirs[:,inc]=Ds
    #print('test')
    out={}
    if Rin is not None:
        out['Rout']=Rout  
        out['RTout']=RTout  
    if geteqdirs:
        out['eqdirs']=eqdirs
    if geteqmats:
        out['eqmats']=eqmats
    if len(out)>0:
        return proj_dirs, out
    else:
        return proj_dirs
def equalarea_intotriangle_fast(dirs,eps=1.0e-5,geteqdirs=False,geteqmats=False,Rin=None,symops=None):
    etamax=np.arctan2(1,1)*180./np.pi
    if len(dirs.shape)==1:
        dirs = np.expand_dims(dirs,axis=1)
    if symops is None:
        symops = symmetry_elements('cubic')
    eqmats=np.array([np.eye(3) for ii in range(dirs.shape[1])])
    eqdirs = np.zeros(dirs.shape)
    Rout=np.zeros(eqmats.shape)
    RTout=np.zeros(eqmats.shape)
    
    for sym in symops:
        #sym=np.eye(3)
        datas=sym.dot(dirs)
        idxs = np.where((datas[0,:]>=0 ) & (datas[1,:]>=0) & (datas[2,:]>=0))[0]
        eta=np.arctan2(datas[0,idxs],np.abs(datas[2,idxs]))*180./np.pi
        chi=np.arctan2(datas[1,idxs],datas[0,idxs])*180./np.pi
        idxs=idxs[np.where((eta<=etamax) & (chi<=etamax))[0]]
        eqmats[idxs,:,:]=sym
        eqdirs[:,idxs]=datas[:,idxs]
        dirs=np.delete(dirs,idxs,1)
        if Rin is not None:
            for idxsi in idxs:
                Rout[idxsi,:,:]=sym.dot(Rin[idxsi])
                Rout[idxsi,:,:]=Rout[idxsi,:,:].T
        
        if dirs.shape[1]==0:
            break    #print('test')
    proj_dirs=equalarea_directions(eqdirs)
    out={}
    if Rin is not None:
        out['Rout']=Rout  
        out['RTout']=RTout  
    if geteqdirs:
        out['eqdirs']=eqdirs
    if geteqmats:
        out['eqmats']=eqmats
    if len(out)>0:
        return proj_dirs, out
    else:
        return proj_dirs

def equalarea_intotriangle(dirs,eps=1.0e-5,geteqdirs=False,geteqmats=False):
    etamax=np.arctan2(1,1)*180./np.pi
    if len(dirs.shape)==1:
        dirs = np.expand_dims(dirs,axis=1)

    symops = symmetry_elements('cubic')
    proj_dirs = np.zeros(dirs.shape)
    eqdirs = np.zeros(dirs.shape)
    eqmats=[]
    inc=-1
    
    for diri in dirs.T:
        inc+=1
        br=False
        for sym in symops:
            Ds = sym.dot(diri)
            
            if Ds[0]>=0 and Ds[1]>=0 and Ds[2]>=0:
                eta=np.arctan2(Ds[0],np.abs(Ds[2]))*180./np.pi
                chi=np.arctan2(Ds[1],Ds[0])*180./np.pi#np.arcsin(Ds[2])*180./np.pi
                if eta<=etamax and chi<=etamax:# and np.abs(chi<=etamax)<=1.0e-5: #and eta<=np.pi/2 and chi<=etamax:# and chi>=chimax:   
                    break
                    br=True
        #if not br:
        #    print("sdddddddddddddddddd")
        proj_dirs[:,inc] = equalarea_directions(Ds)[:,0]
        eqmats.append(sym)
        eqdirs[:,inc]=Ds
#    if geteqdirs and not geteqmats:
#        return proj_dirs,eqdirs
#    elif geteqmats and not geteqdirs:
#        return proj_dirs,eqmats
#    elif eqmats and geteqdirs:
#        return proj_dirs,eqdirs,eqmats
#    else:
#        return proj_dirs


    out={}
    if geteqdirs:
        out['eqdirs']=eqdirs
    if geteqmats:
        out['eqmats']=eqmats
    if len(out)>0:
        return proj_dirs, out
    else:
        return proj_dirs



def stereoprojection_planes(normals,arclength=360.,iniangle=0.,hemisphere='both',getpoints=False):
    #%normals = [x1,x2,...,xn;y1,y2,...,yn;z1,z2,...,zn];
    #%varargin{1} arclength in deg
    #normals = np.transpose(np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]))
    #
    if len(normals.shape)==1:
        normals = np.expand_dims(normals,axis=1)

    normals = normals.astype(float)
    normals /= np.sqrt((normals ** 2).sum(0))

    proj_normals = stereoprojection_directions(normals)

    idxs = np.where(abs(normals[0,:])+abs(normals[1,:])==0)[0]
    
    inplanedirs = np.vstack((-normals[1,:],normals[0,:],np.zeros(normals[0,:].shape)));
    inplanedirs[:,idxs] = np.vstack((np.zeros(normals[0,idxs].shape), -normals[2,idxs],normals[1,idxs]));
    
    inplanedirs /= np.sqrt((inplanedirs ** 2).sum(0))

    thirdaxis=np.cross(normals,inplanedirs,axisa=0,axisb=0,axisc=0)
#    thirdaxis = np.vstack((normals[1,:]*inplanedirs[2,:]-normals[2,:]*inplanedirs[1,:],
#                           -1*(normals[0,:]*inplanedirs[2,:]-normals[2,:]*inplanedirs[0,:]),
#                           normals[0,:]*inplanedirs[1,:]-normals[1,:]*inplanedirs[0,:]));
    t=np.linspace(iniangle,iniangle+arclength,180*2+1)*np.pi/180;
    basicarc = np.vstack((np.cos(t),np.sin(t),np.zeros(t.shape)));
    
    proj_planes=[];
    plane_points=[]
    for i in range(0,normals.shape[1]):
        Rot2Global = np.transpose(np.vstack((inplanedirs[:,i],thirdaxis[:,i],normals[:,i])));
        Ccp = np.matmul(Rot2Global,basicarc);
        
        if hemisphere == "both":            
            Ds = stereoprojection_directions(Ccp)
        elif hemisphere == "triangle":  
            #idxs = np.where(Ccp[2,:]>=0)[0]
            Ds=stereoprojection_intotriangle(Ccp)#[:,idxs])
        else:
            if hemisphere == "upper":
                idxs = np.where(Ccp[2,:]>=0)[0]
            elif hemisphere == "lower":
                idxs = np.where(Ccp[2,:]<=0)[0]
            Ds = stereoprojection_directions(Ccp[:,idxs])

        proj_planes.append(Ds)
        plane_points.append(Ccp[:,idxs])
        
    if len(proj_planes)==1:
        proj_planes = proj_planes[0]
        plane_points=plane_points[0]
    
    if getpoints:
        return  proj_planes, plane_points   
    else:
        return proj_planes

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
            #vm/=min(abs(vm[np.nonzero(vm)[0]]))
        else:
            #print(vm)
            vm=np.round(vm/abs(max(vm[np.abs(vm)>1/tol]))*tol)/tol
            #vm/=max(abs(vm[np.nonzero(vm)[0]]))
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
 
