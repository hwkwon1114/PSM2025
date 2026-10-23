#!/usr/bin/env python3
"""Exploratory full-sheet shape/curvature PNG; no subsampling or z exaggeration."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('fields', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
with np.load(args.fields) as data:
    uv, faces, pos, k = [data[key] for key in ('uv','faces','positions','geometric')]
# Rigid least-squares alignment to material plane for presentation only.
reference = np.column_stack((uv, np.zeros(len(uv))))
a, b = pos-pos.mean(axis=0), reference-reference.mean(axis=0)
u, _, vt = np.linalg.svd(a.T @ b)
correction = np.eye(3)
correction[-1,-1] = np.linalg.det(u @ vt)
shown = (a @ u @ correction @ vt)*1000
triangulation = tri.Triangulation(uv[:,0]*1000,uv[:,1]*1000,faces)
weak = np.take_along_axis(k,np.argmin(abs(k),axis=1)[:,None],axis=1).ravel()
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
    fig = plt.figure(figsize=(11,4.3),layout='constrained')
    ax = fig.add_subplot(131,projection='3d')
    ax.plot_trisurf(shown[:,0],shown[:,1],shown[:,2],triangles=faces,color='#0072B2',linewidth=0,antialiased=False)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',title='(a) Shape, equal scale')
    ax.set_box_aspect((200,300,60), zoom=.72)
    ax.tick_params(labelsize=8, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_size(9)
        axis.labelpad = 0
    ax.set_xlim(-100,100); ax.set_ylim(-150,150); ax.set_zlim(-30,30)
    ax.set_xticks([-100,0,100]); ax.set_yticks([-150,0,150]); ax.set_zticks([-30,0,30])
    ax.view_init(elev=25,azim=-55)
    for index, (field,title,label) in enumerate([(k[:,0]*k[:,1],'(b) Gaussian curvature','K (m$^{-2}$)'),
                                               (weak,'(c) Weaker curvature','k (m$^{-1}$)')],2):
        axis=fig.add_subplot(1,3,index)
        limit=np.nanmax(abs(field))
        cmap=plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#888888')
        artist=axis.tripcolor(triangulation,facecolors=np.ma.masked_invalid(field),cmap=cmap,vmin=-limit,vmax=limit,rasterized=True)
        axis.add_patch(Rectangle((-80,-120),160,240,fill=False,edgecolor='black',linestyle='--',linewidth=.8))
        axis.set(xlabel='u (mm)',ylabel='v (mm)',title=title,aspect='equal')
        fig.colorbar(artist,ax=axis,label=label,shrink=.8)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=170)
    plt.close(fig)
args.output.with_suffix('.caption.md').write_text(
    'Baseline full-sheet diagnostic. Source: '+str(args.fields)+'. One deterministic free-boundary final-only equilibrium; mesh faces are not independent replicates. '
    '(a) Full vertex surface rigidly aligned to the material reference plane, in millimetres; equal physical axis scale, no vertical exaggeration. '
    '(b,c) Geometry-derived signed curvature from fixed 15 mm neighborhoods, shown on material coordinates; full finite ranges, gray denotes invalid values. '
    'Dashed box is the fixed central evaluation region. No uncertainty intervals or face subsampling. These exploratory views do not establish mesh convergence or dome success.\n')
