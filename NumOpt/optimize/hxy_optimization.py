import numpy as np
from scipy.spatial import Delaunay as DelaunayND

from ..cprint import cprint_green,cprint_magenta

class Delaunay1D(object):
    def __init__(self,points):
        self.points=np.reshape(points,(-1,1))
        self.dim=len(self.points)
    @property
    def vertex_neighbor_vertices(self):
        sord_idx=np.argsort(self.points.flatten())
        idxptr=[0]
        idx=[]
        sort_points=np.arange(self.dim)[sord_idx]
        ranks=np.argsort(sord_idx)
        for _,vi in enumerate(ranks):
            len_idx=len(idx)
            if vi==0:
                right=vi+1
                right=sort_points[right]
                idxptr.append(len_idx+1)
                idx.append(right)
            elif vi==self.dim-1:
                left=vi-1 
                left=sort_points[left]
                idxptr.append(len_idx+1)
                idx.append(left)
            else:
                left=vi-1 
                left=sort_points[left]
                right=vi+1
                right=sort_points[right]
                idxptr.append(len_idx+2)
                idx.extend([left,right])      

        idxptr=np.array(idxptr)
        idx=np.array(idx)
        return idxptr,idx