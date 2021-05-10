from sys import intern
from matplotlib import cm
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.lib.function_base import append
from numpy.lib.type_check import nan_to_num
import pywt
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.decomposition import PCA
def crop_center(img,cropx,cropy):
    y,x, z=img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]


def shift(mat, value, axis):
    result=np.ndarray(mat.shape)
    if axis==1:
        if value==1:
            result[:value, :]=1
            result[value:, :]=mat[:-value, :]
        else:
            result[:value, :]=mat[:-value, :]
            result[value:, :]=1
    else:
        if value==1:
            result[:, :value]=1
            result[:, value:]=mat[:, :-value]
        else:
            result[:, :value]=mat[:, :-value]
            result[:, value:]=1
    return np.abs(result.flatten())

def scale(mat, shape):
    result1=np.repeat(mat, 2, 1)
    result=np.repeat(result1, 2, 0)
    return np.abs(result[:shape[0], :shape[1]].flatten())


X=None
Y=[]

for im in os.listdir('./PI_train'):
    photo=crop_center(np.array(Image.open("./PI_train/"+im)), 256, 256)
    red=photo[:, :, 0]
    green=photo[:, :, 1]
    blue=photo[:, :, 2]
    (A1r, (H1r, V1r, D1r))=pywt.dwt2(red, "haar")
    (A2r, (H2r, V2r, D2r))=pywt.dwt2(A1r, "haar")
    (A3r, (H3r, V3r, D3r))=pywt.dwt2(A2r, "haar")
    (A4r, (H4r, V4r, D4r))=pywt.dwt2(A3r, "haar")
    (A1g, (H1g, V1g, D1g))=pywt.dwt2(green, "haar")
    (A2g, (H2g, V2g, D2g))=pywt.dwt2(A1g, "haar")
    (A3g, (H3g, V3g, D3g))=pywt.dwt2(A2g, "haar")
    (A4g, (H4g, V4g, D4g))=pywt.dwt2(A3g, "haar")
    (A1b, (H1b, V1b, D1b))=pywt.dwt2(blue, "haar")
    (A2b, (H2b, V2b, D2b))=pywt.dwt2(A1b, "haar")
    (A3b, (H3b, V3b, D3b))=pywt.dwt2(A2b, "haar")
    (A4b, (H4b, V4b, D4b))=pywt.dwt2(A3b, "haar")
    Vg1=V1g.flatten()
    Vb1=V1b.flatten()
    Vr1=V1r.flatten()
    Hg1=H1g.flatten()
    Hb1=H1b.flatten()
    Hr1=H1r.flatten()
    Dg1=D1g.flatten()
    Db1=D1b.flatten()
    Dr1=D1r.flatten()

    Vg2=V2g.flatten()
    Vb2=V2b.flatten()
    Vr2=V2r.flatten()
    Hg2=H2g.flatten()
    Hb2=H2b.flatten()
    Hr2=H2r.flatten()
    Dg2=D2g.flatten()
    Db2=D2b.flatten()
    Dr2=D2r.flatten()

    Vg3=V3g.flatten()
    Vb3=V3b.flatten()
    Vr3=V3r.flatten()
    Hg3=H3g.flatten()
    Hb3=H3b.flatten()
    Hr3=H3r.flatten()
    Dg3=D3g.flatten()
    Db3=D3b.flatten()
    Dr3=D3r.flatten()

    Vg4=V4g.flatten()
    Vb4=V4b.flatten()
    Vr4=V4r.flatten()
    Hg4=H4g.flatten()
    Hb4=H4b.flatten()
    Hr4=H4r.flatten()
    Dg4=D4g.flatten()
    Db4=D4b.flatten()
    Dr4=D4r.flatten()
    Qvg1=np.column_stack((shift(V1g, -1, 0), shift(V1g, 1, 0), shift(V1g, -1, 1), shift(V1g, 1, 1), scale(V2g, V1g.shape), np.abs(Dg1), scale(D2g, V1g.shape), np.abs(Vr1), np.abs(Vb1)))
    Qvb1=np.column_stack((shift(V1b, -1, 0), shift(V1b, 1, 0), shift(V1b, -1, 1), shift(V1b, 1, 1), scale(V2b, V1g.shape), np.abs(Db1), scale(D2b, V1g.shape), np.abs(Vb1), np.abs(Vg1)))
    Qvr1=np.column_stack((shift(V1r, -1, 0), shift(V1r, 1, 0), shift(V1r, -1, 1), shift(V1r, 1, 1), scale(V2r, V1g.shape), np.abs(Dr1), scale(D2r, V1g.shape), np.abs(Vg1), np.abs(Vr1)))
    Qhg1=np.column_stack((shift(H1g, -1, 0), shift(H1g, 1, 0), shift(H1g, -1, 1), shift(H1g, 1, 1), scale(H2g, V1g.shape), np.abs(Dg1), scale(D2g, V1g.shape), np.abs(Hr1), np.abs(Hb1)))
    Qhb1=np.column_stack((shift(H1b, -1, 0), shift(H1b, 1, 0), shift(H1b, -1, 1), shift(H1b, 1, 1), scale(H2b, V1g.shape), np.abs(Db1), scale(D2b, V1g.shape), np.abs(Hb1), np.abs(Hg1)))
    Qhr1=np.column_stack((shift(H1r, -1, 0), shift(H1r, 1, 0), shift(H1r, -1, 1), shift(H1r, 1, 1), scale(H2r, V1g.shape), np.abs(Dr1), scale(D2r, V1g.shape), np.abs(Hg1), np.abs(Hr1)))
    Qdg1=np.column_stack((shift(D1g, -1, 0), shift(D1g, 1, 0), shift(D1g, -1, 1), shift(D1g, 1, 1), scale(D2g, V1g.shape), np.abs(Hg1), np.abs(Vg1), np.abs(Dr1), np.abs(Db1)))
    Qdb1=np.column_stack((shift(D1b, -1, 0), shift(D1b, 1, 0), shift(D1b, -1, 1), shift(D1b, 1, 1), scale(D2b, V1g.shape), np.abs(Hb1), np.abs(Vb1), np.abs(Db1), np.abs(Dg1)))
    Qdr1=np.column_stack((shift(D1r, -1, 0), shift(D1r, 1, 0), shift(D1r, -1, 1), shift(D1r, 1, 1), scale(D2r, V1g.shape), np.abs(Hr1), np.abs(Vr1), np.abs(Dg1), np.abs(Dr1)))
    
    wvg1=np.matmul(np.linalg.pinv(Qvg1), Vg1).squeeze()
    wvb1=np.matmul(np.linalg.pinv(Qvb1), Vb1).squeeze()
    wvr1=np.matmul(np.linalg.pinv(Qvr1), Vr1).squeeze()
    whg1=np.matmul(np.linalg.pinv(Qhg1), Hg1).squeeze()
    whb1=np.matmul(np.linalg.pinv(Qhb1), Hb1).squeeze()
    whr1=np.matmul(np.linalg.pinv(Qhr1), Hr1).squeeze()
    wdg1=np.matmul(np.linalg.pinv(Qdg1), Dg1).squeeze()
    wdb1=np.matmul(np.linalg.pinv(Qdb1), Db1).squeeze()
    wdr1=np.matmul(np.linalg.pinv(Qdr1), Dr1).squeeze()
    
    xvr1=np.abs(np.matmul(Qvr1, wvr1))
    xvb1=np.abs(np.matmul(Qvb1, wvb1))
    xvg1=np.abs(np.matmul(Qvg1, wvg1))
    xhr1=np.abs(np.matmul(Qhr1, whr1))
    xhb1=np.abs(np.matmul(Qhr1, whr1))
    xhg1=np.abs(np.matmul(Qhr1, whr1))
    xdr1=np.abs(np.matmul(Qdr1, wdr1))
    xdb1=np.abs(np.matmul(Qdr1, wdr1))
    xdg1=np.abs(np.matmul(Qdr1, wdr1))
    xvr1[xvr1<=1e-10]=1e-10
    xvb1[xvb1<=1e-10]=1e-10
    xvg1[xvg1<=1e-10]=1e-10
    xhr1[xhr1<=1e-10]=1e-10
    xhb1[xhb1<=1e-10]=1e-10
    xhg1[xhg1<=1e-10]=1e-10
    xdr1[xdr1<=1e-10]=1e-10
    xdb1[xdb1<=1e-10]=1e-10
    xdg1[xdg1<=1e-10]=1e-10
    Vr1[Vr1<=1e-10]=1e-10
    Vb1[Vb1<=1e-10]=1e-10
    Vg1[Vg1<=1e-10]=1e-10
    Hr1[Hr1<=1e-10]=1e-10
    Hb1[Hb1<=1e-10]=1e-10
    Hg1[Hg1<=1e-10]=1e-10
    Dr1[Dr1<=1e-10]=1e-10
    Db1[Db1<=1e-10]=1e-10
    Dg1[Dg1<=1e-10]=1e-10
    pvr1=np.nan_to_num(np.log(Vr1/xvr1), nan=0, posinf=0, neginf=0)
    pvg1=np.nan_to_num(np.log(Vg1/xvg1), nan=0, posinf=0, neginf=0)
    pvb1=np.nan_to_num(np.log(Vb1/xvb1), nan=0, posinf=0, neginf=0)
    phr1=np.nan_to_num(np.log(Hr1/xhr1), nan=0, posinf=0, neginf=0)
    phg1=np.nan_to_num(np.log(Hg1/xhg1), nan=0, posinf=0, neginf=0)
    phb1=np.nan_to_num(np.log(Hb1/xhb1), nan=0, posinf=0, neginf=0)
    pdr1=np.nan_to_num(np.log(Dr1/xdr1), nan=0, posinf=0, neginf=0)
    pdg1=np.nan_to_num(np.log(Dg1/xdg1), nan=0, posinf=0, neginf=0)
    pdb1=np.nan_to_num(np.log(Db1/xdb1), nan=0, posinf=0, neginf=0)
######################################################################################################################
    Qvg2=np.column_stack((shift(V2g, -1, 0), shift(V2g, 1, 0), shift(V2g, -1, 1), shift(V2g, 1, 1), scale(V3g, V2g.shape), np.abs(Dg2), scale(D3g, V2g.shape), np.abs(Vr2), np.abs(Vb2)))
    Qvb2=np.column_stack((shift(V2b, -1, 0), shift(V2b, 1, 0), shift(V2b, -1, 1), shift(V2b, 1, 1), scale(V3b, V2g.shape), np.abs(Db2), scale(D3b, V2g.shape), np.abs(Vb2), np.abs(Vg2)))
    Qvr2=np.column_stack((shift(V2r, -1, 0), shift(V2r, 1, 0), shift(V2r, -1, 1), shift(V2r, 1, 1), scale(V3r, V2g.shape), np.abs(Dr2), scale(D3r, V2g.shape), np.abs(Vg2), np.abs(Vr2)))
    Qhg2=np.column_stack((shift(H2g, -1, 0), shift(H2g, 1, 0), shift(H2g, -1, 1), shift(H2g, 1, 1), scale(H3g, V2g.shape), np.abs(Dg2), scale(D3g, V2g.shape), np.abs(Hr2), np.abs(Hb2)))
    Qhb2=np.column_stack((shift(H2b, -1, 0), shift(H2b, 1, 0), shift(H2b, -1, 1), shift(H2b, 1, 1), scale(H3b, V2g.shape), np.abs(Db2), scale(D3b, V2g.shape), np.abs(Hb2), np.abs(Hg2)))
    Qhr2=np.column_stack((shift(H2r, -1, 0), shift(H2r, 1, 0), shift(H2r, -1, 1), shift(H2r, 1, 1), scale(H3r, V2g.shape), np.abs(Dr2), scale(D3r, V2g.shape), np.abs(Hg2), np.abs(Hr2)))
    Qdg2=np.column_stack((shift(D2g, -1, 0), shift(D2g, 1, 0), shift(D2g, -1, 1), shift(D2g, 1, 1), scale(D3g, V2g.shape), np.abs(Hg2), np.abs(Vg2), np.abs(Dr2), np.abs(Db2)))
    Qdb2=np.column_stack((shift(D2b, -1, 0), shift(D2b, 1, 0), shift(D2b, -1, 1), shift(D2b, 1, 1), scale(D3b, V2g.shape), np.abs(Hb2), np.abs(Vb2), np.abs(Db2), np.abs(Dg2)))
    Qdr2=np.column_stack((shift(D2r, -1, 0), shift(D2r, 1, 0), shift(D2r, -1, 1), shift(D2r, 1, 1), scale(D3r, V2g.shape), np.abs(Hr2), np.abs(Vr2), np.abs(Dg2), np.abs(Dr2)))
    
    wvg2=np.matmul(np.linalg.pinv(Qvg2), Vg2).squeeze()
    wvb2=np.matmul(np.linalg.pinv(Qvb2), Vb2).squeeze()
    wvr2=np.matmul(np.linalg.pinv(Qvr2), Vr2).squeeze()
    whg2=np.matmul(np.linalg.pinv(Qhg2), Hg2).squeeze()
    whb2=np.matmul(np.linalg.pinv(Qhb2), Hb2).squeeze()
    whr2=np.matmul(np.linalg.pinv(Qhr2), Hr2).squeeze()
    wdg2=np.matmul(np.linalg.pinv(Qdg2), Dg2).squeeze()
    wdb2=np.matmul(np.linalg.pinv(Qdb2), Db2).squeeze()
    wdr2=np.matmul(np.linalg.pinv(Qdr2), Dr2).squeeze()
    
    xvr2=np.abs(np.matmul(Qvr2, wvr2))
    xvb2=np.abs(np.matmul(Qvb2, wvb2))
    xvg2=np.abs(np.matmul(Qvg2, wvg2))
    xhr2=np.abs(np.matmul(Qhr2, whr2))
    xhb2=np.abs(np.matmul(Qhr2, whr2))
    xhg2=np.abs(np.matmul(Qhr2, whr2))
    xdr2=np.abs(np.matmul(Qdr2, wdr2))
    xdb2=np.abs(np.matmul(Qdr2, wdr2))
    xdg2=np.abs(np.matmul(Qdr2, wdr2))
    xvr2[xvr2<=1e-10]=1e-10
    xvb2[xvb2<=1e-10]=1e-10
    xvg2[xvg2<=1e-10]=1e-10
    xhr2[xhr2<=1e-10]=1e-10
    xhb2[xhb2<=1e-10]=1e-10
    xhg2[xhg2<=1e-10]=1e-10
    xdr2[xdr2<=1e-10]=1e-10
    xdb2[xdb2<=1e-10]=1e-10
    xdg2[xdg2<=1e-10]=1e-10
    Vr3[Vr3<=1e-10]=1e-10
    Vb3[Vb3<=1e-10]=1e-10
    Vg3[Vg3<=1e-10]=1e-10
    Hr3[Hr3<=1e-10]=1e-10
    Hb3[Hb3<=1e-10]=1e-10
    Hg3[Hg3<=1e-10]=1e-10
    Dr3[Dr3<=1e-10]=1e-10
    Db3[Db3<=1e-10]=1e-10
    Dg3[Dg3<=1e-10]=1e-10
    pvr2=np.nan_to_num(np.log(Vr2/xvr2), nan=0, posinf=0, neginf=0)
    pvg2=np.nan_to_num(np.log(Vg2/xvg2), nan=0, posinf=0, neginf=0)
    pvb2=np.nan_to_num(np.log(Vb2/xvb2), nan=0, posinf=0, neginf=0)
    phr2=np.nan_to_num(np.log(Hr2/xhr2), nan=0, posinf=0, neginf=0)
    phg2=np.nan_to_num(np.log(Hg2/xhg2), nan=0, posinf=0, neginf=0)
    phb2=np.nan_to_num(np.log(Hb2/xhb2), nan=0, posinf=0, neginf=0)
    pdr2=np.nan_to_num(np.log(Dr2/xdr2), nan=0, posinf=0, neginf=0)
    pdg2=np.nan_to_num(np.log(Dg2/xdg2), nan=0, posinf=0, neginf=0)
    pdb2=np.nan_to_num(np.log(Db2/xdb2), nan=0, posinf=0, neginf=0)

###################################################################################################################

    Qvg3=np.column_stack((shift(V3g, -1, 0), shift(V3g, 1, 0), shift(V3g, -1, 1), shift(V3g, 1, 1), scale(V4g, V3g.shape), np.abs(Dg3), scale(D4g, V3g.shape), np.abs(Vr3), np.abs(Vb3)))
    Qvb3=np.column_stack((shift(V3b, -1, 0), shift(V3b, 1, 0), shift(V3b, -1, 1), shift(V3b, 1, 1), scale(V4b, V3g.shape), np.abs(Db3), scale(D4b, V3g.shape), np.abs(Vb3), np.abs(Vg3)))
    Qvr3=np.column_stack((shift(V3r, -1, 0), shift(V3r, 1, 0), shift(V3r, -1, 1), shift(V3r, 1, 1), scale(V4r, V3g.shape), np.abs(Dr3), scale(D4r, V3g.shape), np.abs(Vg3), np.abs(Vr3)))
    Qhg3=np.column_stack((shift(H3g, -1, 0), shift(H3g, 1, 0), shift(H3g, -1, 1), shift(H3g, 1, 1), scale(H4g, V3g.shape), np.abs(Dg3), scale(D4g, V3g.shape), np.abs(Hr3), np.abs(Hb3)))
    Qhb3=np.column_stack((shift(H3b, -1, 0), shift(H3b, 1, 0), shift(H3b, -1, 1), shift(H3b, 1, 1), scale(H4b, V3g.shape), np.abs(Db3), scale(D4b, V3g.shape), np.abs(Hb3), np.abs(Hg3)))
    Qhr3=np.column_stack((shift(H3r, -1, 0), shift(H3r, 1, 0), shift(H3r, -1, 1), shift(H3r, 1, 1), scale(H4r, V3g.shape), np.abs(Dr3), scale(D4r, V3g.shape), np.abs(Hg3), np.abs(Hr3)))
    Qdg3=np.column_stack((shift(D3g, -1, 0), shift(D3g, 1, 0), shift(D3g, -1, 1), shift(D3g, 1, 1), scale(D4g, V3g.shape), np.abs(Hg3), np.abs(Vg3), np.abs(Dr3), np.abs(Db3)))
    Qdb3=np.column_stack((shift(D3b, -1, 0), shift(D3b, 1, 0), shift(D3b, -1, 1), shift(D3b, 1, 1), scale(D4b, V3g.shape), np.abs(Hb3), np.abs(Vb3), np.abs(Db3), np.abs(Dg3)))
    Qdr3=np.column_stack((shift(D3r, -1, 0), shift(D3r, 1, 0), shift(D3r, -1, 1), shift(D3r, 1, 1), scale(D4r, V3g.shape), np.abs(Hr3), np.abs(Vr3), np.abs(Dg3), np.abs(Dr3)))
    
    wvg3=np.matmul(np.linalg.pinv(Qvg3), Vg3).squeeze()
    wvb3=np.matmul(np.linalg.pinv(Qvb3), Vb3).squeeze()
    wvr3=np.matmul(np.linalg.pinv(Qvr3), Vr3).squeeze()
    whg3=np.matmul(np.linalg.pinv(Qhg3), Hg3).squeeze()
    whb3=np.matmul(np.linalg.pinv(Qhb3), Hb3).squeeze()
    whr3=np.matmul(np.linalg.pinv(Qhr3), Hr3).squeeze()
    wdg3=np.matmul(np.linalg.pinv(Qdg3), Dg3).squeeze()
    wdb3=np.matmul(np.linalg.pinv(Qdb3), Db3).squeeze()
    wdr3=np.matmul(np.linalg.pinv(Qdr3), Dr3).squeeze()
    
    xvr3=np.abs(np.matmul(Qvr3, wvr3))
    xvb3=np.abs(np.matmul(Qvb3, wvb3))
    xvg3=np.abs(np.matmul(Qvg3, wvg3))
    xhr3=np.abs(np.matmul(Qhr3, whr3))
    xhb3=np.abs(np.matmul(Qhr3, whr3))
    xhg3=np.abs(np.matmul(Qhr3, whr3))
    xdr3=np.abs(np.matmul(Qdr3, wdr3))
    xdb3=np.abs(np.matmul(Qdr3, wdr3))
    xdg3=np.abs(np.matmul(Qdr3, wdr3))
    xvr3[xvr3<=1e-10]=1e-10
    xvb3[xvb3<=1e-10]=1e-10
    xvg3[xvg3<=1e-10]=1e-10
    xhr3[xhr3<=1e-10]=1e-10
    xhb3[xhb3<=1e-10]=1e-10
    xhg3[xhg3<=1e-10]=1e-10
    xdr3[xdr3<=1e-10]=1e-10
    xdb3[xdb3<=1e-10]=1e-10
    xdg3[xdg3<=1e-10]=1e-10
    Vr3[Vr3<=1e-10]=1e-10
    Vb3[Vb3<=1e-10]=1e-10
    Vg3[Vg3<=1e-10]=1e-10
    Hr3[Hr3<=1e-10]=1e-10
    Hb3[Hb3<=1e-10]=1e-10
    Hg3[Hg3<=1e-10]=1e-10
    Dr3[Dr3<=1e-10]=1e-10
    Db3[Db3<=1e-10]=1e-10
    Dg3[Dg3<=1e-10]=1e-10
    pvr3=np.nan_to_num(np.log(Vr3/xvr3), nan=0, posinf=0, neginf=0)
    pvg3=np.nan_to_num(np.log(Vg3/xvg3), nan=0, posinf=0, neginf=0)
    pvb3=np.nan_to_num(np.log(Vb3/xvb3), nan=0, posinf=0, neginf=0)
    phr3=np.nan_to_num(np.log(Hr3/xhr3), nan=0, posinf=0, neginf=0)
    phg3=np.nan_to_num(np.log(Hg3/xhg3), nan=0, posinf=0, neginf=0)
    phb3=np.nan_to_num(np.log(Hb3/xhb3), nan=0, posinf=0, neginf=0)
    pdr3=np.nan_to_num(np.log(Dr3/xdr3), nan=0, posinf=0, neginf=0)
    pdg3=np.nan_to_num(np.log(Dg3/xdg3), nan=0, posinf=0, neginf=0)
    pdb3=np.nan_to_num(np.log(Db3/xdb3), nan=0, posinf=0, neginf=0)

    stat=[np.mean(Vr1), np.mean(Vb1), np.mean(Vb1), np.mean(Hr1), np.mean(Hg1), np.mean(Hb1), np.mean(Dr1), np.mean(Dg1), np.mean(Db1), 
    np.var(Vr1), np.var(Vb1), np.var(Vb1), np.var(Hr1), np.var(Hg1), np.var(Hb1), np.var(Dr1), np.var(Dg1), np.var(Db1), 
    stats.skew(Vr1), stats.skew(Vb1), stats.skew(Vb1), stats.skew(Hr1), stats.skew(Hg1), stats.skew(Hb1), stats.skew(Dr1), stats.skew(Dg1), stats.skew(Db1), 
    stats.kurtosis(Vr1), stats.kurtosis(Vb1), stats.kurtosis(Vb1), stats.kurtosis(Hr1), stats.kurtosis(Hg1), stats.kurtosis(Hb1), stats.kurtosis(Dr1), stats.kurtosis(Dg1), stats.kurtosis(Db1), 
    np.mean(xvr1), np.mean(xvg1), np.mean(xvb1), np.mean(xhr1), np.mean(xhg1), np.mean(xhb1), np.mean(xdr1), np.mean(xdg1), np.mean(xdb1),
    np.var(xvr1), np.var(xvg1), np.var(xvb1), np.var(xhr1), np.var(xhg1), np.var(xhb1), np.var(xdr1), np.var(xdg1), np.var(xdb1),
    stats.skew(xvr1), stats.skew(xvg1), stats.skew(xvb1), stats.skew(xhr1), stats.skew(xhg1), stats.skew(xhb1), stats.skew(xdr1), stats.skew(xdg1), stats.skew(xdb1),
    stats.kurtosis(xvr1), stats.kurtosis(xvg1), stats.kurtosis(xvb1), stats.kurtosis(xhr1), stats.kurtosis(xhg1), stats.kurtosis(xhb1), stats.kurtosis(xdr1), stats.kurtosis(xdg1), stats.kurtosis(xdb1),
    
    np.mean(Vr2), np.mean(Vb2), np.mean(Vb2), np.mean(Hr2), np.mean(Hg2), np.mean(Hb2), np.mean(Dr2), np.mean(Dg2), np.mean(Db2), 
    np.var(Vr2), np.var(Vb2), np.var(Vb2), np.var(Hr2), np.var(Hg2), np.var(Hb2), np.var(Dr2), np.var(Dg2), np.var(Db2), 
    stats.skew(Vr2), stats.skew(Vb2), stats.skew(Vb2), stats.skew(Hr2), stats.skew(Hg2), stats.skew(Hb2), stats.skew(Dr2), stats.skew(Dg2), stats.skew(Db2), 
    stats.kurtosis(Vr2), stats.kurtosis(Vb2), stats.kurtosis(Vb2), stats.kurtosis(Hr2), stats.kurtosis(Hg2), stats.kurtosis(Hb2), stats.kurtosis(Dr2), stats.kurtosis(Dg2), stats.kurtosis(Db2), 
    np.mean(xvr2), np.mean(xvg2), np.mean(xvb2), np.mean(xhr2), np.mean(xhg2), np.mean(xhb2), np.mean(xdr2), np.mean(xdg2), np.mean(xdb2),
    np.var(xvr2), np.var(xvg2), np.var(xvb2), np.var(xhr2), np.var(xhg2), np.var(xhb2), np.var(xdr2), np.var(xdg2), np.var(xdb2),
    stats.skew(xvr2), stats.skew(xvg2), stats.skew(xvb2), stats.skew(xhr2), stats.skew(xhg2), stats.skew(xhb2), stats.skew(xdr2), stats.skew(xdg2), stats.skew(xdb2),
    stats.kurtosis(xvr2), stats.kurtosis(xvg2), stats.kurtosis(xvb2), stats.kurtosis(xhr2), stats.kurtosis(xhg2), stats.kurtosis(xhb2), stats.kurtosis(xdr2), stats.kurtosis(xdg2), stats.kurtosis(xdb2),
    
    np.mean(Vr3), np.mean(Vb3), np.mean(Vb3), np.mean(Hr3), np.mean(Hg3), np.mean(Hb3), np.mean(Dr3), np.mean(Dg3), np.mean(Db3), 
    np.var(Vr3), np.var(Vb3), np.var(Vb3), np.var(Hr3), np.var(Hg3), np.var(Hb3), np.var(Dr3), np.var(Dg3), np.var(Db3), 
    stats.skew(Vr3), stats.skew(Vb3), stats.skew(Vb3), stats.skew(Hr3), stats.skew(Hg3), stats.skew(Hb3), stats.skew(Dr3), stats.skew(Dg3), stats.skew(Db3), 
    stats.kurtosis(Vr3), stats.kurtosis(Vb3), stats.kurtosis(Vb3), stats.kurtosis(Hr3), stats.kurtosis(Hg3), stats.kurtosis(Hb3), stats.kurtosis(Dr3), stats.kurtosis(Dg3), stats.kurtosis(Db3), 
    np.mean(xvr3), np.mean(xvg3), np.mean(xvb3), np.mean(xhr3), np.mean(xhg3), np.mean(xhb3), np.mean(xdr3), np.mean(xdg3), np.mean(xdb3),
    np.var(xvr3), np.var(xvg3), np.var(xvb3), np.var(xhr3), np.var(xhg3), np.var(xhb3), np.var(xdr3), np.var(xdg3), np.var(xdb3),
    stats.skew(xvr3), stats.skew(xvg3), stats.skew(xvb3), stats.skew(xhr3), stats.skew(xhg3), stats.skew(xhb3), stats.skew(xdr3), stats.skew(xdg3), stats.skew(xdb3),
    stats.kurtosis(xvr3), stats.kurtosis(xvg3), stats.kurtosis(xvb3), stats.kurtosis(xhr3), stats.kurtosis(xhg3), stats.kurtosis(xhb3), stats.kurtosis(xdr3), stats.kurtosis(xdg3), stats.kurtosis(xdb3),
    ]
    if X is None:
        X=stat
    else:
        X=np.vstack((X, stat))
    Y.append(0)


for im in os.listdir('./CG_train'):
    photo=crop_center(np.array(Image.open("./CG_train/"+im)), 256, 256)
    red=photo[:, :, 0]
    green=photo[:, :, 1]
    blue=photo[:, :, 2]
    (A1r, (H1r, V1r, D1r))=pywt.dwt2(red, "haar")
    (A2r, (H2r, V2r, D2r))=pywt.dwt2(A1r, "haar")
    (A3r, (H3r, V3r, D3r))=pywt.dwt2(A2r, "haar")
    (A4r, (H4r, V4r, D4r))=pywt.dwt2(A3r, "haar")
    (A1g, (H1g, V1g, D1g))=pywt.dwt2(green, "haar")
    (A2g, (H2g, V2g, D2g))=pywt.dwt2(A1g, "haar")
    (A3g, (H3g, V3g, D3g))=pywt.dwt2(A2g, "haar")
    (A4g, (H4g, V4g, D4g))=pywt.dwt2(A3g, "haar")
    (A1b, (H1b, V1b, D1b))=pywt.dwt2(blue, "haar")
    (A2b, (H2b, V2b, D2b))=pywt.dwt2(A1b, "haar")
    (A3b, (H3b, V3b, D3b))=pywt.dwt2(A2b, "haar")
    (A4b, (H4b, V4b, D4b))=pywt.dwt2(A3b, "haar")
    Vg1=V1g.flatten()
    Vb1=V1b.flatten()
    Vr1=V1r.flatten()
    Hg1=H1g.flatten()
    Hb1=H1b.flatten()
    Hr1=H1r.flatten()
    Dg1=D1g.flatten()
    Db1=D1b.flatten()
    Dr1=D1r.flatten()

    Vg2=V2g.flatten()
    Vb2=V2b.flatten()
    Vr2=V2r.flatten()
    Hg2=H2g.flatten()
    Hb2=H2b.flatten()
    Hr2=H2r.flatten()
    Dg2=D2g.flatten()
    Db2=D2b.flatten()
    Dr2=D2r.flatten()

    Vg3=V3g.flatten()
    Vb3=V3b.flatten()
    Vr3=V3r.flatten()
    Hg3=H3g.flatten()
    Hb3=H3b.flatten()
    Hr3=H3r.flatten()
    Dg3=D3g.flatten()
    Db3=D3b.flatten()
    Dr3=D3r.flatten()

    Vg4=V4g.flatten()
    Vb4=V4b.flatten()
    Vr4=V4r.flatten()
    Hg4=H4g.flatten()
    Hb4=H4b.flatten()
    Hr4=H4r.flatten()
    Dg4=D4g.flatten()
    Db4=D4b.flatten()
    Dr4=D4r.flatten()
    Qvg1=np.column_stack((shift(V1g, -1, 0), shift(V1g, 1, 0), shift(V1g, -1, 1), shift(V1g, 1, 1), scale(V2g, V1g.shape), np.abs(Dg1), scale(D2g, V1g.shape), np.abs(Vr1), np.abs(Vb1)))
    Qvb1=np.column_stack((shift(V1b, -1, 0), shift(V1b, 1, 0), shift(V1b, -1, 1), shift(V1b, 1, 1), scale(V2b, V1g.shape), np.abs(Db1), scale(D2b, V1g.shape), np.abs(Vb1), np.abs(Vg1)))
    Qvr1=np.column_stack((shift(V1r, -1, 0), shift(V1r, 1, 0), shift(V1r, -1, 1), shift(V1r, 1, 1), scale(V2r, V1g.shape), np.abs(Dr1), scale(D2r, V1g.shape), np.abs(Vg1), np.abs(Vr1)))
    Qhg1=np.column_stack((shift(H1g, -1, 0), shift(H1g, 1, 0), shift(H1g, -1, 1), shift(H1g, 1, 1), scale(H2g, V1g.shape), np.abs(Dg1), scale(D2g, V1g.shape), np.abs(Hr1), np.abs(Hb1)))
    Qhb1=np.column_stack((shift(H1b, -1, 0), shift(H1b, 1, 0), shift(H1b, -1, 1), shift(H1b, 1, 1), scale(H2b, V1g.shape), np.abs(Db1), scale(D2b, V1g.shape), np.abs(Hb1), np.abs(Hg1)))
    Qhr1=np.column_stack((shift(H1r, -1, 0), shift(H1r, 1, 0), shift(H1r, -1, 1), shift(H1r, 1, 1), scale(H2r, V1g.shape), np.abs(Dr1), scale(D2r, V1g.shape), np.abs(Hg1), np.abs(Hr1)))
    Qdg1=np.column_stack((shift(D1g, -1, 0), shift(D1g, 1, 0), shift(D1g, -1, 1), shift(D1g, 1, 1), scale(D2g, V1g.shape), np.abs(Hg1), np.abs(Vg1), np.abs(Dr1), np.abs(Db1)))
    Qdb1=np.column_stack((shift(D1b, -1, 0), shift(D1b, 1, 0), shift(D1b, -1, 1), shift(D1b, 1, 1), scale(D2b, V1g.shape), np.abs(Hb1), np.abs(Vb1), np.abs(Db1), np.abs(Dg1)))
    Qdr1=np.column_stack((shift(D1r, -1, 0), shift(D1r, 1, 0), shift(D1r, -1, 1), shift(D1r, 1, 1), scale(D2r, V1g.shape), np.abs(Hr1), np.abs(Vr1), np.abs(Dg1), np.abs(Dr1)))
    
    wvg1=np.matmul(np.linalg.pinv(Qvg1), Vg1).squeeze()
    wvb1=np.matmul(np.linalg.pinv(Qvb1), Vb1).squeeze()
    wvr1=np.matmul(np.linalg.pinv(Qvr1), Vr1).squeeze()
    whg1=np.matmul(np.linalg.pinv(Qhg1), Hg1).squeeze()
    whb1=np.matmul(np.linalg.pinv(Qhb1), Hb1).squeeze()
    whr1=np.matmul(np.linalg.pinv(Qhr1), Hr1).squeeze()
    wdg1=np.matmul(np.linalg.pinv(Qdg1), Dg1).squeeze()
    wdb1=np.matmul(np.linalg.pinv(Qdb1), Db1).squeeze()
    wdr1=np.matmul(np.linalg.pinv(Qdr1), Dr1).squeeze()
    
    xvr1=np.abs(np.matmul(Qvr1, wvr1))
    xvb1=np.abs(np.matmul(Qvb1, wvb1))
    xvg1=np.abs(np.matmul(Qvg1, wvg1))
    xhr1=np.abs(np.matmul(Qhr1, whr1))
    xhb1=np.abs(np.matmul(Qhr1, whr1))
    xhg1=np.abs(np.matmul(Qhr1, whr1))
    xdr1=np.abs(np.matmul(Qdr1, wdr1))
    xdb1=np.abs(np.matmul(Qdr1, wdr1))
    xdg1=np.abs(np.matmul(Qdr1, wdr1))
    xvr1[xvr1<=1e-10]=1e-10
    xvb1[xvb1<=1e-10]=1e-10
    xvg1[xvg1<=1e-10]=1e-10
    xhr1[xhr1<=1e-10]=1e-10
    xhb1[xhb1<=1e-10]=1e-10
    xhg1[xhg1<=1e-10]=1e-10
    xdr1[xdr1<=1e-10]=1e-10
    xdb1[xdb1<=1e-10]=1e-10
    xdg1[xdg1<=1e-10]=1e-10
    Vr1[Vr1<=1e-10]=1e-10
    Vb1[Vb1<=1e-10]=1e-10
    Vg1[Vg1<=1e-10]=1e-10
    Hr1[Hr1<=1e-10]=1e-10
    Hb1[Hb1<=1e-10]=1e-10
    Hg1[Hg1<=1e-10]=1e-10
    Dr1[Dr1<=1e-10]=1e-10
    Db1[Db1<=1e-10]=1e-10
    Dg1[Dg1<=1e-10]=1e-10
    pvr1=np.nan_to_num(np.log(Vr1/xvr1), nan=0, posinf=0, neginf=0)
    pvg1=np.nan_to_num(np.log(Vg1/xvg1), nan=0, posinf=0, neginf=0)
    pvb1=np.nan_to_num(np.log(Vb1/xvb1), nan=0, posinf=0, neginf=0)
    phr1=np.nan_to_num(np.log(Hr1/xhr1), nan=0, posinf=0, neginf=0)
    phg1=np.nan_to_num(np.log(Hg1/xhg1), nan=0, posinf=0, neginf=0)
    phb1=np.nan_to_num(np.log(Hb1/xhb1), nan=0, posinf=0, neginf=0)
    pdr1=np.nan_to_num(np.log(Dr1/xdr1), nan=0, posinf=0, neginf=0)
    pdg1=np.nan_to_num(np.log(Dg1/xdg1), nan=0, posinf=0, neginf=0)
    pdb1=np.nan_to_num(np.log(Db1/xdb1), nan=0, posinf=0, neginf=0)
######################################################################################################################
    Qvg2=np.column_stack((shift(V2g, -1, 0), shift(V2g, 1, 0), shift(V2g, -1, 1), shift(V2g, 1, 1), scale(V3g, V2g.shape), np.abs(Dg2), scale(D3g, V2g.shape), np.abs(Vr2), np.abs(Vb2)))
    Qvb2=np.column_stack((shift(V2b, -1, 0), shift(V2b, 1, 0), shift(V2b, -1, 1), shift(V2b, 1, 1), scale(V3b, V2g.shape), np.abs(Db2), scale(D3b, V2g.shape), np.abs(Vb2), np.abs(Vg2)))
    Qvr2=np.column_stack((shift(V2r, -1, 0), shift(V2r, 1, 0), shift(V2r, -1, 1), shift(V2r, 1, 1), scale(V3r, V2g.shape), np.abs(Dr2), scale(D3r, V2g.shape), np.abs(Vg2), np.abs(Vr2)))
    Qhg2=np.column_stack((shift(H2g, -1, 0), shift(H2g, 1, 0), shift(H2g, -1, 1), shift(H2g, 1, 1), scale(H3g, V2g.shape), np.abs(Dg2), scale(D3g, V2g.shape), np.abs(Hr2), np.abs(Hb2)))
    Qhb2=np.column_stack((shift(H2b, -1, 0), shift(H2b, 1, 0), shift(H2b, -1, 1), shift(H2b, 1, 1), scale(H3b, V2g.shape), np.abs(Db2), scale(D3b, V2g.shape), np.abs(Hb2), np.abs(Hg2)))
    Qhr2=np.column_stack((shift(H2r, -1, 0), shift(H2r, 1, 0), shift(H2r, -1, 1), shift(H2r, 1, 1), scale(H3r, V2g.shape), np.abs(Dr2), scale(D3r, V2g.shape), np.abs(Hg2), np.abs(Hr2)))
    Qdg2=np.column_stack((shift(D2g, -1, 0), shift(D2g, 1, 0), shift(D2g, -1, 1), shift(D2g, 1, 1), scale(D3g, V2g.shape), np.abs(Hg2), np.abs(Vg2), np.abs(Dr2), np.abs(Db2)))
    Qdb2=np.column_stack((shift(D2b, -1, 0), shift(D2b, 1, 0), shift(D2b, -1, 1), shift(D2b, 1, 1), scale(D3b, V2g.shape), np.abs(Hb2), np.abs(Vb2), np.abs(Db2), np.abs(Dg2)))
    Qdr2=np.column_stack((shift(D2r, -1, 0), shift(D2r, 1, 0), shift(D2r, -1, 1), shift(D2r, 1, 1), scale(D3r, V2g.shape), np.abs(Hr2), np.abs(Vr2), np.abs(Dg2), np.abs(Dr2)))
    
    wvg2=np.matmul(np.linalg.pinv(Qvg2), Vg2).squeeze()
    wvb2=np.matmul(np.linalg.pinv(Qvb2), Vb2).squeeze()
    wvr2=np.matmul(np.linalg.pinv(Qvr2), Vr2).squeeze()
    whg2=np.matmul(np.linalg.pinv(Qhg2), Hg2).squeeze()
    whb2=np.matmul(np.linalg.pinv(Qhb2), Hb2).squeeze()
    whr2=np.matmul(np.linalg.pinv(Qhr2), Hr2).squeeze()
    wdg2=np.matmul(np.linalg.pinv(Qdg2), Dg2).squeeze()
    wdb2=np.matmul(np.linalg.pinv(Qdb2), Db2).squeeze()
    wdr2=np.matmul(np.linalg.pinv(Qdr2), Dr2).squeeze()
    
    xvr2=np.abs(np.matmul(Qvr2, wvr2))
    xvb2=np.abs(np.matmul(Qvb2, wvb2))
    xvg2=np.abs(np.matmul(Qvg2, wvg2))
    xhr2=np.abs(np.matmul(Qhr2, whr2))
    xhb2=np.abs(np.matmul(Qhr2, whr2))
    xhg2=np.abs(np.matmul(Qhr2, whr2))
    xdr2=np.abs(np.matmul(Qdr2, wdr2))
    xdb2=np.abs(np.matmul(Qdr2, wdr2))
    xdg2=np.abs(np.matmul(Qdr2, wdr2))
    xvr2[xvr2<=1e-10]=1e-10
    xvb2[xvb2<=1e-10]=1e-10
    xvg2[xvg2<=1e-10]=1e-10
    xhr2[xhr2<=1e-10]=1e-10
    xhb2[xhb2<=1e-10]=1e-10
    xhg2[xhg2<=1e-10]=1e-10
    xdr2[xdr2<=1e-10]=1e-10
    xdb2[xdb2<=1e-10]=1e-10
    xdg2[xdg2<=1e-10]=1e-10
    Vr3[Vr3<=1e-10]=1e-10
    Vb3[Vb3<=1e-10]=1e-10
    Vg3[Vg3<=1e-10]=1e-10
    Hr3[Hr3<=1e-10]=1e-10
    Hb3[Hb3<=1e-10]=1e-10
    Hg3[Hg3<=1e-10]=1e-10
    Dr3[Dr3<=1e-10]=1e-10
    Db3[Db3<=1e-10]=1e-10
    Dg3[Dg3<=1e-10]=1e-10
    pvr2=np.nan_to_num(np.log(Vr2/xvr2), nan=0, posinf=0, neginf=0)
    pvg2=np.nan_to_num(np.log(Vg2/xvg2), nan=0, posinf=0, neginf=0)
    pvb2=np.nan_to_num(np.log(Vb2/xvb2), nan=0, posinf=0, neginf=0)
    phr2=np.nan_to_num(np.log(Hr2/xhr2), nan=0, posinf=0, neginf=0)
    phg2=np.nan_to_num(np.log(Hg2/xhg2), nan=0, posinf=0, neginf=0)
    phb2=np.nan_to_num(np.log(Hb2/xhb2), nan=0, posinf=0, neginf=0)
    pdr2=np.nan_to_num(np.log(Dr2/xdr2), nan=0, posinf=0, neginf=0)
    pdg2=np.nan_to_num(np.log(Dg2/xdg2), nan=0, posinf=0, neginf=0)
    pdb2=np.nan_to_num(np.log(Db2/xdb2), nan=0, posinf=0, neginf=0)

###################################################################################################################

    Qvg3=np.column_stack((shift(V3g, -1, 0), shift(V3g, 1, 0), shift(V3g, -1, 1), shift(V3g, 1, 1), scale(V4g, V3g.shape), np.abs(Dg3), scale(D4g, V3g.shape), np.abs(Vr3), np.abs(Vb3)))
    Qvb3=np.column_stack((shift(V3b, -1, 0), shift(V3b, 1, 0), shift(V3b, -1, 1), shift(V3b, 1, 1), scale(V4b, V3g.shape), np.abs(Db3), scale(D4b, V3g.shape), np.abs(Vb3), np.abs(Vg3)))
    Qvr3=np.column_stack((shift(V3r, -1, 0), shift(V3r, 1, 0), shift(V3r, -1, 1), shift(V3r, 1, 1), scale(V4r, V3g.shape), np.abs(Dr3), scale(D4r, V3g.shape), np.abs(Vg3), np.abs(Vr3)))
    Qhg3=np.column_stack((shift(H3g, -1, 0), shift(H3g, 1, 0), shift(H3g, -1, 1), shift(H3g, 1, 1), scale(H4g, V3g.shape), np.abs(Dg3), scale(D4g, V3g.shape), np.abs(Hr3), np.abs(Hb3)))
    Qhb3=np.column_stack((shift(H3b, -1, 0), shift(H3b, 1, 0), shift(H3b, -1, 1), shift(H3b, 1, 1), scale(H4b, V3g.shape), np.abs(Db3), scale(D4b, V3g.shape), np.abs(Hb3), np.abs(Hg3)))
    Qhr3=np.column_stack((shift(H3r, -1, 0), shift(H3r, 1, 0), shift(H3r, -1, 1), shift(H3r, 1, 1), scale(H4r, V3g.shape), np.abs(Dr3), scale(D4r, V3g.shape), np.abs(Hg3), np.abs(Hr3)))
    Qdg3=np.column_stack((shift(D3g, -1, 0), shift(D3g, 1, 0), shift(D3g, -1, 1), shift(D3g, 1, 1), scale(D4g, V3g.shape), np.abs(Hg3), np.abs(Vg3), np.abs(Dr3), np.abs(Db3)))
    Qdb3=np.column_stack((shift(D3b, -1, 0), shift(D3b, 1, 0), shift(D3b, -1, 1), shift(D3b, 1, 1), scale(D4b, V3g.shape), np.abs(Hb3), np.abs(Vb3), np.abs(Db3), np.abs(Dg3)))
    Qdr3=np.column_stack((shift(D3r, -1, 0), shift(D3r, 1, 0), shift(D3r, -1, 1), shift(D3r, 1, 1), scale(D4r, V3g.shape), np.abs(Hr3), np.abs(Vr3), np.abs(Dg3), np.abs(Dr3)))
    
    wvg3=np.matmul(np.linalg.pinv(Qvg3), Vg3).squeeze()
    wvb3=np.matmul(np.linalg.pinv(Qvb3), Vb3).squeeze()
    wvr3=np.matmul(np.linalg.pinv(Qvr3), Vr3).squeeze()
    whg3=np.matmul(np.linalg.pinv(Qhg3), Hg3).squeeze()
    whb3=np.matmul(np.linalg.pinv(Qhb3), Hb3).squeeze()
    whr3=np.matmul(np.linalg.pinv(Qhr3), Hr3).squeeze()
    wdg3=np.matmul(np.linalg.pinv(Qdg3), Dg3).squeeze()
    wdb3=np.matmul(np.linalg.pinv(Qdb3), Db3).squeeze()
    wdr3=np.matmul(np.linalg.pinv(Qdr3), Dr3).squeeze()
    
    xvr3=np.abs(np.matmul(Qvr3, wvr3))
    xvb3=np.abs(np.matmul(Qvb3, wvb3))
    xvg3=np.abs(np.matmul(Qvg3, wvg3))
    xhr3=np.abs(np.matmul(Qhr3, whr3))
    xhb3=np.abs(np.matmul(Qhr3, whr3))
    xhg3=np.abs(np.matmul(Qhr3, whr3))
    xdr3=np.abs(np.matmul(Qdr3, wdr3))
    xdb3=np.abs(np.matmul(Qdr3, wdr3))
    xdg3=np.abs(np.matmul(Qdr3, wdr3))
    xvr3[xvr3<=1e-10]=1e-10
    xvb3[xvb3<=1e-10]=1e-10
    xvg3[xvg3<=1e-10]=1e-10
    xhr3[xhr3<=1e-10]=1e-10
    xhb3[xhb3<=1e-10]=1e-10
    xhg3[xhg3<=1e-10]=1e-10
    xdr3[xdr3<=1e-10]=1e-10
    xdb3[xdb3<=1e-10]=1e-10
    xdg3[xdg3<=1e-10]=1e-10
    Vr3[Vr3<=1e-10]=1e-10
    Vb3[Vb3<=1e-10]=1e-10
    Vg3[Vg3<=1e-10]=1e-10
    Hr3[Hr3<=1e-10]=1e-10
    Hb3[Hb3<=1e-10]=1e-10
    Hg3[Hg3<=1e-10]=1e-10
    Dr3[Dr3<=1e-10]=1e-10
    Db3[Db3<=1e-10]=1e-10
    Dg3[Dg3<=1e-10]=1e-10
    pvr3=np.nan_to_num(np.log(Vr3/xvr3), nan=0, posinf=0, neginf=0)
    pvg3=np.nan_to_num(np.log(Vg3/xvg3), nan=0, posinf=0, neginf=0)
    pvb3=np.nan_to_num(np.log(Vb3/xvb3), nan=0, posinf=0, neginf=0)
    phr3=np.nan_to_num(np.log(Hr3/xhr3), nan=0, posinf=0, neginf=0)
    phg3=np.nan_to_num(np.log(Hg3/xhg3), nan=0, posinf=0, neginf=0)
    phb3=np.nan_to_num(np.log(Hb3/xhb3), nan=0, posinf=0, neginf=0)
    pdr3=np.nan_to_num(np.log(Dr3/xdr3), nan=0, posinf=0, neginf=0)
    pdg3=np.nan_to_num(np.log(Dg3/xdg3), nan=0, posinf=0, neginf=0)
    pdb3=np.nan_to_num(np.log(Db3/xdb3), nan=0, posinf=0, neginf=0)

    stat=[np.mean(Vr1), np.mean(Vb1), np.mean(Vb1), np.mean(Hr1), np.mean(Hg1), np.mean(Hb1), np.mean(Dr1), np.mean(Dg1), np.mean(Db1), 
    np.var(Vr1), np.var(Vb1), np.var(Vb1), np.var(Hr1), np.var(Hg1), np.var(Hb1), np.var(Dr1), np.var(Dg1), np.var(Db1), 
    stats.skew(Vr1), stats.skew(Vb1), stats.skew(Vb1), stats.skew(Hr1), stats.skew(Hg1), stats.skew(Hb1), stats.skew(Dr1), stats.skew(Dg1), stats.skew(Db1), 
    stats.kurtosis(Vr1), stats.kurtosis(Vb1), stats.kurtosis(Vb1), stats.kurtosis(Hr1), stats.kurtosis(Hg1), stats.kurtosis(Hb1), stats.kurtosis(Dr1), stats.kurtosis(Dg1), stats.kurtosis(Db1), 
    np.mean(xvr1), np.mean(xvg1), np.mean(xvb1), np.mean(xhr1), np.mean(xhg1), np.mean(xhb1), np.mean(xdr1), np.mean(xdg1), np.mean(xdb1),
    np.var(xvr1), np.var(xvg1), np.var(xvb1), np.var(xhr1), np.var(xhg1), np.var(xhb1), np.var(xdr1), np.var(xdg1), np.var(xdb1),
    stats.skew(xvr1), stats.skew(xvg1), stats.skew(xvb1), stats.skew(xhr1), stats.skew(xhg1), stats.skew(xhb1), stats.skew(xdr1), stats.skew(xdg1), stats.skew(xdb1),
    stats.kurtosis(xvr1), stats.kurtosis(xvg1), stats.kurtosis(xvb1), stats.kurtosis(xhr1), stats.kurtosis(xhg1), stats.kurtosis(xhb1), stats.kurtosis(xdr1), stats.kurtosis(xdg1), stats.kurtosis(xdb1),
    
    np.mean(Vr2), np.mean(Vb2), np.mean(Vb2), np.mean(Hr2), np.mean(Hg2), np.mean(Hb2), np.mean(Dr2), np.mean(Dg2), np.mean(Db2), 
    np.var(Vr2), np.var(Vb2), np.var(Vb2), np.var(Hr2), np.var(Hg2), np.var(Hb2), np.var(Dr2), np.var(Dg2), np.var(Db2), 
    stats.skew(Vr2), stats.skew(Vb2), stats.skew(Vb2), stats.skew(Hr2), stats.skew(Hg2), stats.skew(Hb2), stats.skew(Dr2), stats.skew(Dg2), stats.skew(Db2), 
    stats.kurtosis(Vr2), stats.kurtosis(Vb2), stats.kurtosis(Vb2), stats.kurtosis(Hr2), stats.kurtosis(Hg2), stats.kurtosis(Hb2), stats.kurtosis(Dr2), stats.kurtosis(Dg2), stats.kurtosis(Db2), 
    np.mean(xvr2), np.mean(xvg2), np.mean(xvb2), np.mean(xhr2), np.mean(xhg2), np.mean(xhb2), np.mean(xdr2), np.mean(xdg2), np.mean(xdb2),
    np.var(xvr2), np.var(xvg2), np.var(xvb2), np.var(xhr2), np.var(xhg2), np.var(xhb2), np.var(xdr2), np.var(xdg2), np.var(xdb2),
    stats.skew(xvr2), stats.skew(xvg2), stats.skew(xvb2), stats.skew(xhr2), stats.skew(xhg2), stats.skew(xhb2), stats.skew(xdr2), stats.skew(xdg2), stats.skew(xdb2),
    stats.kurtosis(xvr2), stats.kurtosis(xvg2), stats.kurtosis(xvb2), stats.kurtosis(xhr2), stats.kurtosis(xhg2), stats.kurtosis(xhb2), stats.kurtosis(xdr2), stats.kurtosis(xdg2), stats.kurtosis(xdb2),
    
    np.mean(Vr3), np.mean(Vb3), np.mean(Vb3), np.mean(Hr3), np.mean(Hg3), np.mean(Hb3), np.mean(Dr3), np.mean(Dg3), np.mean(Db3), 
    np.var(Vr3), np.var(Vb3), np.var(Vb3), np.var(Hr3), np.var(Hg3), np.var(Hb3), np.var(Dr3), np.var(Dg3), np.var(Db3), 
    stats.skew(Vr3), stats.skew(Vb3), stats.skew(Vb3), stats.skew(Hr3), stats.skew(Hg3), stats.skew(Hb3), stats.skew(Dr3), stats.skew(Dg3), stats.skew(Db3), 
    stats.kurtosis(Vr3), stats.kurtosis(Vb3), stats.kurtosis(Vb3), stats.kurtosis(Hr3), stats.kurtosis(Hg3), stats.kurtosis(Hb3), stats.kurtosis(Dr3), stats.kurtosis(Dg3), stats.kurtosis(Db3), 
    np.mean(xvr3), np.mean(xvg3), np.mean(xvb3), np.mean(xhr3), np.mean(xhg3), np.mean(xhb3), np.mean(xdr3), np.mean(xdg3), np.mean(xdb3),
    np.var(xvr3), np.var(xvg3), np.var(xvb3), np.var(xhr3), np.var(xhg3), np.var(xhb3), np.var(xdr3), np.var(xdg3), np.var(xdb3),
    stats.skew(xvr3), stats.skew(xvg3), stats.skew(xvb3), stats.skew(xhr3), stats.skew(xhg3), stats.skew(xhb3), stats.skew(xdr3), stats.skew(xdg3), stats.skew(xdb3),
    stats.kurtosis(xvr3), stats.kurtosis(xvg3), stats.kurtosis(xvb3), stats.kurtosis(xhr3), stats.kurtosis(xhg3), stats.kurtosis(xhb3), stats.kurtosis(xdr3), stats.kurtosis(xdg3), stats.kurtosis(xdb3),
    ]
    if X is None:
        X=stat
    else:
        X=np.vstack((X, stat))
    Y.append(1)


joblib.dump(X, 'X.pkl')
clf=LogisticRegression(max_iter=1000000).fit(X, Y)
print(clf.score(X, Y))
joblib.dump(clf, 'data.pkl')