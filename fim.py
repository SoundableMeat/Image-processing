from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy as sp

def convert_grayscale(imname):
    pil_im = Image.open(imname).convert('LA')
    IM = np.array(list(pil_im.getdata(band=0)), float)
    IM.shape = (pil_im.size[1], pil_im.size[0])

    return IM

def gausshigh(IM,D0):

    x,y = np.shape(IM)

    u = np.arange(int(-x/2),int(x/2),1)
    v = np.arange(int(-y/2),int(y/2),1)
    D = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            D[i][j]=np.sqrt(u[i]**2+v[j]**2)

    H = 1-np.exp(-D**2/(2*D0**2))

    freq_IM = np.fft.fftshift(np.fft.fft2(IM))

    filtered_freq = freq_IM*H

    filtered_IM = np.abs(np.fft.ifft2(filtered_freq))

    return filtered_IM

def median_filter(IM,size):

    padding_size = int((size-1)/2)
    IM_size = np.shape(IM)
    x_size = IM_size[0]+2*padding_size
    y_size = IM_size[1]+2*padding_size

    padded_IM = np.zeros((x_size,y_size))

    padded_IM[padding_size:IM.shape[0]+padding_size,padding_size:IM.shape[1]+padding_size] = IM
    new_IM = np.zeros_like(IM)

    for i in range(padding_size,padding_size+IM_size[0]):
        for j in range(padding_size,padding_size+IM_size[1]):
            tmp_arr = padded_IM[i-padding_size:i+padding_size+1,j-padding_size:j+padding_size+1]
            median = np.median(tmp_arr)
            new_IM[i-padding_size][j-padding_size] = median

    return new_IM

def hist_eq(arr,L):
    amp=np.zeros(L)
    for list in arr:
        for number in list:
            amp[number]+=1

    M,N = np.shape(arr)

    pk = amp/(M*N)

    s = []

    for i in range(L):
        tmp = int((L-1)*(np.sum(pk[:i])))
        s.append(tmp)

    amp_out = np.zeros_like(amp)

    for i in range(L):
        amp_out[s[i]]+=amp[i]

    arr_out = np.zeros_like(arr)

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr_out[i][j]=s[arr[i][j]]

    return amp_out, arr_out

def convert_array_1(arr,L):
    new_arr=arr/L

    return new_arr

def convert_array_L(arr,arr_max):
    new_arr=arr*arr_max
    return new_arr

def my_linear_int(IM,newsize,scale):
    new_IM=np.zeros(newsize)
    enlargement=len(IM)*(scale-1)/(len(IM)-1)+1

    for j in range(len(new_IM)):
        for i in range(len(new_IM[j])):
            interpolation_left = (1-i%enlargement/enlargement)*IM[int(np.floor(i/enlargement))][int(np.floor(j/enlargement))] \
            +(i%enlargement/enlargement)*IM[int(np.ceil(i/enlargement))][int(np.floor(j/enlargement))]

            interpolation_right = (1-i%enlargement/enlargement)*IM[int(np.floor(i/enlargement))][int(np.ceil(j/enlargement))] \
            +(i%enlargement/enlargement)*IM[int(np.ceil(i/enlargement))][int(np.ceil(j/enlargement))]

            interpolation = (1-j%enlargement/enlargement)*interpolation_left+(j%enlargement/enlargement)*interpolation_right

            new_IM[i][j]=interpolation

    return new_IM

def IM_zoomed_out(IM,newsize,method):
    a,b=np.shape(IM)
    x=np.arange(0,a)
    y=np.arange(0,b)
    xx=np.linspace(0,a,newsize[0])
    yy=np.linspace(0,b,newsize[1])

    if method=='linear' or method=='cubic':
        myInterpolator = sp.interpolate.interp2d(x,y,IM,method)
        return myInterpolator(xx,yy)

    elif method=='nearest':
        X,Y = np.meshgrid(x,y)
        X_array = np.reshape(X,(int(a*b),1))
        Y_array = np.reshape(Y,(int(a*b),1))

        points = np.zeros((int(a*b),2))
        points[:,0] = X_array[:,0]
        points[:,1] = Y_array[:,0]

        IM_reshape = np.reshape(IM, (1,int(a*b)))
        myInterpolator = sp.interpolate.NearestNDInterpolator(points,IM_reshape[0,:])
        XX,YY = np.meshgrid(xx,yy)
        return myInterpolator(XX,YY)


def gamma_transform(c,r_in,gamma):
    r = convert_array_1(r_in,255)

    s = c*(r**gamma)

    s_out = convert_array_L(s,255)

    return s_out
