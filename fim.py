from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy as sp
from scipy.signal import convolve2d

plt.rcParams['image.cmap'] = 'gray'

def arit_mean(IM,mask_size):
    """Applying a harmonic mean filter to image

    Parameters
    ----------
    IM : numpy array
        Containing the original image
    mask_size : integer
        Size of the kernel

    Returns
    -------
    numpy array
        Containing the filtered image

    """
    mask = np.ones((mask_size,mask_size))/mask_size**2

    blurred_IM = convolve2d(IM,mask)

    blurred_IM = blurred_IM.astype(np.uint8)

    return blurred_IM

def geom_mean(IM,mask_size):
    """Applying a geometric mean filter to image

    Parameters
    ----------
    IM : numpy array
        Containing the original image
    mask_size : integer
        Size of the kernel

    Returns
    -------
    numpy array
        Containing the filtered image

    """
    padding_size = int(mask_size/2)

    IM_size = np.shape(IM)
    x_size = IM_size[0]+2*padding_size
    y_size = IM_size[1]+2*padding_size

    padded_IM = np.zeros((x_size,y_size))
    padded_IM[padding_size:IM.shape[0]+padding_size,padding_size:IM.shape[1]+padding_size] = IM

    new_IM = np.zeros_like(IM)

    for i in range(padding_size,padding_size+IM_size[0]):
        for j in range(padding_size,padding_size+IM_size[1]):
            tmp_arr = padded_IM[i-padding_size:i+padding_size+1,j-padding_size:j+padding_size+1]
            mean = (np.prod(tmp_arr))**(1/(mask_size**2))
            new_IM[i-padding_size][j-padding_size] = mean

    return new_IM

def harm_mean(IM,mask_size):
    """Applying a harmonic mean filter to image

    Parameters
    ----------
    IM : numpy array
        Containing the original image
    mask_size : integer
        Size of the kernel

    Returns
    -------
    numpy array
        Containing the filtered image

    """
    padding_size = int(mask_size/2)

    IM_size = np.shape(IM)
    x_size = IM_size[0]+2*padding_size
    y_size = IM_size[1]+2*padding_size

    padded_IM = np.zeros((x_size,y_size))
    padded_IM[padding_size:IM.shape[0]+padding_size,padding_size:IM.shape[1]+padding_size] = IM

    new_IM = np.zeros_like(IM)

    for i in range(padding_size,padding_size+IM_size[0]):
        for j in range(padding_size,padding_size+IM_size[1]):
            tmp_arr = padded_IM[i-padding_size:i+padding_size+1,j-padding_size:j+padding_size+1]
            mean = mask_size**2/(np.sum(1/tmp_arr))
            new_IM[i-padding_size][j-padding_size] = mean

    return new_IM

def remove_last_digits(IM):
    """Removes least significant bits from grayscale image

    Parameters
    ----------
    IM : numpy array
        Original image

    Returns
    -------
    numpy array
        Output image

    """
    x,y=np.shape(IM)

    newIM = np.zeros_like(IM)
    for i in range(x):
        for j in range(y):
            tmp = format(IM[i,j], '#010b')
            first = '0b'
            mid = tmp[2:8]
            third = '00'
            str = first + mid + third
            result = int(str,2)
            newIM[i,j]=result

    return newIM

def convert_grayscale(imname):
    """Converts rgb images to grayscale

    Parameters
    ----------
    imname : string
        Contains the name and location of the image

    Returns
    -------
    numpy array
        Returnes array containing the pixel values of the image.

    """
    pil_im = Image.open(imname).convert('LA')
    IM = np.array(list(pil_im.getdata(band=0)), float)
    IM.shape = (pil_im.size[1], pil_im.size[0])

    return IM

def gaussfilter(IM,D0,high):
    """Apply a Gaussian high- or low-pass to image

    Parameters
    ----------
    IM : numpy array
        Fourier transform of image we want to filter
    D0 : integer
        Cutoff frequency
    high : True/False argument
        If True we get highpass and if False we get lowpass

    Returns
    -------
    numpy array
        Filtered Fourier transform of image

    """
    x,y = np.shape(IM)

    u = np.arange(int(-x/2),int(x/2),1)
    v = np.arange(int(-y/2),int(y/2),1)
    D = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            D[i][j]=np.sqrt(u[i]**2+v[j]**2)

    if high:
        H = 1-np.exp(-D**2/(2*D0**2))
    else:
        H = np.exp(-D**2/(2*D0**2))

    filtered_freq = IM*H

    return filtered_freq

def median_filter(IM,size):
    """Applying a median filter to image

    Parameters
    ----------
    IM : numpy array
        Containing the original image
    size : integer
        Size of the median mask

    Returns
    -------
    numpy array
        Containing the filtered image

    """
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
    """Equalizing the input histogram to stretch contrast

    Parameters
    ----------
    arr : numpy array
        Containing the histogram we want to equalize
    L : integer
        Maximum intensity of the new histogram

    Returns
    -------
    amp_out: numpy array
        Intensities corresponding to the new histogram
    arr_out: numpy array
        The new hisogram

    """
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
    """Converts n-bit array to float between 0 and 1

    Parameters
    ----------
    arr : numpy array
        Array we want to convert
    L : integer
        Maximum intensity of array

    Returns
    -------
    numpy array
        Floating point array with intensities between 0 and 1

    """
    new_arr=arr/L

    return new_arr

def convert_array_L(arr,arr_max):
    """Converts floating point array to n-bit array

    Parameters
    ----------
    arr : numpy array
        Floating point array
    arr_max : integer
        Max intensity of output array

    Returns
    -------
    numpy array
        Containes the output array with max inlencity arr_max

    """
    new_arr=arr*arr_max
    return new_arr.astype(np.uint8)

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

def downsample(IM,newshape):
    """Downsamples an image to new size.

    Parameters
    ----------
    IM : numpy array
        Containes the intensities of the original image
    newshape : numpy array
        Shape of the new image

    Returns
    -------
    numpy array
        Containes the intensities of the downsampled image

    """
    newIM = np.zeros(newshape,np.uint8)

    for i in range(newshape[0]):
        for j in range(newshape[1]):
            newIM[i,j] = IM[2*i,2*j]

    return newIM


def gamma_transform(c,IM,gamma):
    """Gamma transforms a grayscale image

    Parameters
    ----------
    c : integer
        Gamma transform constant
    r_in : numpy array
        Contains the input grayscale image
    gamma : integer
        Gamma for the gamma transform

    Returns
    -------
    numpy array
        containes the new grayscale image

    """
    r = convert_array_1(IM,255)

    s = c*(r**gamma)

    newIM = convert_array_L(s,255)

    return newIM

def histogram(IM):
    """Find the histogram of the imput image

    Parameters
    ----------
    IM : numpy array
        Containes a grayscale immage

    Returns
    -------
    numpy array
        Containes the hisogram if the image

    """
    tmp=[]
    a=np.unique(IM)
    for i in range(0,256):
        if i not in a:
            tmp.append(i)
    hist = np.append(IM,tmp)

    return hist


def notch(IM,D0,freqs,n):
    """Makes notch filter from several frequencies

    Parameters
    ----------
    IM : numpy array
        Containing the original image
    D0 : integer
        Cutoff frequency for notch filter
    freqs : Arry
        Containing the frequencies we notch out
    n : integer
        Order of the Butterworth filter

    Returns
    -------
    numpy array
        Floating point array containing the product of all notch filters.

    """
    filters=[]
    for val in freqs:
        M,N = np.shape(IM)
        tmp_filter = np.zeros((M,N))
        for i in range(M):
            for j in range(N):
                D_x_pos = np.sqrt((i-M/2-val[0])**2+(j-N/2-val[1])**2)
                D_x_neg = np.sqrt((i-M/2+val[0])**2+(j-N/2+val[1])**2)

                if D_x_pos == 0:
                    prod1 = 0
                else:
                    prod1 = 1/(1+(D0/D_x_pos)**n)

                if D_x_neg == 0:
                    prod2 = 0
                else:
                    prod2 = 1/(1+(D0/D_x_neg)**n)

                prod = prod1*prod2

                tmp_filter[i,j] = prod
        tmp_filter = np.array(tmp_filter)
        filters.append(tmp_filter)

    filters = np.array(filters)

    notches = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            tmp = [filter[i,j] for filter in filters]
            notches[i,j] = np.prod(tmp)

    return notches

def blur_image(IM,mask_size):
    """Blurr an image using convolution

    Parameters
    ----------
    IM : numpy array
        Containes the intensities of the original image
    mask_size : integer
        Size of the blurring mask

    Returns
    -------
    numpy array
        Containes the intensities of the blurred image

    """
    mask = np.ones((mask_size,mask_size))/mask_size**2

    blurred_IM = convolve2d(IM,mask)

    blurred_IM = blurred_IM.astype(np.uint8)

    return blurred_IM

def reshape_image(IM):
    """Takes RGB image with shape (3,x,y) and converts it to (x,y,3)

    Parameters
    ----------
    IM : numpy array
        Contanies the RGB image

    Returns
    -------
    numpy array
        Containes the new RGB image

    """
    z,x,y = np.shape(IM)

    newIM = np.zeros((x,y,z))

    for i in range(x):
        for j in range(y):
            for k in range(z):
                newIM[i,j,k] = IM[k,i,j]

    newIM = newIM.astype(np.uint8)

    return newIM

def hide_picture(toppic,hidden):
    """Hides a full RGB image in the last bits of another image.
       The hidden image must be at least three times smaller in
       x-direction than the other image.

    Parameters
    ----------
    toppic : numpy array
        Containing the image to hide the other in
    hidden : numpy array
        Contains the soon to be hidden image

    Returns
    -------
    numpy array
        Containes the toppic with an image hidden in the last bits

    """
    x,y,z = np.shape(hidden)
    hidden_array = np.zeros((x,3*y,z))
    for i in range(x):
        for j in range(y):
            for k in range(z):
                tmp = format(hidden[i,j,k], '#010b')
                hidden_array[i,3*j  ,k] = tmp[2:4]
                hidden_array[i,3*j+1,k] = tmp[4:6]
                hidden_array[i,3*j+2,k] = tmp[6:8]


    x,y,z = np.shape(hidden_array)
    x1,y1,z1 = np.shape(toppic)
    newtop = np.zeros_like(toppic)
    for i in range(x1):
        for j in range(y1):
            for k in range(z1):
                tmp = format(toppic[i,j,k], '#010b')
                first = '0b'
                mid = tmp[2:8]
                if i<x and j<y:
                    third = str(int(hidden_array[i,j,k]))
                    if third == '0':
                        third = '00'
                    if third == '1':
                        third = '01'
                else:
                    third = '00'
                string = first + mid + third
                result = int(string,2)
                newtop[i,j,k]=result

    return newtop

def find_picture(IM):
    """Find image hidden in least significant bits

    Parameters
    ----------
    IM : numpy array
        Contains visible image with another image hidden
        in the least significant bits

    Returns
    -------
    numpy array
        Contains the previously hidden image

    """
    x,y,z = np.shape(IM)

    new_arr = np.zeros_like(IM)
    for k in range(z):
        for i in range(x):
            count = 0
            for j in range(y):
                tmp = format(IM[i,j,k], '#010b')
                hidden = tmp[8:]
                new_arr[i,j,k] = hidden

    new_IM = np.zeros((x,y,z))
    for i in range(x):
        for j in range(int(y/3)):
            for k in range(z):
                first = str(new_arr[i,3*j  ,k])
                if first == '0':
                    first = '00'
                elif first == '1':
                    first = '01'
                second = str(new_arr[i,3*j+1,k])
                if second == '0':
                    second = '00'
                elif second == '1':
                    second = '01'
                third = str(new_arr[i,3*j+2,k])
                if third == '0':
                    third = '00'
                elif third == '1':
                    third = '01'
                tmp = '0b'+first+second+third+'00'
                result = int(tmp,2)
                new_IM[i,j,k] = result


    """
    This Part is for reshaping the hidden image,
    so it won't have the shape of the image it was hidden
    inside of
    """
    x,y,z = np.shape(new_IM)
    for i in range(1,x-1):
        tmp = new_IM[i-1:i+2,4:7]
        if np.sum(tmp)==0:
            break
    for j in range(1,y-1):
        tmp = new_IM[4:7,j-1:j+2]
        if np.sum(tmp)==0:
            break

    newIM = new_IM[:i,:j]

    return newIM
