from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy as sp
from scipy.signal import convolve2d
from fim import *
import cv2

def encrypt():
    imname = "hobby_pics/martin.jpg"
    IM = convert_grayscale(imname)

    newIM = imageencrypt(IM,True)

    orgIM = imageencrypt(newIM,False)

    plt.imshow(IM,cmap='gray')
    plt.show()

    plt.imshow(np.abs(newIM),cmap='gray')
    plt.show()

    plt.imshow(np.abs(orgIM),cmap='gray')
    plt.show()

def bits():
    imname = "hobby_pics/martin.jpg"
    IM = convert_grayscale(imname).astype(np.uint8)

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

    cv2.imshow('image',IM)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('image',newIM)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def stugt():
    imname = "hobby_pics/martin.jpg"
    IM = convert_grayscale(imname).astype(np.uint8)

    freq = np.fft.fftshift(np.fft.fft2(IM))
    tmp = np.log(freq)

    remin = np.min(tmp.real)
    immin = np.min(tmp.imag)

    tmp_real = tmp.real-remin

    tmp_imag = tmp.real-immin

    four = np.vectorize(complex)(tmp_real+remin,tmp_imag+immin)

    print(four,'\n','\n',freq)

    newIM = np.fft.ifft2(np.exp(four))

    """cv2.imshow('image',tmp.real.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    plt.imshow(np.abs(newIM))
    plt.show()

def highpass_merge():
    imname = "hobby_pics/kermit.jpeg"
    IM = convert_grayscale(imname).astype(np.uint8)

    high_martin = gaussfilter(IM,10,True).astype(np.uint8)

    cv2.imshow('image',high_martin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def paint_filter():
    IM1 = cv2.imread('home_exam_pictures/F3.png',0)
    IM2 = cv2.imread('home_exam_pictures/F4.png',0)

    four1 = np.log(np.fft.fftshift(np.fft.fft2(IM1)))
    remax = np.max(four1.real)
    immax = np.max(four1.imag)
    refour1 = four1.real*255/remax
    imfour1 = four1.imag*255/immax
    #four2 = np.abs(np.log(np.fft.fftshift(np.fft.fft2(IM2)))).astype(np.uint8)

    #cv2.imwrite('hobby_pics/four_real.PNG',refour1)
    #cv2.imwrite('hobby_pics/four_imag.PNG',imfour1)

    newrefour = cv2.imread('hobby_pics/four_real.PNG',0)
    newrefour = newrefour*remax/255

    newimfour = cv2.imread('hobby_pics/four_imag.PNG',0)
    newimfour = newimfour*immax/255

    newfour = np.vectorize(complex)(newrefour,newimfour)

    newIM1 = np.abs(np.fft.ifft2(np.exp(newfour)))

    newIM1 = gamma_transform(10,newIM1,0.0001)

    #newIM1 = np.fft.ifft2(np.exp(four1*max/255))

    # plt.imshow(four1.real)
    # plt.show()

    # plt.imshow(np.abs(four2))
    # plt.show()

    plt.imshow(newIM1)
    plt.show()


def hide_picture(toppic,hidden):
    x,y = np.shape(hidden)
    hidden_array = np.zeros((3*x,y))
    for i in range(x):
        for j in range(y):
            tmp = format(hidden[i,j], '#010b')
            hidden_array[3*i,j] = tmp[2:4]
            hidden_array[3*i+1,j] = tmp[4:6]
            hidden_array[3*i+2,j] = tmp[6:8]

    x,y = np.shape(toppic)
    newtop = np.zeros_like(toppic)
    for i in range(x):
        for j in range(y):
            tmp = format(toppic[i,j], '#010b')
            first = '0b'
            mid = tmp[2:8]
            third = '00'
            string = first + mid + third
            result = int(string,2)
            newtop[i,j]=result

    x,y = np.shape(hidden_array)
    for i in range(x):
        for j in range(y):
            tmp = format(newtop[i,j], '#010b')
            first = '0b'
            mid = tmp[2:8]
            third = str(int(hidden_array[i,j]))
            if len(third)==1:
                third = '0'+third
            string = first + mid + third
            result = int(string,2)
            newtop[i,j]=result

    return newtop

def find_picture(IM):
    x,y = np.shape(IM)

    new_arr = np.zeros_like(IM)
    for i in range(x):
        for j in range(y):
            tmp = format(IM[i,j], '#010b')
            hidden = mid = tmp[8:]
            new_arr[i,j] = hidden

    new_IM = np.zeros((x,y))
    for i in range(int(x/3)):
        for j in range(j):
            tmp = '0b'+str(new_arr[3*i,j])+str(new_arr[3*i+1,j])+str(new_arr[3*i+1,j])+'00'
            result = int(tmp,2)
            new_IM[i,j] = result

    return new_IM


def test_martin(IM):
    x,y = np.shape(IM)
    newIM = np.zeros_like(IM)
    for i in range(x):
        for j in range(y):
            tmp = format(IM[i,j], '#010b')
            first = '0b'
            mid = tmp[2:8]
            third = '00'
            string = first + mid + third
            result = int(string,2)
            newIM[i,j]=result

    return newIM

def hide_martin():
    toppic = cv2.imread('hobby_pics/kermit.JPEG',0)

    hidden = np.transpose(cv2.imread('hobby_pics/martin.JPG',0))
    hideshape = np.shape(hidden)
    hidden = hidden[int(hideshape[0]/3):int(2*hideshape[0]/3),0:29*int(hideshape[1]/30)]

    newtop = hide_picture(toppic,hidden)

    martin = find_picture(newtop).astype(np.uint8)

    real_martin = test_martin(hidden)

    cv2.imshow('image',real_martin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('image',martin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



hide_martin()
