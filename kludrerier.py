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

def paint_filter(find):
    if not find:
        IM = cv2.imread('home_exam_pictures/F3.png',0)

        four = np.log(np.fft.fftshift(np.fft.fft2(IM)))
        remax = np.max(four.real)
        immax = np.max(four.imag)
        refour = four.real*255/remax
        imfour = four.imag*255/immax

        print(four)

        refour[0,0]=remax
        imfour[0,0]=immax

        cv2.imwrite('hobby_pics/four_real.PNG',refour)
        cv2.imwrite('hobby_pics/four_imag.PNG',imfour)

    if find:
        newrefour = cv2.imread('hobby_pics/four_real.PNG',0)
        remax = newrefour[0,0]
        newrefour = newrefour*remax/255

        newimfour = cv2.imread('hobby_pics/four_imag.PNG',0)
        immax = newimfour[0,0]
        newimfour = newimfour*immax/255

        newfour = np.vectorize(complex)(newrefour,newimfour)

        print(np.exp(newfour))

        newIM1 = np.abs(np.fft.ifft2(np.exp(newfour)))

        #newIM1 = gamma_transform(10,newIM1,0.0001)

        plt.imshow(newIM1)
        plt.show()


def hide_martin():
    toppic = cv2.imread('hobby_pics/abstract.jpg')

    hidden = cv2.imread('hobby_pics/martin.JPG')

    if np.shape(toppic)[0]==3:
        toppic = reshape_image(toppic)
    if np.shape(hidden)[0]==3:
        hidden = reshape_image(hidden)

    newtop = hide_picture(toppic,hidden)

    cv2.imwrite('hobby_pics/hidden_martin.png',newtop)


def find_martin():
    hidden_martin = cv2.imread('hobby_pics/hidden_martin.png')

    martin = find_picture(hidden_martin)

    cv2.imwrite('hobby_pics/found_martin.png',martin)

#hide_martin()
#find_martin()
paint_filter(False)
paint_filter(True)
