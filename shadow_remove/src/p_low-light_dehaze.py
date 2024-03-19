import cv2;
import math;
import numpy as np;
from time import time
# import time
import os
import random

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    # gray = np.float64(gray)/255;
    gray = gray/255
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def dehaze(src):
    I = src/255
    time0 = time()
    dark = DarkChannel(I,15)
    # time1 = time()
    # print("DarkChannel time: %.2f" % (time1-time0))

    A = AtmLight(I,dark)
    # time2 = time()
    # print("Atmlight time: %.2f" % (time2-time1))

    te = TransmissionEstimate(I,A,15)
    # time3 = time()
    # print("Transmission estimate time: %.2f" % (time3-time2))

    t = TransmissionRefine(src,te)
    # time4 = time()
    # print("Transmission Refine time: %.2f" % (time4-time3))

    J = Recover(I,t,A,0.1)
    # time5 = time()
    # print("Recovery time: %.2f" % (time5-time4))

    
    J = J*255
    J[J>255] = 255
    J[J<0] = 0
    return J.astype(np.uint8)

def lowlight_enhance(src):
    # start = time()
    src = 255-src
    src = dehaze(src)
    src = 255-src
    # print(f"Enhance time: %.2f" % (time()-start))
    return src

def median_blur(frame):
    return cv2.medianBlur(frame,3)

def gaussian_blur(frame):
    return cv2.GaussianBlur(frame,(5,5),0)

def sharpen(frame):
    return 2*frame - gaussian_blur(frame)

def he(frame):
    '''
    - Histogram Equalize
    '''
    frame_YCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    channels = cv2.split(frame_YCC)
    channels[0] = cv2.equalizeHist(channels[0])
    frame_YCC = cv2.merge((channels[0], channels[1], channels[2]))
    return cv2.cvtColor(frame_YCC, cv2.COLOR_YUV2BGR)

def lowlight_enhance_me(img, scale = 1):
    '''
    - Lowlight enhance
    '''
    #Downscale the image for better performance
    if scale == 1:
        return lowlight_enhance(img)

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = lowlight_enhance(img)

    scale = 1/scale #Scale back
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return result.astype('uint8')

def dehaze_me( img, scale = 1):
    '''
    - Lowlight enhance
    '''
    #Downscale the image for better performance
    if scale == 1:
        return dehaze(img)

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = dehaze(img)

    scale = 1/scale #Scale back
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return result

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = np.floor(255*(1-rdn))
            elif rdn > thres:
                output[i][j] = np.floor(255*rdn)
            else:
                output[i][j] = image[i][j]
    return output

if __name__ == '__main__':
    # img = cv2.imread('../sample/frame_8.jpg')
    # img = dehaze(img)
    # img = lowlight_enhance(img)
    # cv2.imwrite("../res/2lowlight_result.png",img)


    sample_dir = '../sample/'


    result_dir = '../res/delight_iamges/'

    total_time = 0
    
    for filename in os.listdir(sample_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):

            img = cv2.imread(os.path.join(sample_dir, filename))
            start_time = time()
            
            img = dehaze(img)
            img = lowlight_enhance(img)
            # g_blur = cv2.GaussianBlur(img, (9,9), 0)
            # img = cv2.addWeighted(img,2.5, g_blur, -1.5, -20)
            end_time = time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            
            cv2.imwrite(os.path.join(result_dir, filename), img)
            
    average_time = total_time / len(os.listdir(sample_dir))
    print(f"Thời gian trung bình: {average_time} giây")