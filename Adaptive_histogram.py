import cv2
import os
import matplotlib.pyplot as plt
import math 
import numpy as np

def show(img):
    cv2.imshow('Output', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def showv(img):
    cv2.imshow('Output', img)
    cv2.waitKey(100)
    # cv2.destroyAllWindows()

def output():
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	vout = cv2.VideoWriter("Output1.mp4", fourcc, 1, (1224,1110))
	for i in range(24):
		if i < 10:
			imgi = cv2.imread("adaptive_hist_data/000000000"+str(i)+".png")
		else :
			imgi = cv2.imread("adaptive_hist_data/00000000"+str(i)+".png")
		imgo = cv2.imread("Output/Output"+str(i)+".png")
		imga = cv2.imread("Output/Output_A"+str(i)+".png")
		res = np.vstack((imgi,imgo,imga))
		showv(res)
		# print(res.shape)
		F = cv2.resize(res, (1224,1110), interpolation = cv2.INTER_AREA)
		vout.write(F)
	vout.release()

def main():
	for i in range(24):
		if i < 10:
			img = cv2.imread("adaptive_hist_data/000000000"+str(i)+".png")
		else :
			img = cv2.imread("adaptive_hist_data/00000000"+str(i)+".png")
		imgr = equalize(img,0)
		cv2.imwrite("Output/Output"+str(i)+".png",imgr)
		imgr =adaptive(img)
		# imgr = equalize(imgr)
		cv2.imwrite("Output/Output_A"+str(i)+".png",imgr)
		print(i)
		
	

def adaptive(img):
	h,w,c = img.shape
	si,sj = round(h//8),round(w//8)
	i = 0
	imgr = img.copy()
	while i < h:
		j = 0
		while j < w:
			if j+sj<w and i+si<h:
				img_crop = img[i:i+si,j:j+sj,:]
				imgt =  equalize(img_crop)
				imgr[i:i+si,j:j+sj,:] = imgt
			elif i+si<h:
				img_crop = img[i:i+si,j:w,:]
				imgr[i:i+si,j:w,:] = equalize(img_crop)
			elif j+sj<w:
				img_crop = img[i:h,j:j+sj,:]
				imgr[i:h,j:j+sj,:] = equalize(img_crop)
			else :
				img_crop = img[i:h,j:w,:]
				imgr[i:h,j:w,:] = equalize(img_crop)
			j = j+sj
		i = i+si
		#print(i)
	return imgr


def createhist(image) :
	hist = np.zeros(256).astype(int)
	m, n = np.unique(image.flatten(), return_counts = True, axis=0)
	hist[m] = n
	return hist

def contrast_limiting(hist, clip = 40):
	count = 0
	length = len(hist)
	for j,i in enumerate(hist):
		if i > clip:
			count += i - clip
			hist[j] = clip
	hist = (np.array(hist) + (count//length))
	return hist

def equalize(img,ss=1):
	h,w,c = img.shape
	he = (255/(h*w))

	rh = np.zeros(256)
	bh = np.zeros(256)
	gh = np.zeros(256)
	eqr = np.zeros(256)
	eqg = np.zeros(256)
	eqb = np.zeros(256)

	rh = createhist(img[:,:,2])
	bh = createhist(img[:,:,0])
	gh = createhist(img[:,:,1])

	if ss == 1: 
		rh = contrast_limiting(rh)
		bh = contrast_limiting(bh)
		gh = contrast_limiting(gh)

	for i in range(256):
		for j in range(i+1):
			eqr[i] += rh[j] * he 
			eqg[i] += gh[j] * he 
			eqb[i] += bh[j] * he 
		eqr[i] = round(eqr[i])
		eqg[i] = round(eqg[i])
		eqb[i] = round(eqb[i])
	
	imgr =img.copy()

	for i in range(h):
		for j in range(w):
			b_value,g_value,r_value = img[i,j]
			imgr[i,j,0] = eqb[b_value]
			imgr[i,j,1] = eqg[g_value]
			imgr[i,j,2] = eqr[r_value]
	res = np.vstack((img,imgr))
	#show(res)
	return imgr

if __name__ == '__main__':
	main()
	output()