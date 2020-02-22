import cv2
import numpy as np
import os
import sys

#python hw4_align.py in_dir out_dir

#ORB function returns keypoints
def use_orb(img_g):
    num_features = 2000 #EXPERIMENT
    orb = cv2.ORB_create(num_features)
    kp_des = orb.detectAndCompute(img_g, None)
    return kp_des

def getOffset(homog_mat, img1, img2):
    #apply transform on img2 to get points on img1 plane
	UL = np.dot(homog_mat, np.array([0,0,1]))
	BR = np.dot(homog_mat, np.array([img2.shape[1], img2.shape[0], 1]))
	BL = np.dot(homog_mat, np.array([0,img2.shape[0],1]))
	UR = np.dot(homog_mat, np.array([img2.shape[1],0,1]))
    #divide w to get u v coordinates
	UL /= UL[-1]
	BR /= BR[-1]
	BL /= BL[-1]
	UR /= UR[-1]
	xcoords = min([UL[0], BR[0], BL[0], UR[0]])
	ycoords = min([UL[1], BR[1], BL[1], UR[1]])
    #calculate the offsets for canvas
	offset_x = 0
	offset_y = 0
	if xcoords < 0:
		offset_x = abs(xcoords)
	if ycoords < 0:
		offset_y = abs(ycoords)
	return [offset_x, offset_y]

def stichHelper(composite, offset_x, offset_y, img1, img2, outName):
	UL_img2 = (offset_x, offset_y)
	BR_img2 = (offset_x + img2.shape[1], offset_y + img2.shape[0])
	UR_img2 = (offset_x + img2.shape[1], offset_y)
	BL_img2 = (offset_x, offset_y + img2.shape[0])
	# find where img maps to on composite canvas
	UL = np.dot(composite, np.array([0,0,1]))
	BR = np.dot(composite, np.array([img1.shape[1], img1.shape[0], 1]))
	BL = np.dot(composite, np.array([0,img1.shape[0],1]))
	UR = np.dot(composite, np.array([img1.shape[1],0,1]))
    #divide w to get u v coordinates for composite canvas
	UL /= UL[-1]
	BR /= BR[-1]
	BL /= BL[-1]
	UR /= UR[-1]

	# find the new img size
	imgCol = max(UL[0], BR[0], BL[0], UR[0])
	img2Col = max(UL_img2[0], BR_img2[0], UR_img2[0], BL_img2[0])
	canvas_col = int(max(imgCol, img2Col))
	imgRow = max(UL[1], BR[1], BL[1], UR[1])
	img2Row = max(UL_img2[1], BR_img2[1], UR_img2[1], BL_img2[1])
	canvas_row = int(max(imgRow, img2Row))

	# use calculate the warped image
	warpResult = cv2.warpPerspective(img1, composite, (canvas_col, canvas_row),cv2.INTER_LINEAR)

	# create a mask for each image
	canvasImg1 = np.zeros(warpResult.shape)
	canvasImg1[np.where(warpResult > 0)] = 1
	canvasImg2 = np.zeros(warpResult.shape)
	canvasImg2[int(offset_y):img2.shape[0]+int(offset_y),\
		int(offset_x):img2.shape[1]+int(offset_x)] = 1
	newComposite = canvasImg1 + canvasImg2
	mask1 = np.copy(canvasImg1)
	mask1[np.where(newComposite == 2)] = 0.2
	mask2 = np.copy(canvasImg2)
	mask2[np.where(newComposite == 2)] = 0.8

	img1Partial = mask1 * warpResult
	im2Warp = np.zeros(warpResult.shape)
	im2Warp[int(offset_y):img2.shape[0]+int(offset_y), \
		int(offset_x):img2.shape[1]+int(offset_x)] = img2

	img2Partial = mask2 * im2Warp
	finalimg = img1Partial + img2Partial

	# Gaussian blur hopefully blends image
	finalimg = cv2.GaussianBlur(finalimg, (3,3), 2)
	cv2.imwrite(outName+"_stitched.jpg", finalimg)


in_dir = os.listdir(sys.argv[1])
in_img_list = [f for f in in_dir if f.lower().endswith(".jpg")]

f1 = in_img_list[0].split(".")[0]
f2 = in_img_list[1].split(".")[0]
out_dir = sys.argv[2]

name = sys.argv[1].split("\\")
name1 = sys.argv[1].split("/")
if(len(name)<len(name1)):
	name = name1
name=name[-1]

img1 = cv2.imread(os.path.join(sys.argv[1], in_img_list[0]))
img2 = cv2.imread(os.path.join(sys.argv[1], in_img_list[1]))
image1 = cv2.imread(os.path.join(sys.argv[1], in_img_list[0]),0)
image2 = cv2.imread(os.path.join(sys.argv[1], in_img_list[1]),0)

img1_kp, img1_des = use_orb(image1)
img2_kp, img2_des = use_orb(image2)

#using hamming to find the shortest distance of two keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(img1_des,img2_des)
matches = sorted(matches, key = lambda x:x.distance)

#separate the two set of points
points1=list()
points2=list()
kp1=list()
kp2=list()
for i in matches:
    points1.append(img1_kp[i.queryIdx].pt)
    points2.append(img2_kp[i.trainIdx].pt)

#turn into np array
points1 = np.asarray(points1)
points2 = np.asarray(points2)

#sanity check for keypoints so far
img3 = cv2.drawMatches(img1,img1_kp,img2,img2_kp,matches, None,flags=2)
cv2.imwrite(out_dir+"/"+name+"_BF.jpg",img3)

#find fundamental matrix
fundamental_mat, mask = cv2.findFundamentalMat(points2, points1, cv2.FM_RANSAC)
temp1=np.ones(points1.shape[0]).reshape(points1.shape[0],1)
points1_ = np.concatenate((points1,temp1),axis=1)
points2_ = np.concatenate((points2,temp1),axis=1)

#Refine keypoints: make sure abs(a-Fu)<0.08 as threshold
temp2 = points1_@fundamental_mat@points2_.T
temp2 = temp2.diagonal()
fund_idx = np.where(np.abs(temp2)<0.88)
matches = np.asarray(matches)
matches_1 = matches[fund_idx]

# Sanity Check show image with keypoints
img3 = cv2.drawMatches(img1,img1_kp,img2,img2_kp,matches_1, None,flags=2)
cv2.imwrite(out_dir+"/"+name+"_fundamental.jpg",img3)

#collect from fundamental matrix keypoints
homog_p1 = points1[fund_idx]
homog_p2 = points2[fund_idx]

#find the homography
homog_mat, homog_mask = cv2.findHomography(homog_p2,homog_p1,cv2.RANSAC)

#create ones to concatenate
temp3 = np.ones(homog_p1.shape[0]).reshape(homog_p1.shape[0],1)
homog_p1_ = np.concatenate((homog_p1,temp3),axis=1)
homog_p2_ = np.concatenate((homog_p2,temp3),axis=1)

#compares H*Point2 to Point1 using distance formula
p = homog_mat@homog_p2_.T
PP = homog_p1_-p.T
PP_ = np.sum(np.abs(PP)**2,axis=-1)**(1./2)
pre_size = len(PP_)

#threshold using average
averaged_ = np.average(PP_)
averaged_2 = 0.75*averaged_
homog_idx = np.where(PP_<averaged_2)
post_size = len(homog_idx[0])
if post_size/pre_size<0.7 and post_size<150:
    print("Cannot be panoramic")
    sys.exit(0)

#shows homography keypoints
matches_2 = matches_1[homog_idx]
img3 = cv2.drawMatches(img1,img1_kp,img2,img2_kp,matches_2, None,flags=2)
cv2.imwrite(out_dir+"/"+name+"_Homography.jpg",img3)

normImg1 = np.sqrt(homog_mat[-1,0]**2 + homog_mat[-1,1]**2)
homog_matInv = np.linalg.inv(homog_mat)
normImg2 = np.sqrt(homog_matInv[-1,0]**2 + homog_matInv[-1,1]**2)
#decide which image to map on to which
offset_x1, offset_y1 = getOffset(homog_mat, img2, img1)
offsetMat = np.array([[1,0,offset_x1], [0,1,offset_y1], [0,0,1]])
composite = np.dot(offsetMat, homog_mat)
stichHelper(composite, offset_x1, offset_y1, img2, img1, out_dir+"/"+name)