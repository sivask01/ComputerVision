import numpy as np
import numpy
from PIL import Image
import random
import cv2
sift = cv2.xfeatures2d.SIFT_create()
from matplotlib import pyplot as plt 

img1 = Image.open('uttower_left.JPG')
im1 = np.array(img1)



img2 = Image.open('uttower_right.JPG')
im2 = np.array(img2)

threshold = 0.7

def Sift(im):
    return sift.detectAndCompute(im,None)

def find_dist(f1,f):
    return np.sqrt(np.sum(np.square(np.subtract(f1,f)),axis=1))


def find_dist2(f1,f):
    return np.sqrt(np.sum(np.square(np.subtract(f1,f))))

def Match(p1,p2,f1,f2):
    print(len(f1))
    new_p1 = [[],[]]
    new_p2 = [[],[]]
    i=0
    for f in f2:
        dis = find_dist(f1,f)
        sort_dis = np.argsort(dis)[:2] #gives sorted indices
        if find_dist2(f1[sort_dis[0]],f) < threshold*find_dist2(f1[sort_dis[1]],f):
            new_p2[0].append(p2[i])
            new_p1[0].append(p1[sort_dis[0]])
            new_p1[1].append(f1[sort_dis[0]])
            new_p2[1].append(f)
        i=i+1
    return (new_p1,new_p2)

l=np.array([[1,2,3],[4,5,6]])

p1,f1 = Sift(im1)
p2,f2 = Sift(im2)

(new_p1,new_p2) = Match(p1,p2,f1,f2)

print(new_p1[0])

(x1,y1) = im1.shape[:2]
(x2,y2) = im2.shape[:2]

combined = np.zeros((max(x1,x2),y1+y2,3),dtype=np.uint8)
combined[:x1,:y1] = im1
combined[:x2,y1:] = im2
for i in range(len(new_p1[0])):
    ptA = (int(new_p1[0][i].pt[0]),int(new_p1[0][i].pt[1]))
    ptB = (int(new_p2[0][i].pt[0])+y1,int(new_p2[0][i].pt[1]))
    cv2.line(combined,ptA,ptB,(0,0,0),1)

matched_im1 = np.array([[i.pt[0] , i.pt[1],1] for i in new_p1[0]])
matched_im2 = np.array([[i.pt[0] , i.pt[1],1] for i in new_p2[0]])

if len(new_p1[0])>4:
    best_h = np.array([])
    count = -1
    
    for i in range(len(new_p1[0])):
        four_pts = np.array(random.sample(list(range(len(new_p1[0]))),4))
        print(four_pts)
        x1_pts = [] ; x2_pts=[]
        for j in four_pts:
            x1_pts.append( [ new_p1[0][j].pt[0] , new_p1[0][j].pt[1] , 1 ] )
            x2_pts.append( [ new_p2[0][j].pt[0] , new_p2[0][j].pt[1] , 1 ] )
        x1_pts,x2_pts = np.array(x2_pts),np.array(x1_pts)
        corresponding_points = x1_pts.shape[0]

        A = np.zeros((2*corresponding_points,9))

        for i in range(corresponding_points):
            A[2*i] = [-x1_pts[i][0] , -x1_pts[i][1] , -1 , 0 , 0 , 0 , x1_pts[i][0]*x2_pts[i][0] , x1_pts[i][1]*x2_pts[i][0] , x2_pts[i][0] ]
            A[2*i +1] = [ 0 , 0 , 0 , -x1_pts[i][0] , -x1_pts[i][1] , -1 , x1_pts[i][0]*x2_pts[i][1] , x1_pts[i][1]*x2_pts[i][1] , x2_pts[i][1] ]
        
        U, S, V = np.linalg.svd(A)
        H = V[8].reshape((3,3))
        h = H/H[2,2]
        new_x = np.matmul(h,matched_im2.T)
        error = np.linalg.norm(np.subtract(matched_im1 , new_x.T),axis=1) < 3
        if count<np.count_nonzero(error):
                count = np.count_nonzero(error)
                best_h = h
print(count)
result = cv2.warpPerspective(im2,best_h,
    (y1+y2, x1))
result[:x1,:y2] = im1
plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()
plt.imshow(combined)
plt.show()
plt.imshow(result)
plt.show()