#python version 3.9.7
#opncv-python version = 4.5.4.60
#mediapipe version = 0.8.9.1

import cv2
import math
import numpy as np

# ACTUAL MEDIAPIPE FACE LANDMARKS
# Left Eyebrow = [70,63,105,66,107,55,65,52,53,46]
# Right Eyebrow = [300,293,334,296,336,285,295,282,283,276]
# Left Eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
# Right Eye = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
# Inner Lip = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
# Outer Lip = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
# Face Boundary = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
# Left iris = [468,469,470,471,472]
# Right iris = [473,474,475,476,477]
# Nose = [64,4,294]

# SIMPLIFIED FACE LANDMARKS AFTER SEQUENCING
# Left Eyebrow (0->9)
# right Eyebrow (10->19)
# Left Eye (20->35)
# Right Eye (36->51)
# iner Lip (52->71)
# outer Lip (72->91)
# face boundary (92->127)
# Left iris (128->132)
# Right iris (133->137)
# Nose (138 -> 140)

class mpFace:
    import mediapipe as mp
    import cv2
    def __init__(self,width=640,height=480):
        self.findFace = self.mp.solutions.face_detection.FaceDetection()
        self.faceMesh = self.mp.solutions.face_mesh.FaceMesh(False,3,True,0.5,0.5)#(staticFrame,number of faces,True for extra iris landmarks,trackingParameter,findingParameter)
        self.width = width
        self.height = height
    def faceLandmarks(self,frame):#Full Face Landmarks
        frameRGB = self.cv2.cvtColor(frame,self.cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(frameRGB)
        myFacesLandmarks=[]
        if results.multi_face_landmarks != None:
            for faceLandmarks in results.multi_face_landmarks:
                myFaceLandmarks=[]
                for i in range(len(faceLandmarks.landmark)):
                    lm=faceLandmarks.landmark[i]
                    # print(i) 从0 开始编号
                    myFaceLandmarks.append((int(lm.x*self.width),int(lm.y*self.height),int(lm.z*self.width)))
                myFacesLandmarks.append(myFaceLandmarks)            
        return myFacesLandmarks

    def faceLandmarksSimplified(self,frame):#essential face landmarks(left eyebrow,righteyebrow,left eye,right eye,inner lips,outer lips,face outline,left iris and right iris)
        frameRGB = self.cv2.cvtColor(frame,self.cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(frameRGB)
        #sequenced indexes of required landmarks from original landmarks
        points = [70,63,105,66,107,55,65,52,53,46,300,293,334,296,336,285,295,282,283,276,33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249,78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,468,469,470,471,472,473,474,475,476,477,64,4,294]
        myFaceLandmarksSimplified=[]
        myFacesLandmarksSimplified=[]
        myFaceLandmarksRearranged = [0]*len(points)
        myIndex=[]
    
        if results.multi_face_landmarks != None:
            for faceLandmarks in results.multi_face_landmarks:
                for lm,indx in zip(faceLandmarks.landmark,range(len(faceLandmarks.landmark))):
                    if indx in points:#only collect required landmarks
                        myFaceLandmarksSimplified.append((int(lm.x*self.width),int(lm.y*self.height),int(lm.z*self.width)))
                        myIndex.append(points.index(indx))#for rearranging the points, collect sequenced index
                    for i,indx in zip(range(len(points)),myIndex):
                        myFaceLandmarksRearranged[indx] = myFaceLandmarksSimplified[i]#rearranging according to sequenced index

                myFacesLandmarksSimplified.append(myFaceLandmarksRearranged)          
        return myFacesLandmarksSimplified


def generate_mask(points, height, width):
    # 创建一个空白画布
    mask = np.zeros((height, width), dtype=np.uint8)

    # 连接给定的点
    for i in range(len(points) - 1):
        cv2.line(mask, points[i], points[i + 1], (255), 3)

    # 填充内部区域为白色
    cv2.fillPoly(mask, [np.array(points)], (255))

    return mask
# # 定义需要连接的点坐标
# points = [(100, 100), (200, 200), (300, 200), (400, 300)]
# # 定义画布的尺寸
# height, width = 500, 500
# # 生成掩码图像
# mask = generate_mask(points, height, width)
# # 输出背景为黑色的掩码图像
# cv2.imwrite('mask.png', mask)



frame=cv2.imread('t.png')
h,w,_=frame.shape

radius =8
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
yellow = (0,255,255)

faceLm = mpFace(width=w,height=h)

#For connecting range of dots
def connectPoints(indx1,indx2):
    for i in range(indx1,indx2):
        if i==(indx2-1):
            cv2.line(frame,(face[i][0],face[i][1]),(face[indx1][0],face[indx1][1]),green,1)
            break
        cv2.line(frame,(face[i][0],face[i][1]),(face[i+1][0],face[i+1][1]),green,1)
#Finding length between two points
def findRadius(pt1,pt2):
    x1,y1 = (pt1[0],pt1[1])
    x2,y2 = (pt2[0],pt2[1])
    radius = math.sqrt(((y2-y1)*(y2-y1))+((x2-x1)*(x2-x1)))
    return radius

faces = faceLm.faceLandmarks(frame)

left_f=[int(x) for x in '123 117 118 101 36 206 207 187'.split()]


for face in faces:
    
    left_fs=left_f

    for indx in range(0,len(face)):# Total Landmarks = 141
        # print(face[indx])
        if indx in left_f:
            cv2.circle(frame,(face[indx][0],face[indx][1]),radius,yellow,-1)
            index = left_f.index(indx)
            left_fs[index]= (face[indx][0],face[indx][1])
    print(left_fs)
    left_fs = [x for x in left_fs if x] 

    mask = generate_mask(left_fs, h, w)
    # 输出背景为黑色的掩码图像
    cv2.imwrite('mask.png', mask)

    # connectPoints(0,10)#Left Eyebrow (0->9)
    # connectPoints(10,20)#right Eyebrow (10->19)
    # connectPoints(20,36)#Left Eye (20->35)
    # connectPoints(36,52)#Right Eye (36->51)
    # connectPoints(52,72)#iner Lip (52->71)
    # connectPoints(72,92)#outer Lip (72->91)
    # connectPoints(92,128)#face boundary (92->127)

    # cv2.circle(frame,(face[128][0],face[128][1]),3,yellow,-1)#left pupil (centre->128,adjacent->129)
    # rl=findRadius(face[128],face[129])#left iris radius
    # cv2.circle(frame,(face[128][0],face[128][1]),int(rl),blue,1)

    # cv2.circle(frame,(face[133][0],face[133][1]),3,yellow,-1)#right pupil (centre->133,adjacent->134)
    # rr=findRadius(face[133],face[134])#right iris radius
    # cv2.circle(frame,(face[133][0],face[133][1]),int(rr),blue,1)


cv2.imwrite('res.png',frame)