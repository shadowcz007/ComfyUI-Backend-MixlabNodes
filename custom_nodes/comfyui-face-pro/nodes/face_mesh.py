import mediapipe as mp
import cv2
from PIL import Image, ImageOps,ImageFilter,ImageEnhance,ImageDraw,ImageSequence, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def generate_mask(points, height, width,mask=None):
    # 创建一个空白画布
    if mask is None:
        mask = np.zeros((height, width), dtype=np.uint8)

    # 连接给定的点
    for i in range(len(points) - 1):
        cv2.line(mask, points[i], points[i + 1], (255), 3)

    # 填充内部区域为白色
    cv2.fillPoly(mask, [np.array(points)], (255))

    return mask

radius = 2
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
yellow = (0,255,255)


left_eyebrow = [70,63,105,66,107,55,65,52,53,46]
right_eyebrow = [300,293,334,296,336,285,295,282,283,276]
left_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
right_eye = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
inner_lip = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
outer_lip = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
face_boundary = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
left_iris = [468,469,470,471,472]
right_iris = [473,474,475,476,477]
nose = [64,4,294]



class FaceMeshParse:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "debug": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE","FACEMASH_",)
    RETURN_NAMES = ("debug_image","facemesh_result",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Face"

    def run(self,image,debug):
        faceMesh=mp.solutions.face_mesh.FaceMesh(False,3,True,0.5,0.5)
        
        frame=tensor2pil(image)

        width, height = frame.size

        image_np = np.array(frame)

        #将NumPy数组转换为cv2图像
        frameRGB = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        frameRGB = cv2.cvtColor(frameRGB,cv2.COLOR_BGR2RGB)

        if debug:
            rs=4
            resized_image = cv2.resize(frameRGB, (width*rs, height*rs))

        results = faceMesh.process(frameRGB)
        faces=[]
        if results.multi_face_landmarks != None:
            for faceLandmarks in results.multi_face_landmarks:
                myFaceLandmarks=[]
                for i in range(len(faceLandmarks.landmark)):
                    lm=faceLandmarks.landmark[i]
                    x=int(lm.x*width)
                    y=int(lm.y*height)

                    if debug:
                        #debug
                        cv2.circle(resized_image,(x*rs,y*rs),radius,yellow,-1)
                        text_position = (x*rs + radius, y*rs)
                        text = str(i)
                        # 绘制文字编号
                        cv2.putText(resized_image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
                    
                    # print(i) 从0 开始编号
                    myFaceLandmarks.append((x,y,int(lm.z*width)))
                faces.append(myFaceLandmarks)            

        image_d=None
        if debug:
            image_pil = Image.fromarray(resized_image)
            image_d=pil2tensor(image_pil)

        facemesh_result={
            "width":width,
            "height":height,
            "faces":faces
        }
        return (image_d,facemesh_result,)



class FaceMeshResults:
    @classmethod
    def INPUT_TYPES(cls):
        opts=[x.strip() for x in '''left_eyebrow
right_eyebrow
left_eye
right_eye
inner_lip
outer_lip
face_boundary
left_iris
right_iris
nose'''.split('\n') if x.strip()]
        # print(opts)
        return {
            "required": {
                "facemesh_result": ("FACEMASH_", {}),
                "landmarks": ("STRING", 
                         {
                            "multiline": True, 
                            "default": '64 4 294',
                            "dynamicPrompts": False
                          }),
                "option":(opts,  {"default": "nose"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ( "mask",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Face"

    def run(self,facemesh_result,landmarks,option):
      
        width=facemesh_result['width']
        height=facemesh_result['height']
        faces=facemesh_result['faces']
      
        
        left_f=[int(x) for x in landmarks.strip().split()]


        mask=None

        for face in faces:
            
            left_fs=left_f

            for indx in range(0,len(face)):# Total Landmarks = 141
                # print(face[indx])
                if indx in left_f:
                    # cv2.circle(frame,(face[indx][0],face[indx][1]),radius,yellow,-1)
                    index = left_f.index(indx)
                    left_fs[index]= (face[indx][0],face[indx][1])
    
            left_fs = [x for x in left_fs if x] 

            # 输出背景为黑色的掩码图像
            mask = generate_mask(left_fs, height, width,mask)
        
        if mask is None:
            mask=np.zeros((height, width), dtype=np.uint8)
            img_pil = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            img_gray = img_pil.convert('L')
            im=pil2tensor(img_gray)
            return (im,)

        # cv2.imwrite('mask.png', mask)
        img_pil = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

        # 将PIL图片转为灰度图
        img_gray = img_pil.convert('L')

        im=pil2tensor(img_gray)

        return (im,)



    