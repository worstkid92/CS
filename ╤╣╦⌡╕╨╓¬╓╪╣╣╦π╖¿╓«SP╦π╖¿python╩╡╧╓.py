#coding:utf-8

#导入集成库
import math
import pywt
# 导入所需的第三方库文件
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包


#读取图像，并变成numpy类型的 array
im = np.array(Image.open('C:\Users\kaka1\Desktop\New.bmp'))#图片大小256*256

#生成高斯随机测量矩阵
sampleRate=0.7  #采样率
Phi=np.random.randn(int(256*sampleRate),256)   #高斯随机矩阵     
print(Phi.shape)

#生成稀疏基DCT矩阵
mat_dct_1d=np.zeros((256,256))
v=range(256)
for k in range(0,256):  
    dct_1d=np.cos(np.dot(v,k*math.pi/256))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)   #256*256
print(mat_dct_1d.shape)



#随机测量
img_cs_1d=np.dot(Phi,im)   #179*256   观测矩阵
print('img_cs_1d\n',img_cs_1d.shape)

#SP算法函数
def cs_sp(y,D):     
    K=math.floor(y.shape[0]/3)  
    pos_last=np.array([],dtype=np.int64)
    result=np.zeros((256))

    product=np.fabs(np.dot(D.T,y))
    pos_temp=product.argsort() 
    pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
    pos_current=pos_temp[0:K]#初始化索引集 对应初始化步骤1
    residual_current=y-np.dot(D[:,pos_current],np.dot(np.linalg.pinv(D[:,pos_current]),y))#初始化残差 对应初始化步骤2

    while True:  #迭代次数
        product=np.fabs(np.dot(D.T,residual_current))       
        pos_temp=np.argsort(product)
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        pos=np.union1d(pos_current,pos_temp[0:K])#对应步骤1     
        pos_temp=np.argsort(np.fabs(np.dot(np.linalg.pinv(D[:,pos]),y)))#对应步骤2  
        pos_temp=pos_temp[::-1]
        pos_last=pos_temp[0:K]#对应步骤3    
        residual_last=y-np.dot(D[:,pos_last],np.dot(np.linalg.pinv(D[:,pos_last]),y))#更新残差 #对应步骤4
        if np.linalg.norm(residual_last)>=np.linalg.norm(residual_current): #对应步骤5  
            pos_last=pos_current
            break
        residual_current=residual_last
        pos_current=pos_last
    result[pos_last[0:K]]=np.dot(np.linalg.pinv(D[:,pos_last[0:K]]),y) #对应输出步骤  
    return  result

#重建
sparse_rec_1d=np.zeros((256,256))   # 初始化稀疏系数矩阵    
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
for i in range(256):
    print('正在重建第',i,'列。。。')
    column_rec=cs_sp(img_cs_1d[:,i],Theta_1d)  #利用SP算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec;        
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

#显示重建后的图片
image2=Image.fromarray(img_rec)
image2.show()
