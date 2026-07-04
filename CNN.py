from PIL import Image 
import math
import json
import random
import os

dataset=[]

def fill_dataset(mode):
    for label in range(10):
        if mode=='train':
            folder=f'train/{label}'
        else:
            folder=f'test/{label}'
        for image in os.listdir(folder):
            img_path=os.path.join(folder,image)
            Y=[0]*10
            Y[label]=1
            dataset.append((img_path,Y))
    random.shuffle(dataset)
    
    return dataset
    
def img_to_grayscale(img_path):
    img=Image.open(img_path)
    img=img.convert("L")
    img=img.resize((28,28))
    pixels = list(img.getdata())
    pixels=[[pixels[28*i+j]/255 for j in range(28)] for i in range(28)]
    
    return pixels

def Softmax(l):
    s=0
    
    for i in range(len(l)):
        l[i]=math.exp(l[i])
        s+=l[i]
    l=[p/s for p in l]
    
    return l

def Padding(Img:list):
    p_Img=[[0]*(len(Img)+2)]
    for row in Img:
        prow= [0]
        prow.extend(row)
        prow.append(0)
        p_Img.append(prow)
    p_Img.append([0]*(len(Img)+2))
    return p_Img

def ConvND(N,Img):
    f=open('CNN_Filters.json','r')
    Filters=json.load(f)[N-1]
    f.close()
    
    l=28//N
    Feature_Maps=[[[0]*l for I in range(l)] for n in range(16)]
    depth=1
    if N!=1:
        depth=16
    
    Pad_Img=[]
    for n1 in range(depth):
        if N!=1:
            Pad_Img.append(Padding(Img[n1]))
        else:
            Pad_Img=Padding(Img)
    for n in range(16):
        f=Filters[n]
        for I in range(28//N):
            for J in range(28//N):
                s=0
                for n1 in range(depth):
                    for i in range(3):
                        for j in range(3):
                            if N==1:
                                s+=f[i][j]*Pad_Img[I+i][J+j]
                            else:
                                s+=f[n1][i][j]*Pad_Img[n1][I+i][J+j]           
                Feature_Maps[n][I][J]=s
    
    return Feature_Maps                                    

def ReLU(f_maps):
    for n in range(16):
        for i in range(len(f_maps[n])):
            for j in range(len(f_maps[n])):  #Here, len(f_maps[n])=len(f_maps[n][i])
                if f_maps[n][i][j]<0:
                    f_maps[n][i][j]=0
    return f_maps

def Pooling(f_maps):
    up_fmaps=[]
    for n in range(16):
        fmap=f_maps[n]
        up_fmaps.append([])
        for i in range(0,len(fmap),2):
            up_fmaps[n].append([])
            for j in range(0,len(fmap),2):
                a=max([fmap[i][j],fmap[i][j+1],fmap[i+1][j],fmap[i+1][j+1]])
                up_fmaps[n][i//2].append(a)
    
    return up_fmaps

def Flatten(f_maps):
    Neurons=[]
    for i in f_maps:
        for j in i:
            for k in j:
                Neurons.append([k])
    
    return Neurons

def dense(Neurons):
    Output=[]
    
    f=open('CNN_weights.json','r')
    weights=json.load(f)
    f.close()
    
    # f=open('CNN_bias.json','r')
    # bias=json.load(f)
    # f.close()
    
    for i in range(10):
        a=0
        for j in range(784):
            a+=weights[i][j]*Neurons[j][0]
        # a-=bias[i][0]
        Output.append(a)
    
    Output=Softmax(Output)
    
    return Output

def forward(img):
    Img=img_to_grayscale(img)
    fmaps=ConvND(1,Img)
    fmaps=ReLU(fmaps)
    up_fmaps1=Pooling(fmaps)
    up_fmaps2=ConvND(2,up_fmaps1)
    up_fmaps2=ReLU(up_fmaps2)
    up_fmaps3=Pooling(up_fmaps2)
    Neurons=Flatten(up_fmaps3)
    Output=dense(Neurons)
    up_fmaps=[up_fmaps3,up_fmaps2,up_fmaps1]

    return Output,Neurons,up_fmaps,fmaps,Img 
    

def Backpropagation(Img,f_maps,up_fmaps,Neurons,Output,Y):
    f=open('CNN_weights.json','r')
    weights=json.load(f)
    f.close()
    
    f=open('CNN_Filters.json','r')    
    Filters=json.load(f)
    Filters1D=Filters[0]
    Filters2D=Filters[1]
    f.close()
    
    def Back_Dense():
        dell_dense=[]
        Gradient=[]
        for i in range(10):
            Gradient.append([])
            for j in range(784):
                dell_dense.append((Output[i]-Y[i])*weights[i][j])
                Gradient[i].append((Output[i]-Y[i])*Neurons[j][0])

        return Gradient, dell_dense
    
    def Back_Flatten(dell_Dense):
        dell_Flatten=[]
        for i in range(16):
            dell_Flatten.append([])
            for j in range(7):
                dell_Flatten[i].append([])
                for k in range(7):
                    idx=49*i+7*j+k
                    dell_Flatten[i][j].append(dell_Dense[idx])
        
        return dell_Flatten
    
    def Back_Pooling(up_fmapsBP,f_mapsBP,dell):   # Also, Back_ReLU included.
        dell_Pool=[]
        
        for n in range(16):
            f_map=f_mapsBP[n]
            dell_Pool.append([])
            l=len(up_fmapsBP[n])
            for i in range(l):
                I=2*i
                dell_Pool[n].extend([[],[]])
                for j in range(l):
                    J=2*j
                    dell_Pool[n][I].extend([0,0])
                    dell_Pool[n][I+1].extend([0,0])
                    list1=[f_map[I][J],f_map[I+1][J],f_map[I][J+1],f_map[I+1][J+1]]
                    list2=[dell_Pool[n][I][J],dell_Pool[n][I+1][J],dell_Pool[n][I][J+1],dell_Pool[n][I+1][J+1]]
                    for k in range(4):
                        if list1[k]==up_fmapsBP[n][i][j]:
                            list2[k]=dell[n][i][j]
        
        return dell_Pool
     
    def Back_ConvND(N,f_mapsC, dell_ReLU):
        #First for 2D;
        depth=16
        if N==1:
            depth=1
        l= 28//N
        Gradients=[]
        
        def Crop(fmaps,I,J,N):
            Cfmaps=[f[J:J+l] for f in fmaps[I:I+l]]
            return Cfmaps
        
        for filter_no in range(16):
            Gradient=[]
            for n in range(depth):
                if N!=1:
                    Gradient.append([])
                for I in range(3):
                    if N!=1:
                        Gradient[n].append([])
                    else:
                        Gradient.append([])
                    for J in range(3):
                        if N!=1:
                            fmaps=Crop(f_mapsC[n],I,J,N)
                        else:
                            fmaps=Crop(f_mapsC,I,J,N)
                        s=0
                        for i in range(l):
                            for j in range(l):
                                s+=fmaps[i][j]*dell_ReLU[filter_no][i][j]
                        if N!=1:
                            Gradient[n][I].append(s)
                        else:
                            Gradient[I].append(s)
            Gradients.append(Gradient)

        if N!=1:
            dell_Conv=[]
            for n in range(16):
                dell_Conv.append([])
                for I in range(l):
                    dell_Conv[n].append([])
                    for J in range(l):
                        dell_Conv[n][I].append(0)
                        for fno in range(16):
                            for i in range(3):
                                for j in range(3):
                                    dell_val=Filters2D[fno][n][i][j]*dell_ReLU[n][i][j]
                                    if I==0:
                                        if J==0:
                                            if i in [0,1] and j in [0,1]:
                                                dell_Conv[n][I][J]+=dell_val
                                        
                                        elif J==l-1:
                                            if i in [0,1] and j in [1,2]:
                                                dell_Conv[n][I][J]+=dell_val
                                        
                                        else:
                                            if i in [0,1]:
                                                dell_Conv[n][I][J]+=dell_val
                                    
                                    elif I==l-1:
                                        if J==0:
                                            if i!=0 and j!=2:
                                                dell_Conv[n][I][J]+=dell_val
                                        
                                        elif J==l-1:
                                            if i in [1,2] and j in [1,2]:
                                                dell_Conv[n][I][J]+=dell_val
                                        
                                        else:
                                            if i!=0:
                                                dell_Conv[n][I][J]+=dell_val
                                    
                                    else:
                                        if J==0:
                                            if j!=2:
                                                dell_Conv[n][I][J]+=dell_val
                                        
                                        elif J==l-1:
                                            if j!=0:
                                                dell_Conv[n][I][J]+=dell_val
                                        
                                        else:
                                            dell_Conv[n][I][J]+=dell_val
                                            
            return Gradients,dell_Conv
        return Gradients
    
    Weight_Grad,dell_dense=Back_Dense()
    dell_Flatten=Back_Flatten(dell_dense)
    PImg=Padding(Img)
    
    Pf_maps=[]
    for f in up_fmaps[1]:
        Pf_maps.append(Padding(f))
    dell_Pool2=Back_Pooling(up_fmaps[0],up_fmaps[1],dell_Flatten)
    dell_ReLU2=dell_Pool2
    F2_grad,dell_Conv2=Back_ConvND(2,Pf_maps,dell_ReLU2)
    dell_Pool=Back_Pooling(up_fmaps[2],f_maps,dell_Conv2)
    dell_ReLU=dell_Pool

    F1_grad=Back_ConvND(1,PImg,dell_ReLU)
    
    Gradients=[F1_grad,F2_grad,Weight_Grad]
    return Gradients
  
def Grad_desc(Gradients):
    lr=0.0001
    f=open('CNN_weights.json','r')
    weights=json.load(f)
    f.close()
    
    f=open('CNN_Filters.json','r')    
    Filters=json.load(f)
    Filters1D=Filters[0]
    Filters2D=Filters[1]
    f.close()
    
    for n in range(len(Filters1D)):
        for i in range(len(Filters1D[n])):
            for j in range(len(Filters1D[n][i])):
                Filters1D[n][i][j]-=lr*Gradients[0][n][i][j]
                
    for n in range(len(Filters2D)):
        for i in range(len(Filters2D[n])):
            for j in range(len(Filters2D[n][i])):
                for k in range(len(Filters2D[n][i][j])):
                    Filters2D[n][i][j][k]-=lr*Gradients[1][n][i][j][k]
    
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            weights[i][j]-=lr*Gradients[2][i][j]
    
    f=open('CNN_weights.json','w')
    json.dump(weights,f)
    f.close()
    
    f=open('CNN_Filters.json','w')    
    Filters=[Filters1D,Filters2D]
    json.dump(Filters,f)
    f.close()

def loss_compute(Output,Y):
    loss=0
    e=math.exp(-13)
    for i in range(len(Output)):
        loss+= -1*Y[i]*math.log10(Output[i]+e)
    
    return loss

def Grad0():
    Grad0=[[[[0]*3 for i in range(3)] for n in range(16)],
          [[[[0]*3 for j in range(3)] for i in range(16)] for n in range(16)],
          [[0]*784 for i in range(10)]]
    
    return Grad0

def Up_BGrad(BG,G):
    for n in range(len(BG[0])):
        for i in range(len(BG[0][n])):
            for j in range(len(BG[0][n][i])):
                BG[0][n][i][j]+=G[0][n][i][j]
    
    for n in range(len(BG[1])):
        for i in range(len(BG[1][n])):
            for j in range(len(BG[1][n][i])):
                for k in range(len(BG[1][n][i][j])):
                    BG[1][n][i][j][k]+=G[1][n][i][j][k]
    
    for i in range(len(BG[2])):
        for j in range(len(BG[2][i])):
            BG[2][i][j]+=G[2][i][j]
    
    return BG

def train(bs):
    dataset=fill_dataset('train')
    # dataset=[('Num.png',[0,0,1,0,0,0,0,0,0,0])]
    total_turns=len(dataset)//bs-1
    turn=0
    
    while turn<=total_turns:
        batch=dataset[turn*bs:(turn+1)*bs]
        B_Gradients=Grad0()
        loss=0
        for Img_path,Y in batch:
            Output,Neurons,up_fmaps,f_maps,Img=forward(Img_path)
            Gradients=Backpropagation(Img,f_maps,up_fmaps,Neurons,Output,Y)
            Up_BGrad(B_Gradients,Gradients)
            loss+=loss_compute(Output,Y)/bs
        
        Grad_desc(B_Gradients)
        print(loss)
        turn+=1

train(1)
        
    
    
                
                
                      
