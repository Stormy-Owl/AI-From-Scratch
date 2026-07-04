from PIL import Image 
import math
import json
import random
import os

Neurons=[]
HL1=[]
HL2=[]
output=[]
dataset=[]
#dataset=[("Num.png",[0,0,0,0,0,0,0,0,1,0])] for trial
gradiants=[]

def updates():
    global weights_list,weights,hid_weights,out_weights,bias_list,bias,hid_bias,out_bias
    f=open('weights.json','r')
    weights_list=json.load(f)
    weights=weights_list[0]
    hid_weights=weights_list[1]
    out_weights=weights_list[2]
    f.close()

    f=open('bias.json','r')
    bias_list=json.load(f)
    bias=bias_list[0]
    hid_bias=bias_list[1]
    out_bias=bias_list[2]
    f.close()

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

def img_to_grayscale(img):
    img=Image.open(img)
    img=img.convert("L")
    img=img.resize((28,28))
    pixels = list(img.getdata())
    return pixels

def tanh(x):
    ex=math.exp(x)
    ex_inv=math.exp(-x)
    y=(ex-ex_inv)/(ex+ex_inv)
    return y

def softmax(l):
    global output
    output=[]
    s=0
    for i in l:
        s+=math.exp(i)
        output.append(math.exp(i))
    output=[p/s for p in output]
        

def Neu2HL1():
    global HL1
    for i in range(16):
        s=0
        for j in range(784):
            s+=weights[i][j]*Neurons[j][0]

        s+=bias[i][0]
        HL1[i]=[tanh(s)] 
    return HL1

def HL1_toHL2():
    global HL2
    for i in range(16):
        s=0
        for j in range(16):
            s+=hid_weights[i][j]*HL1[j][0]
        s+=hid_bias[i][0]
        HL2[i]=[tanh(s)]
        return HL2
        
def HL2_to_output():
    global output
    l=[]
    for i in range(10):
        s=0
        for j in range(16):
            s+=out_weights[i][j]*HL2[j][0]
        s+=out_bias[i][0]
        l.append(s)
    softmax(l)
    # print(output)
    return output
    

def Backpropagation(g,Y):
    def for_HL2():
        global HL2_gradiant
        HL2_gradiant=[]
        deltas=[]
        for i in range(10):
            dell_out=(output[i]-Y[i])
            HL2_gradiant.append([])
            for j in range(16):
                HL2_gradiant[i].append(dell_out*HL2[j][0])
            HL2_gradiant[i].append(dell_out)
            deltas.append(dell_out)
        
        HL2_func=[]
        for i in range(16):
            s=0
            for j in range(10):
                s+=deltas[j]*out_weights[j][i]*(1-(HL2[i][0]**2))
            HL2_func.append(s)
        return HL2_func
    
    def for_HL1():
        global HL1_gradiant
        HL1_gradiant=[]
        deltas=[]
        HL2_func=for_HL2()
        
        for i in range(16):
            dell_hid=HL2_func[i]
            HL1_gradiant.append([])
            for j in range(16):
                HL1_gradiant[i].append(dell_hid*HL1[j][0])
            HL1_gradiant[i].append(dell_hid)
            deltas.append(dell_hid)
            
        HL1_func=[]
        for i in range(16):
            s=0
            for j in range(16):
                s+=deltas[j]*hid_weights[j][i]*(1-(HL1[i][0])**2)
            HL1_func.append(s)
        return HL1_func
    
    def for_Neu():
        global Neurons_gradiant
        
        Neurons_gradiant=[]
        HL1_func=for_HL1()

        for i in range(16):
            dell_neu=HL1_func[i]
            Neurons_gradiant.append([])
            for j in range(784):
                Neurons_gradiant[i].append(dell_neu*Neurons[j][0])
            Neurons_gradiant[i].append(dell_neu)
            
    for_Neu()
    
    g.append(Neurons_gradiant)
    g.append(HL1_gradiant)
    g.append(HL2_gradiant)

def grad_desc(g):
    lr=0.01
     
    for i in range(3):
        for j in range(len(g[i])):
            for k in range(len(g[i][j])):
                if k!=len(g[i][j])-1:
                    weights_list[i][j][k]-= lr*g[i][j][k]
                else:
                    bias_list[i][j][0]-= lr*g[i][j][k]
    f=open('weights.json','w')
    json.dump(weights_list,f)
    f.close()
    f=open('bias.json','w')
    json.dump(bias_list,f)
    f.close()
    
def batch_training(batch_size=32):
    global  Neurons,HL1,HL2,output,gradiants,dataset,deltas
    fill_dataset('train')
    n=len(dataset)//batch_size
    
    for i in range(n):
        updates()
        batch=dataset[batch_size*i:batch_size*(i+1)]
        batch_grad=[
            [[0]*785 for _ in range(16)],   # Input → HL1
            [[0]*17  for _ in range(16)],   # HL1 → HL2
            [[0]*17  for _ in range(10)]    # HL2 → output
        ]
        loss=0
        for img,Y in batch:
            # print(Y)
            pixels = img_to_grayscale(img)

            for j in range(784):
                Neurons.append([pixels[j]/255])

            HL1=[[0] for _ in range(16)]
            HL2=[[0] for _ in range(16)]
                
            Neu2HL1()
            HL1_toHL2()
            HL2_to_output()
            Backpropagation(gradiants,Y)

            for p in range(3):
                for q in range(len(gradiants[p])):
                    for r in range(len(gradiants[p][q])):
                        batch_grad[p][q][r]+=gradiants[p][q][r]/batch_size
            gradiants=[]
            Neurons.clear()
            HL1.clear()
            HL2.clear()
            for j in range(10):
                loss+=Y[j]*math.log(output[j]+math.exp(-9))/batch_size
        
        cost=(-1)*loss
        print(cost)
        grad_desc(batch_grad)

def test():
    fill_dataset('test')
    img=dataset[0][0]
    Y=dataset[0][1]
    pixels=img_to_grayscale(img)
    Neurons=[[p/255] for p in pixels]
    
    Neu2HL1()
    HL1_toHL2()
    HL2_to_output()
    
    print(output,Y)
    
    for i in range(10):
        for j in range(10):
            if i<j:
                break
        else:
            print(i)
            
batch_training(15)   
    
        
