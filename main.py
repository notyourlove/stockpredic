import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import json



window = 30 
file = 'aapl.csv' #the file of which weight and bias will be used 
file2 = 'tesla.csv'#the file which will be predicted 


lr = 0.001
epochs = 801


x = []
y = []
checkx = []
checky = []
df = pd.read_csv(file)
xmin = min(list(df['Close']))
xmax = max(list(df['Close']))
def preparexy(df,x,y,checkx, checky):
  xmin = min(list(df['Close']))
  xmax = max(list(df['Close']))
  window = 30 
  eighty = int(len(list(df['Close']))*80/100)
  N = eighty - 30
  twenty = int(len(list(df['Close']))*20/100) 
  traindata = list(df['Close'].head(eighty))
  checkdata = list(df['Close'].tail(twenty))
  for i in range(len(traindata)-30):
    x.append(traindata[i:i+window])
    y.append(traindata[i+window])

  for i in range(len(checkdata)-30):
    checkx.append(checkdata[i:i+window])
    checky.append(checkdata[i+window])


  checkx = np.array(checkx)
  checkx = (checkx - xmin)/(xmax - xmin)
  checky = np.array(checky).reshape(-1,1)
  checky = (checky - xmin )/(xmax - xmin)

  x = np.array(x) #Nx30
  x = (x-xmin)/(xmax-xmin)
#real_price = prediction × (x_max − x_min) + x_min
  y = np.array(y).reshape(-1,1)
  y = (y - xmin) / (xmax - xmin) 

  return x, y ,checkx, checky, N


testx = []
testy = []
df2 = pd.read_csv(file2)
testdata = list(df2['Close'].head(8000))

for i in range(len(testdata)-30):
  testx.append(testdata[i:i+window])
  testy.append(testdata[i+window])

testx = np.array(testx)
testx = (testx - xmin)/(xmax - xmin)
testy = np.array(testy).reshape(-1,1)
testy = (testy - xmin )/(xmax - xmin)



def relu(x):
    return np.maximum(0, x)


f1 =  open('json1.json','r') 
fil = f1.read()
if fil == "":
    ope2 = open('json1.json','w')
    ope2.write('{}')
    ope2.close()
    f1.close()
f1 =  open('json1.json','r') 
fil = f1.read()
dic = json.loads(fil)
if file in dic:
    pass
else :
   x, y ,checkx, checky , N = preparexy(df,x,y,checkx,checky)
#####################


np.random.seed(42)
w1 = np.random.randn(window, 64) * np.sqrt(2/window)
#after x.w1 = Nx64
b1 = np.zeros((1, 64))
#chilll it will add in every roww :)


w2 = np.random.randn(64, 32) * np.sqrt(2/64)
#cozz to match row and colum
b2 = np.zeros((1, 32))

w3 = np.random.randn(32, 1) * np.sqrt(2/32)
b3 = np.zeros((1,1))



def training(x,y,w1,b1,w2,b2,w3,b3, N):
  for epoch in range(epochs):

    z1 = np.dot(x, w1) + b1
    a1 = relu(z1) # Nx64
    z2 = np.dot(a1,w2) + b2
    a2 = relu(z2) #Nx32
    z3 = np.dot(a2, w3) + b3 #aka y^

    ########

    l = np.mean((y - z3)**2) #it equal to 1/n(sum of (y - z3ory^)**2)


    dz3 = 2/N*(z3-y)
    dw3 = np.dot(a2.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)
    da2 = np.dot(dz3, w3.T)
    dz2 = da2 * (z2 > 0)
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * (z1 > 0)
    dw1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)


    w1 -= lr*dw1
    b1 -= lr*db1
    w2 -= lr*dw2
    b2 -= lr*db2
    w3 -= lr*dw3
    b3 -= lr*db3

    if epoch%100 == 0:
        print(f"step {epoch} : {l}")
    
  
  z1 = np.dot(checkx, w1) + b1
  a1 = relu(z1) # Nx64
  z2 = np.dot(a1,w2) + b2
  a2 = relu(z2) #Nx32
  z3 = np.dot(a2, w3) + b3

  
  l = np.mean((checky - z3)**2)
  print(f"The MSE or loss for the checkdata of {file} is {l}")
  return w1,b1,w2,b2,w3,b3
      


f1 =  open('json1.json','r') 
fil = f1.read()
dic = json.loads(fil)
if file in dic:
    print("Got file")
    w1 = np.array(dic[file]['w1'])
    w2 = np.array(dic[file]['w2'])
    w3 = np.array(dic[file]['w3'])
    b1 = np.array(dic[file]['b1'])
    b2 = np.array(dic[file]['b2'])
    b3 = np.array(dic[file]['b3'])
    print('Got all weights and bias')
        
else:
    w1, b1, w2, b2, w3, b3 = training(x,y,w1,b1,w2,b2,w3,b3,N)
    
    with open('json1.json','r') as f:
      dic = json.load(f)
    dic[file] = {
        "w1": w1.tolist(),
        "w2": w2.tolist(),
        "w3": w3.tolist(),
        "b1": b1.tolist(),
        "b2": b2.tolist(),
        "b3": b3.tolist()
    }
    with open('json1.json','w') as f:
        json.dump(dic, f)
 





z1 = np.dot(testx, w1) + b1
a1 = relu(z1) # Nx64
z2 = np.dot(a1,w2) + b2
a2 = relu(z2) #Nx32
z3 = np.dot(a2, w3) + b3 #aka y^

  
    
real_price2 = z3 * (xmax - xmin) + xmin
actual_price2    = testy * (xmax - xmin) + xmin   
    
    
acc = (np.round(real_price2)) - (np.round(actual_price2))
accc = np.sum(acc ==0)
l = np.mean((testy - z3)**2)
print(f"The MSE of {file2} is {l}")
print(f'Accuracy {accc} / {actual_price2.shape[0]} (Note:round figure is taken)')


plt.plot(actual_price2, label="Actual",    color="blue")
plt.plot(real_price2,   label="Predicted", color="orange")
plt.legend()
plt.title(file2)
plt.xlabel("Day")
plt.ylabel("Price (USD)")
plt.show()



