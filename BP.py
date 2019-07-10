import numpy as np 
x= np.array([[1,0,0],
             [1,0,1],
             [1,1,0],
             [1,1,1]



            ])
y=np.array([[0,1,1,0]])
v=np.random.random((3,4))
w=np.random.random((4,1))

lr=0.11

def sigmod(x):
  return 1/(1+np.exp(-x))

def dsigmod(x):
  return x*(1-x)

def update():
  global x,y,v,w,lr,l1,l2

  l1=sigmod(np.dot(x,v))
  l2=sigmod(np.dot(l1,w))

  l2_delta=(y.T-l2)*dsigmod(l2)
  l1_delta=l2_delta.dot(w.T)*dsigmod(l1)
     
  w_c=lr*l1.T.dot(l2_delta)
  v_c=lr*x.T.dot(l1_delta)

  w=w+w_c
  v=v+v_c

for i in range(20000):
  update()
  if i % 1000==0:
    print("error:",np.mean(np.abs(y.T-l2)))
print(l2)
