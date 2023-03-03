# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
datanumber = 8
df=pd.read_csv('diabetes.csv')
print(df.head())

print(df.isnull().sum())


import seaborn as sns
import numpy as np
#df['Outcome']=np.where(df['Outcome']==1,"Diabetic","No Diabetic")

df.head()

sns.pairplot(df,hue="Outcome")


X=df.drop('Outcome',axis=1).values### independent features
y=df['Outcome'].values###dependent features

print(X)
print(y)
print(type(X))
print(type(y))
"""
X : <class 'numpy.ndarray'>
y : <class 'numpy.ndarray'>
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#### Libraries From Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

##### Creating Tensors
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
# y_train=torch.tensor(y_train, requires_grad=True).type(torch.float32)
# y_test=torch.tensor(y_test, requires_grad=True).type(torch.float32)

print(df.shape)
print(f"X_train {X_train}")
print(f"y_train {y_train}")

#### Creating Modelwith Pytorch

class ANN_Model(nn.Module):
    def __init__(self,input_features=datanumber,hidden1=20,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x
####instantiate my ANN_model
torch.manual_seed(20)
model=ANN_Model()
print(model.parameters)

###Backward Propogation-- Define the loss_function,define the optimizer
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

epochs=500
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    # print("Epoch number: {} /// {} {}".format(i, y_pred, y_train))
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss.item())
    if i%10==1:
        print("Epoch number: {} and the loss : {} /// {} {}".format(i,loss.item(),y_pred,y_train))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### plot the loss function
import matplotlib.pyplot as plt

plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

#### Prediction In X_test data
predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        predictions.append(y_pred.argmax().item())
        #print(y_pred.argmax().item())


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predictions)
print(cm)

plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,predictions)
print(score)

print(model.eval())

# ### Predcition of new data point
# print(list(df.iloc[0,:-1]))
# #### New Data
# lst1=[6.0, 130.0, 72.0, 40.0, 0.0, 25.6, 0.627, 45.0]
# new_data=torch.tensor(lst1)
# #### Predict new data using Pytorch
# with torch.no_grad():
#     print(model(new_data))
#     print(model(new_data).argmax().item())


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
