import pandas as pd
import numpy as np
dataframe = pd.read_csv('Cricket_chirps.csv')

a = 0.6
dataset1 = dataframe.copy()
train_dataset = dataset1.sample(frac = a)
dataset1 = dataset1.drop(train_dataset.index)
a = 0.5
validation_dataset = dataset1.sample(frac = a)
test_dataset = dataset1.drop(validation_dataset.index)

print("train_dataset.....")
print(train_dataset)

print("validation_dataset.....")
print(validation_dataset)

print("test_dataset.....")
print(test_dataset)

def MSE_ofGivenDataset(dataset):

  max1 = dataset['X'].max()
  max2 = dataset['Y'].max()


  learning_rates = [0.01, 0.001, 0.3, 0.5, 1]
  w0, w1 = 3, 3;
  #h = w0  + w1*x
  pre_j = 0
  epoch = 100
  p = 0.0000001
  dataset = dataset.to_numpy()

  #normalize the datset
  for i in range(len(dataset)):
    dataset[i][0] /= max1
    dataset[i][1] /= max2

  for lr in learning_rates:
    for itr  in range(epoch):
      j = 0
      m = len(dataset)
      temp1 = 0
      temp2 = 0
      #print("total datset..", m)
      for ind in range(len(dataset)):
        j = j + ((w0 + w1*dataset[ind][0]) - dataset[ind][1])**2
        temp1 = temp1 +  (((w0 + w1*dataset[ind][0]) - dataset[ind][1])*1)
        temp2 = temp2 + (((w0 + w1*dataset[ind][0]) - dataset[ind][1])*dataset[ind][0])
      
      j = j/(2*m)
      j = round(j, 8)
      w0 = w0 - (lr * temp1)/m
      w0 = round(w0, 8)
      w1 = w1 - (lr * temp2)/m
      w1 = round(w1, 8)
      #print("cofficient", w0, w1)

      if abs(pre_j - j) <= p or itr == epoch - 1:
        print("For learning rate", lr, "......")
        print("MSE =", j)
        break
      pre_j = j   


print("For Training dataset............\n")
MSE_ofGivenDataset(train_dataset)

print("\n\n\n\n\nFor validation dataset............\n")
MSE_ofGivenDataset(validation_dataset)

print("\n\n\n\n\nFor test dataset............\n")
MSE_ofGivenDataset(test_dataset)



def MSE_givenLearningRate(dataset):

  max1 = dataset['X'].max()
  max2 = dataset['Y'].max()
  MSE_ofEachEpoch = []
  lr = 0.1
  w0, w1 = 3, 3;
  #h = w0  + w1*x
  epoch = 100
  dataset = dataset.to_numpy()

  #normalize the datset
  for i in range(len(dataset)):
    dataset[i][0] /= max1
    dataset[i][1] /= max2

  for itr  in range(epoch):
    j = 0
    m = len(dataset)
    temp1 = 0
    temp2 = 0
    #print("total datset..", m)
    for ind in range(len(dataset)):
      j = j + ((w0 + w1*dataset[ind][0]) - dataset[ind][1])**2
      temp1 = temp1 +  (((w0 + w1*dataset[ind][0]) - dataset[ind][1])*1)
      temp2 = temp2 + (((w0 + w1*dataset[ind][0]) - dataset[ind][1])*dataset[ind][0])
    
    j = j/(2*m)
    j = round(j, 8)
    w0 = w0 - (lr * temp1)/m
    w0 = round(w0, 8)
    w1 = w1 - (lr * temp2)/m
    w1 = round(w1, 8)
    MSE_ofEachEpoch.append(j)
    

  return MSE_ofEachEpoch

print(MSE_givenLearningRate(train_dataset)) # This will print MSE of your model trained

x_axis = [i for i in range(1, 101)]
print(x_axis)
y_axis = MSE_givenLearningRate(train_dataset)
print(y_axis)


import matplotlib.pyplot as plt

plt.plot(x_axis, y_axis)

plt.xlabel('epoch')
plt.ylabel('MSE')

plt.show() # this will plot graph epoch by epoch


