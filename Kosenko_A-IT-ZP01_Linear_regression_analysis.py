#!/usr/bin/env python
# coding: utf-8

# #Introduction to Machine Learning
# Laboratory work №1
# 
# Kosenko Anton IT-ZP01
# 
# Task: Linear regression analysis

# # Dataset generation

# In[6]:


#import necessary library numpy for computing and matplotlib for creation a plot
import numpy as np
import matplotlib.pyplot as plt
#special feature of jupyter notebook for drawing only static images in notebook
#%matplotlib inline

np.random.seed(42) #initialization of pseudo random numbers generator 


# In[7]:


#create three float objects 
true_b = 20.0 #initial value of dependent value y(pred) when x=0
true_w = 2.0 #initial slope coefficient
random_magnitude = 0.3 #value for computing residuals (random errors)


# In[11]:


#create two objects x, y using np.linspace for x and following formula for y 
x = np.linspace(0, 1, 100) #created a sequence (array) of 100 numbers from 0 to 1 
#real line deviation formula for y where true_b is a free term, true_w - slope coefficient 
y = true_b + true_w * x + random_magnitude * np.random.randn(100)


# In[12]:


plt.scatter(x, y) #create a plot with dots
plt.show()#show plot

# In[13]:


#create train and test sets for x and y
train_x = x[:60] #create a train set of x values from the first index (included) to 60 index (excluded) 
test_x = x[60:] #create a test set of x values from 60 index (included) to the last index

train_y = y[:60] #create a train set of y values from the first index (included) to 60 index (excluded)
test_y = y[60:] #create a test set of y values from 60 index (included) to the last index


# # Linear regression fitting

# In[14]:


#For understanding how does Linear regression work, let discover each step of regression
#Create next objects for 
gd_intercept = 0.0 #define the initial value of dependent value y(pred) when X=0 
gd_weight = 1.0 #define the initial coefficient value near X 
epochs = 1000 #define maximum number of iterations   
lr = 0.1 #coefficient of normalization
tol = 0.001 #define tolerance for stopping criteria

pred = gd_intercept + train_x * gd_weight #compute the initial value of dependent value y(prediction)
error = train_y - pred # compute the model error

#create a loop from 1 to 1000
for i in range(1, epochs + 1):
    pred = gd_intercept + train_x * gd_weight #compute dependent variable y(prediction) for current iteration
    
    #compute value of the model error for current iteration
    error = train_y - pred #compute difference between value of train_y and value of predicted y for current iteration
    loss = np.mean(error*error) #compute the mean squad errors
    print(f"{i} - Train Loss: {loss}") #print a f-string with changing value of loss for current iterration
    
    #make the gradient descent 
    dw = np.mean(error * (-2 * train_x)) #compute the directions of descent (slope coefficient) 
    #by multiplication a value of error and train_x for current iteration on (-2) 
    db = np.mean(error * (-2)) #compute the parameter that maps the directions for y-intersection for current iteration
    
    intercept_update = lr * db #compute the normalized corelation for y-intersection dependent value y(pred) for current iteration
    weight_update = lr * dw #compute the normalized corelation for slope coefficient value for current iteration 
    gd_intercept = gd_intercept - intercept_update #compute the value of new dependent value y(pred)-intersection for the next iteration
    gd_weight = gd_weight - weight_update #compute the value of the new slope coefficient near X for the next iteration
    
    #define the condition for breaking a loop: 
    #break a loop if absolute value of normalized corelation for gradient interception and a slope coefficient value are lower 
    #than defined tolerance value 
    if abs(intercept_update) < tol and abs(weight_update) < tol:
        break
        
#print f-strings with final value of gradient interception, slope coefficient value and value of loss for the last train_x before break     
print(f"Intercept: {gd_intercept}")
print(f"Coefficient: {gd_weight}")
print(f"Train Loss: {loss}")
plt.show()#show plot

# In[15]:


#compute and print Test Mean Squared Error
output = gd_intercept + test_x * gd_weight
error = test_y - output
print(f"Test MSE: {np.mean(error*error)}")


# In[16]:


#vizualize results and marks train data set by blue color and test data set with green color, line regression mark by red color
plt.scatter(train_x, train_y, color="blue")
plt.scatter(test_x, test_y, color="green")
plt.plot([0, 1], [gd_intercept, gd_weight + gd_intercept], color="red") #linear regression


# In[ ]:




