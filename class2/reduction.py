#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.keras import layers


# In[14]:


class Block1(layers.Layer):
    def __init__(self,**kwargs):
        super(Block1,self).__init__(**kwargs)
        
        self.op1 = layers.SeparableConv2D(filters=20,kernel_size=3,strides=2,padding ='same',activation='relu')
        self.op2 = layers.MaxPool2D(pool_size=3,padding='same',strides=1)
        
        self.op3= layers.Conv2D(kernel_size=1,filters=20,padding='same')
        
        
    def call(self,x):
        x1,x2 =x
        result1 = self.op1(x1)
        result2 = self.op2(x2)
        
        result2 = self.op3(result2)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        
        
        return out


# In[13]:


class Block2(layers.Layer):
    def __init__(self,**kwargs):
        super(Block2,self).__init__(**kwargs)
        
        self.op1 = layers.Conv2D(filters=20,padding='same',strides = 2,kernel_size=3)
        self.op2 = layers.AveragePooling2D(pool_size=3,strides=2,padding='same')
        
        self.op3= layers.Conv2D(kernel_size=1,filters=20,padding='same')
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        
        result2 = self.op3(result2)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        return out


# In[ ]:





# In[16]:


class Block3(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.op1 = layers.AveragePooling2D(pool_size=3,padding='same',strides=2)
        self.op2 = layers.AveragePooling2D(strides=2,padding='same',pool_size=3)
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        
        return out


# In[18]:


class ReductionCell2(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.block1 = Block1()
        self.block2 = Block2()
        self.block3 = Block3()
        
        self.op4 = layers.Conv2D(filters=20,kernel_size=1,padding='same')
        
    def call(self,x):
        result1 = self.block2(x)
        result2 = self.block3(x)
        result2 = self.op4(result2)
        result3 = self.block1([x,result1])
        
        out = layers.Concatenate()([result2,result3])
        
        return out

