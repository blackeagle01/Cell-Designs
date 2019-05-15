#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


from tensorflow.keras import layers


# In[7]:


class Block1(layers.Layer):
    def __init__(self,**kwargs):
        super(Block1,self).__init__(**kwargs)
        
        self.op1 = layers.SeparableConv2D(filters=20,kernel_size=3,padding ='same',activation='relu')
        
        self.op2= layers.Conv2D(kernel_size=1,filters=20,padding='same')
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)

        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        
        
        return out


# In[5]:


test_image= tf.random.normal((1,100,100,3))


# In[8]:


Block1()(test_image).shape


# In[9]:


class Block2(layers.Layer):
    def __init__(self,**kwargs):
        super(Block2,self).__init__(**kwargs)
        
        self.op1 = layers.Conv2D(kernel_size=3,filters=20,padding='same')
        self.op2 = layers.Conv2D(kernel_size=1,filters=20,padding='same')
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        return out


# In[10]:


Block2()(test_image).shape


# In[11]:


class Block3(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.op1 = layers.AveragePooling2D(pool_size=3,strides=1,padding='same')
        
        self.op2 = layers.AveragePooling2D(pool_size=3,strides=1,padding='same')
        
        self.op3 = layers.Conv2D(filters=20,padding='same',kernel_size=1)
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        
        
        out = layers.Add()([result1,result2])
        out = self.op3(out)
        out = layers.ReLU()(out)
        
        return out


# In[12]:


Block3()(test_image).shape


# In[14]:


class NormalCell3(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.op1 = Block1()
        self.op2 = Block2()
        self.op3 = Block3()
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        result3 = self.op3(x)
        
        out = layers.Concatenate()([result1,result2,result3])
        
        return out


# In[15]:


NormalCell3()(test_image).shape


# In[ ]:




