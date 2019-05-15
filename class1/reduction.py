#!/usr/bin/env python
# coding: utf-8

# In[11]:


import tensorflow as tf


# In[12]:


from tensorflow.keras import layers


# In[13]:


class Block1(layers.Layer):
    def __init__(self,**kwargs):
        super(Block1,self).__init__(**kwargs)
        
        self.op1 = layers.SeparableConv2D(filters=20,kernel_size=3,strides =2,padding ='same',activation='relu')
        self.op2 = layers.SeparableConv2D(filters=20,kernel_size=3,strides=2,padding= 'same',activation='relu')
        
        
    def call(self,x):
        x1,x2=x
        result1 = self.op1(x1)
        result2 = self.op2(x2)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        
        
        return out


# In[21]:


#test_image = tf.random.normal((1,100,100,3))


# In[22]:


#Block1()([test_image]*2).shape


# In[15]:


class Block2(layers.Layer):
    def __init__(self,**kwargs):
        super(Block2,self).__init__(**kwargs)
        
        self.op1 = layers.AveragePooling2D(pool_size=3,strides=1,padding='same')
        self.op2 = layers.AveragePooling2D(pool_size=3,strides=1,padding='same')
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        return out


# In[23]:


#Block2()(test_image).shape


# In[25]:


class Block3(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.op1 = layers.Conv2D(filters=20,kernel_size=5,strides=2,padding = 'same',activation='relu')
        self.op2 = layers.MaxPool2D(pool_size=3,strides=2,padding='same')
        
        self.op3 = layers.Conv2D(filters=20,kernel_size=1,padding='same')
        
        
    def call(self,x):
        x1,x2=x
        result1 = self.op1(x1)
        result2 = self.op2(x2)
        
        result2 = self.op3(result2)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        
        return out


# In[27]:


#Block3()([test_image]*2).shape


# In[28]:


class ReductionCell1(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.op1 = Block1()
        self.op2 = Block2()
        self.op3 = Block3()
        
        self.op4 = layers.Conv2D(filters=20,kernel_size=1,padding='same')
        
        
    def call(self,x):
        result1 = self.op2(x)
        result1 = self.op4(result1)
        result2 = self.op1([x,result1])
        result3 = self.op3([result1,x])
        
        out = layers.Concatenate()([result2,result3])
        
        return out


# In[29]:


#ReductionCell()(test_image).shape


# In[ ]:




