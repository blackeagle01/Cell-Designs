#!/usr/bin/env python
# coding: utf-8

# In[11]:


import tensorflow as tf


# In[12]:


from tensorflow.keras import layers


# In[37]:


class Block1(layers.Layer):
    def __init__(self,**kwargs):
        super(Block1,self).__init__(**kwargs)
        
        self.op1 = layers.SeparableConv2D(filters=20,kernel_size=3,strides =2,padding ='same',activation='relu')
        self.op2 = layers.Conv2D(filters=20,kernel_size=1,padding='same',strides=2)
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        
        
        return out


# In[38]:


test_image = tf.random.normal((1,100,100,3))


# In[40]:


Block1()(test_image).shape


# In[41]:


class Block2(layers.Layer):
    def __init__(self,**kwargs):
        super(Block2,self).__init__(**kwargs)
        
        self.op1 = layers.Conv2D(filters=20,kernel_size=3,strides=2,padding='same')
        
        
    def call(self,x):
        x1,x2 = x
        result1 = self.op1(x1)
        result2 = x2
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        return out


# In[42]:


Block2()([test_image,Block1()(test_image)]).shape


# In[49]:


class Block3(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.op1 = layers.SeparableConv2D(filters=20,kernel_size=5,strides=2,padding = 'same',activation='relu')
        self.op2 = layers.MaxPool2D(pool_size=3,strides=1,padding='same')
    
        
    def call(self,x):
        x1,x2=x
        result1 = self.op1(x1)
        result2 = self.op2(x2)
        
        out = layers.Concatenate()([result1,result2])
        out = layers.ReLU()(out)
        
        return out


# In[50]:


Block3()([test_image,Block2()([test_image,Block1()(test_image)])]).shape


# In[53]:


class ReductionCell3(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.op1 = Block1()
        self.op2 = Block2()
        self.op3 = Block3()
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2([x,result1])
        
        out = self.op3([x,result2])
        
        return out


# In[54]:


ReductionCell3()(test_image).shape


# In[ ]:




