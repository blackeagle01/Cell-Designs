#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.keras import layers


# In[34]:


class Block1(layers.Layer):
    def __init__(self,**kwargs):
        super(Block1,self).__init__(**kwargs)
        
        self.op1 = layers.SeparableConv2D(filters=20,kernel_size=3,padding ='same',activation='relu')
        self.op2 = layers.SeparableConv2D(filters=20,kernel_size=3,padding= 'same',activation='relu')
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        
        
        return out


# In[23]:


#test_image = tf.random.normal((1,100,100,20))


# In[36]:


class Block2(layers.Layer):
    def __init__(self,**kwargs):
        super(Block2,self).__init__(**kwargs)
        
        self.op1 = layers.SeparableConv2D(filters=20,kernel_size=5,padding = 'same',activation='relu')
        self.op2 = layers.Conv2D(filters=20,kernel_size=1,padding= 'same')
        
        
    def call(self,x):
        x1,x2 = x
        result1 = self.op1(x1)
        result2 = self.op2(x2)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        return out


# In[30]:


class Block3(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.op1 = layers.Conv2D(filters=20,kernel_size=5,padding = 'same',activation='relu')
        self.op2 = layers.Conv2D(filters=20,kernel_size=5,padding= 'same',activation='relu')
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2(x)
        
        out = layers.Add()([result1,result2])
        out = layers.ReLU()(out)
        
        return out


# In[40]:


class NormalCell1(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.op1 = Block1()
        self.op2 = Block2()
        self.op3 = Block3()
        
        
    def call(self,x):
        result1 = self.op1(x)
        result2 = self.op2([result1,x])
        result3 = self.op3(x)
        
        out = layers.Concatenate()([result2,result3])
        
        return out


# In[41]:


#NormalCell()(test_image)


# In[ ]:




