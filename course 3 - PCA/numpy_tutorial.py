#!/usr/bin/env python
# coding: utf-8

# # Introduction to Numpy

# NumPy is the fundamental package for scientific computing
# in Python. It is a Python library that provides a multidimensional array
# object. In this course, we will be using NumPy for linear algebra.
# 
# If you are interested in learning more about NumPy, you can find the user
# guide and reference at https://docs.scipy.org/doc/numpy/index.html

# Let's first import the NumPy package

# In[1]:


import numpy as np # we commonly use the np abbreviation when referring to numpy


# ## Creating Numpy Arrays

# New arrays can be made in several ways. We can take an existing list and convert it to a numpy array:

# In[2]:


a = np.array([1,2,3])
a


# There are also functions for creating arrays with ones and zeros

# In[3]:


np.zeros((2,2))


# In[4]:


np.ones((3,2))


# ## Accessing Numpy Arrays
# You can use the common square bracket syntax for accessing elements
# of a numpy array

# In[5]:


A = np.arange(9).reshape(3,3)
print(A)


# In[6]:


print(A[0]) # Access the first row of A
print(A[0, 1]) # Access the second item of the first row
print(A[:, 1]) # Access the second column


# ## Operations on Numpy Arrays
# You can use the operations '*', '**', '\', '+' and '-' on numpy arrays and they operate elementwise.

# In[7]:


a = np.array([[1,2], 
              [2,3]])
b = np.array([[4,5],
              [6,7]])


# In[8]:


print(a + b)


# In[9]:


print(a - b)


# In[10]:


print(a * b)


# In[11]:


print(a / b)


# In[12]:


print(a**2)


# There are also some commonly used function
# For example, you can sum up all elements of an array

# In[13]:


print(a)
print(np.sum(a))


# Or sum along the first dimension

# In[14]:


np.sum(a, axis=0)


# There are many other functions in numpy, and some of them **will be useful**
# for your programming assignments. As an exercise, check out the documentation
# for these routines at https://docs.scipy.org/doc/numpy/reference/routines.html
# and see if you can find the documentation for `np.sum` and `np.reshape`.

# ## Linear Algebra
# 
# In this course, we use the numpy arrays for linear algebra.
# We usually use 1D arrays to represent vectors and 2D arrays to represent
# matrices

# In[15]:


A = np.array([[2,4], 
             [6,8]])


# You can take transposes of matrices with `A.T`

# In[16]:


print('A\n', A)
print('A.T\n', A.T)


# Note that taking the transpose of a 1D array has **NO** effect.

# In[17]:


a = np.ones(3)
print(a)
print(a.shape)
print(a.T)
print(a.T.shape)


# But it does work if you have a 2D array of shape (3,1)
# 

# In[18]:


a = np.ones((3,1))
print(a)
print(a.shape)
print(a.T)
print(a.T.shape)


# ### Dot product

# We can compute the dot product between two vectors with np.dot

# In[19]:


x = np.array([1,2,3])
y = np.array([4,5,6])
np.dot(x, y)


# We can compute the matrix-matrix product, matrix-vector product too. In Python 3, this is conveniently expressed with the @ syntax

# In[20]:


A = np.eye(3) # You can create an identity matrix with np.eye
B = np.random.randn(3,3)
x = np.array([1,2,3])


# In[21]:


# Matrix-Matrix product
A @ B


# In[22]:


# Matrix-vector product
A @ x


# Sometimes, we might want to compute certain properties of the matrices. For example, we might be interested in a matrix's determinant, eigenvalues/eigenvectors. Numpy ships with the `numpy.linalg` package to do
# these things on 2D arrays (matrices).

# In[23]:


from numpy import linalg


# In[24]:


# This computes the determinant
linalg.det(A)


# In[25]:


# This computes the eigenvalues and eigenvectors
eigenvalues, eigenvectors = linalg.eig(A)
print("The eigenvalues are\n", eigenvalues)
print("The eigenvectors are\n", eigenvectors)


# ## Miscellaneous

# ### Time your code
# One tip that is really useful is to use the magic commannd `%time` to time the execution time of your function.

# In[26]:


get_ipython().run_line_magic('time', 'np.abs(A)')


# In[ ]:




