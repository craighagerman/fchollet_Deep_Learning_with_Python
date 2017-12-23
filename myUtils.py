
# coding: utf-8

# In[13]:


# use %matplotlib inline in the first cell of the notebook

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [10, 6]

import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[14]:


# Helper functions for plotting loss & accuracy

def plotLoss(history):
    plt.clf()   # clear figure
    acc = history.history['acc']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print("min val_loss: ", val_loss[np.argmin(val_loss)])

def plotAcc(history):
    plt.clf()   # clear figure
    history_dict = history.history
    
    acc = history.history['acc']
    val_acc = history_dict['val_acc']    
    epochs = range(1, len(acc) + 1)
        
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()    
    print("max val_acc: ", val_acc[np.argmax(val_acc)] )


# In[15]:


get_ipython().run_cell_magic('bash', '', '# Save current Python notebook as python script\njupyter nbconvert --to script myUtils.py')


# In[ ]:




