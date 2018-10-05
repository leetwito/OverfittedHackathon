
# coding: utf-8

# In[1]:


from evaluator.iou_evaluator import evaluate_frame, evaluate_folder
from shutil import copy2
from glob import glob

import pandas as pd


# -------------

# In[2]:


# src_files = glob("../data/Train/vid_1/*labels*")[15:900]
# dst_dir = "voxelling_output/submission_files/vid_1_gt/"


# In[3]:


# for f in src_files:
#     copy2(f, dst_dir)


# ----------------

# In[35]:


idx = 100
filename = '0'*(7-len(str(idx)))+str(idx)+"_labels.csv"
filename


# In[36]:


gt = pd.read_csv("voxelling_output/submission_files/vid_1_gt/"+filename, header=None)
# gt = gt.T[0]
gt.values.sum()


# In[37]:


print(filename)
pred2 = pd.read_csv("voxelling_output/submission_files/vid_1_pred/"+filename, header=None)
# pred = pred.T[0]
pred2.values.sum()


# In[38]:


tp, fn, fp = evaluate_frame(gt, pred2)
iou = tp / (tp+fn+fp)
iou[0], tp[0], fn[0], fp[0]


# In[15]:


def calc_iou(gt, pred):
    tp, fn, fp = evaluate_frame(gt, pred)
    iou = tp / (tp+fn+fp)
    return iou


# -----------------------------------

# In[9]:


# tps, fns, fps = evaluate_folder("voxelling_output/submission_files/vid_1_gt/", "voxelling_output/submission_files/vid_1_pred/")


# In[10]:


iou = tps / (tps+fns+fps)
iou

