
# coding: utf-8

# In[13]:


from evaluator.iou_evaluator import evaluate_frame, evaluate_folder
from shutil import copy2
from glob import glob

import pandas as pd


# -------------

# In[14]:


# src_files = glob("../data/Train/vid_1/*labels*")[15:900]
# dst_dir = "voxelling_output/submission_files/vid_1_gt/"


# In[15]:


# for f in src_files:
#     copy2(f, dst_dir)


# ----------------

# In[39]:


idx = 75
filename = '0'*(7-len(str(idx)))+str(idx)+"_labels.csv"
filename


# In[40]:


gt = pd.read_csv("voxelling_output/submission_files/vid_1_gt/"+filename, header=None)
# gt = gt.T[0]
gt.values.sum()


# In[41]:


print(filename)
pred2 = pd.read_csv("voxelling_output/submission_files/vid_1_pred_lee/"+filename, header=None)
# pred = pred.T[0]
pred2.values.sum()


# In[42]:


tp, fn, fp = evaluate_frame(gt, pred2)
iou = tp / (tp+fn+fp)
iou[0], tp[0], fn[0], fp[0]


# In[30]:


def calc_iou(gt, pred):
    tp, fn, fp = evaluate_frame(gt, pred)
    iou = tp / (tp+fn+fp)
    return iou


# -----------------------------------

# In[48]:


tps, fns, fps = evaluate_folder("voxelling_output/submission_files/vid_1_gt/", "voxelling_output/submission_files/vid_1_pred_lee/")


# In[49]:


iou = tps / (tps+fns+fps)
iou

