
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from time import time
from tqdm import tqdm
from utilities.math_utils import RotationTranslationData
from utilities import data_utils
from glob import glob
import pickle

from time import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def add_grid_and_hash(df, voxel_size):
    """ 
    add grid and hash to each point of a given point cloud
    """
    df_grid = df.iloc[:,:3]//voxel_size
    df_grid.iloc[:, :3] = df_grid.iloc[:, :3]*voxel_size + voxel_size//2
    df_grid['voxel_id'] = df_grid.apply(lambda x: hash((x[0], x[1], x[2])), axis=1)
    df_grid = pd.concat([df, df_grid], axis=1)
    df_grid.columns = ['x', 'y', 'z', 'r', 'x_grid', 'y_grid', 'z_grid', 'voxel_id']
    return df_grid


# -----------------------------

# In[3]:


base_dir = "voxelling_output/World2/Train/vid_1/"
pickles_path = "voxelling_output/"
orig_files_path = "voxelling_output/Original_Frames/vid_1/"
n_frames_per_side=50
shift=20
voxel_size=20


# In[4]:


all_files = glob(base_dir+"/*point*")


# In[5]:


vid = base_dir.split("/")[-2]


# In[6]:


def create_list_dfs_voxel_scene(base_dir, voxel_size, upto=None):
    global all_files
    list_df_voxel_scene = []
#     all_files = glob(base_dir+"/*point*")
    if upto is not None:
        all_files = all_files[:upto]
    
    for f in tqdm(all_files):
        list_df_voxel_scene.append( add_grid_and_hash(pd.read_csv(f, header=None), voxel_size))
    list_df_voxel_scene = pd.Series(list_df_voxel_scene).values
    return list_df_voxel_scene


# In[7]:


list_df_file = pickles_path+"list_df_voxel_scene__%s__voxel_size_%d.p"%(vid, voxel_size)

if os.path.exists(list_df_file):
    list_df_voxel_scene = pickle.load(open(list_df_file, "rb"))
else:
    list_df_voxel_scene = create_list_dfs_voxel_scene(base_dir, voxel_size, upto=None)
    pickle.dump(list_df_voxel_scene, open(pickles_path+"list_df_voxel_scene__%s__voxel_size_%d.p"%(vid, voxel_size), "wb"))


# ----------------------

# In[8]:


def get_list_idx_for_frame(frame_idx, n_frames_per_side, shift):
    list_idx = np.array(list(np.arange(frame_idx-shift-n_frames_per_side,frame_idx-shift)) + list(np.arange(frame_idx+shift+1,frame_idx+shift+n_frames_per_side+1)))
    return list_idx


# In[9]:


def create_scene_voxel_df(list_df_voxel_scene, list_idx, frac_to_drop=0.5):
    list_df_voxel_scene_for_frame = list_df_voxel_scene[list_idx]
    df_scene = pd.concat(list_df_voxel_scene_for_frame)#.drop_duplicates("voxel_id")
#     df_scene.groupby(["x_grid", "y_grid", "z_grid"]).value_counts()
    dfff = df_scene.groupby(["x_grid", "y_grid", "z_grid"]).apply(len).reset_index(drop=False)

    dfff = dfff[dfff[0]>frac_to_drop*len(list_idx)]
    dfff["v_index"] = dfff.iloc[:, :3].apply(lambda x: hash((x.iloc[0], x.iloc[1], x.iloc[2])), axis=1)
#     print(df_scene.shape)
    df_scene.drop_duplicates(["x_grid", "y_grid", "z_grid"], inplace=True)
#     print(df_scene.shape)
    df_scene = df_scene[df_scene.voxel_id.isin(dfff.v_index)]
#     print(df_scene.shape)
    
    return df_scene


# In[10]:


def get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift):
    
    list_idx_uf = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
    list_idx = [i for i in list_idx_uf if i>=0 and i<=list_df_voxel_scene.shape[0]]
#     print ('--get_df_voxel_sub--\nlist idx_uf: {}\nlist_idx   :{}\n-------'.format(list_idx_uf, list_idx))
    df_voxel_scene = create_scene_voxel_df(list_df_voxel_scene, list_idx)
    df_voxel_frame = create_scene_voxel_df(list_df_voxel_scene, [frame_idx])

    df_subtracted_frame = df_voxel_frame[~df_voxel_frame.voxel_id.isin(df_voxel_scene.voxel_id)]
    return df_subtracted_frame


# In[11]:


def point_to_voxel(ser, voxel_size):
    # ser: [x, y, z, r]
    ser_out = ser.iloc[:3]//voxel_size
    ser_out = ser_out*voxel_size + voxel_size//2
    voxel_id = hash((ser_out[0], ser_out[1], ser_out[2]))
    
#     return pd.Series(ser_out.tolist()+[voxel_id])
    return voxel_id


# In[12]:


def save_frame_for_movie(frame, folder, all_files, frame_idx):
    if not os.path.exists(folder):
        os.mkdir(folder)
    filename = os.path.basename(all_files[frame_idx])
    frame.to_csv(folder + filename, header=None, index=False)
    labels_filename = filename.replace('pointcloud', 'labels')
    pd.DataFrame([0]*frame.shape[0]).to_csv(folder + labels_filename, header=None, index=False)

    egomotion_filename = filename.replace('pointcloud', 'egomotion')
    pd.DataFrame([0.]*6).T.to_csv(folder + egomotion_filename, header=None, index=False)
    print('frame {} saved successfuly'.format(frame_idx))


# In[13]:


# frame_idx=15
# todo: add skips


# In[14]:


# list_idx = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
# list_idx


# In[15]:


# df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)


# In[16]:


# df_subtracted_frame.head()


# In[17]:


# df_frame_orig = pd.read_csv(all_files[frame_idx], header=None)


# In[18]:


# %%time
# df_frame_orig["voxel_id"] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)


# In[19]:


# df_frame_orig.head()


# In[20]:


# df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)
# df_labels.sum()


# In[21]:


# df_frame_orig_subtracted = df_frame_orig[df_labels]
# df_labels = df_labels.astype(int)


# In[22]:


# df_frame_orig_subtracted.shape


# In[23]:


# df_frame_orig_subtracted = df_frame_orig_subtracted.iloc[:4]


# In[24]:


# # 
# save_frame_for_movie(df_frame_orig_subtracted, "tmp_only_labeled/", all_files, frame_idx)


# In[25]:


def remove_distant_lines(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

#     dff = dff[dff.x>dff.x.min()+(dff.x.max()-dff.x.min())*0.2]
#     print(dff.shape)
#     dff.head()
    
    dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.3]
#     print(dff.shape)
#     dff.head()

    dff = dff[(dff["r"]>5)&(dff["r"]<15)]
#     print(dff.shape)
#     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[26]:


def remove_distant_points(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 
#     print(dff.describe())
    dff = dff[dff["x"]>5500]
#     print(dff.describe())
#     rr = np.sqrt(dff.x**2+dff.y**2)
#     dff = dff[rr<10]
#     print(dff.shape)
#     dff.head()
    
#     dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.3]
#     print(dff.shape)
#     dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
#     print(dff.shape)
#     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[27]:


def remove_high_theta_points(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

    theta = np.abs(np.arctan(dff.y/dff.x))
    dff = dff[theta>0.9*theta.max()]
    dff = dff[dff.x>300]
#     print(dff.shape)
#     dff.head()
    
#     dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.3]
#     print(dff.shape)
#     dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
#     print(dff.shape)
#     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[28]:


def remove_close_lines(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

    dff = dff[dff.x<dff.x.min()+(dff.x.max()-dff.x.min())*0.1]
#     print(dff.shape)
#     dff.head()
    
    dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.01]
#     print(dff.shape)
#     dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
# #     print(dff.shape)
# #     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[29]:


def remove_high_points(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

#     dff = dff[dff.x<dff.x.min()+(dff.x.max()-dff.x.min())*0.15]
#     print(dff.shape)
#     dff.head()
    
    dff = dff[dff.z>400]
    print(dff.shape)
    dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
# #     print(dff.shape)
# #     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# ---------------------------------

# In[31]:


# frame_idx=15
for frame_idx in tqdm(range(15,900)):
    print("=========================================================================")
    print("frame_idx = %d"%frame_idx)
    print("=========================================================================")

    tic = time()
    list_idx_uf = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
    list_idx = [i for i in list_idx_uf if i>=0 and i<=list_df_voxel_scene.shape[0]]
#     print ('--main--\nlist idx_uf: {}\nlist_idx   :{}\n-------'.format(list_idx_uf, list_idx))
    toc = time(); print(toc-tic, ": list_idx = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)"); tic=time()
    
    df_scene = create_scene_voxel_df(list_df_voxel_scene, list_idx)
    
    df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)
    toc = time(); print(toc-tic, ": df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)"); tic=time()
    
    df_frame_orig = pd.read_csv(all_files[frame_idx], header=None)
#     df_frame_orig["voxel_id"] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)
    df_frame_orig["voxel_id"] =  ((df_frame_orig.iloc[:, :3]//voxel_size)*voxel_size + voxel_size//2).apply(lambda x: hash((x[0], x[1], x[2])), axis=1)
    toc = time(); print(toc-tic, ": df_frame_orig['voxel_id'] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)"); tic=time()
    
    df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)
    toc = time(); print(toc-tic, ": df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)"); tic=time()
    
    # filters
    df_labels = remove_distant_lines(frame_idx, df_labels, orig_files_path)
    df_labels = remove_distant_points(frame_idx, df_labels, orig_files_path)
    df_labels = remove_close_lines(frame_idx, df_labels, orig_files_path)
    df_labels = remove_high_points(frame_idx, df_labels, orig_files_path)
    df_labels = remove_high_theta_points(frame_idx, df_labels, orig_files_path)
#     df_labels.iloc[:] = False
    
    df_frame_orig_subtracted = df_frame_orig[df_labels]
    toc = time(); print(toc-tic, ": df_frame_orig_subtracted = df_frame_orig[df_labels]"); tic=time()
    
    print(df_frame_orig_subtracted.shape)
    df_labels = df_labels.astype(int)
    file_labels = "voxelling_output/submission_files/vid_1_pred_lee/" + os.path.basename(all_files[frame_idx]).replace("pointcloud", "labels")
    print("file_labels:", file_labels)
    df_labels.to_csv(file_labels, header=None, index=False)
    toc = time(); print(toc-tic, ': df_labels.to_csv(pickles_path+"df_labels__frame_%d__%s__voxel_size_%d_(%d,%d,%d).p"%(frame_idx, vid, voxel_size, n_frames_per_side, shift, n_frames_per_side))'); tic=time()
    
    print(df_labels.sum())
    df_frame_orig_subtracted = df_frame_orig_subtracted.iloc[:, :4]
    df_frame_orig_subtracted.to_csv(pickles_path+"df_frame_orig_subtracted__frame_%d__%s__voxel_size%d_(%d,%d,%d).p"%(frame_idx, vid, voxel_size, n_frames_per_side, shift, n_frames_per_side)), header=None, index=False)
#     save_frame_for_movie(df_frame_orig_subtracted, "tmp_only_labeled/", all_files, frame_idx)
    toc = time(); print(toc-tic, ': save_frame_for_movie(df_frame_orig_subtracted, "tmp_only_labeled/", all_files, frame_idx)'); tic=time()


# In[ ]:


df_scene.head()


# In[ ]:


# frac_to_drop=0.5
# dfff = df_scene.groupby(["x_grid", "y_grid", "z_grid"]).apply(len).reset_index(drop=False)

# dfff = dfff[dfff[0]>frac_to_drop*len(list_idx)]
# dfff["v_index"] = dfff.iloc[:, :3].apply(lambda x: hash((x.iloc[0], x.iloc[1], x.iloc[2])), axis=1)
# print(df_scene.shape)
# df_scene.drop_duplicates(["x_grid", "y_grid", "z_grid"], inplace=True)
# print(df_scene.shape)
# df_scene = df_scene[df_scene.voxel_id.isin(dfff.v_index)]
# print(df_scene.shape)


# In[ ]:


# df_scene


# In[ ]:


# base_dir = "E:/Datasets/DataHack/Train"
# for sub_dir in os.listdir(base_dir):
#     print(sub_dir)
# #     for f in glob(sub_dir+"/*point*"):
# #     df = add_grid_and_hash(pd.read_csv(f), header=None), 20)


# --------------------

# In[ ]:


# # for road distant lines:
# df_frame_orig_subtracted.columns = list("xyzr")
# df_frame_orig_subtracted.plot(kind="scatter", x="x", y="y", figsize=(10, 10))


# In[ ]:


# dff = pd.read_csv("voxelling_output/Original_Frames/0000033_pointcloud.csv", header=None)
# dff.columns = list("xyzr")
# dff["l"] = df_labels.tolist()
# dff = dff[dff.l==1]
# print(dff.shape)
# dff.head()


# In[ ]:


# dff = dff[dff.x>dff.x.min()+(dff.x.max()-dff.x.min())*0.4]
# print(dff.shape)
# dff.head()


# In[ ]:


# dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.3]
# print(dff.shape)
# dff.head()


# In[ ]:


# dff = dff[(dff["r"]>5)&(dff["r"]<15)]
# print(dff.shape)
# dff.head()


# In[ ]:


# dff.plot(kind="scatter", x="x", y="y", figsize=(10, 10))


# In[ ]:


# (df_frame_orig_subtracted.iloc[:, :2]//2*2).plot(kind="scatter", x=0, y=1)


# In[ ]:


ddf = df_frame_orig_subtracted.iloc[:, :2]//2*2


# In[ ]:


ddf = (ddf - ddf.min())
ddf = np.floor((ddf/ddf.max()*256).values)
ddf


# In[ ]:


import cv2
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)


# In[ ]:


# def pc_to_grid(df, voxel_size, dest_dir, dest_name):
#     # df_frame points are in world coordinate system
#     # dest_name may be frame number
#     start_time = time()
# #     df = pd.read_csv(csv_path, header=None)

#     df_xyz = df.iloc[:,:3]
#     ref_ser = df.iloc[:, 3]
#     df_grid = df_xyz//voxel_size
#     df_grid[3] = ref_ser
#     df_grid.columns=list("xyzr")
#     df_voxel = df_grid.groupby(['x','y','z']).apply(lambda x: x.iloc[:, 3].mean()).to_frame()
#     df_voxel.reset_index(drop=False, inplace=True)
#     df_voxel.iloc[:, :3] = df_voxel.iloc[:, :3]*voxel_size + voxel_size//2
#     df_voxel= df_voxel.astype(np.int)
#     base_name = os.path.join(dest_dir, dest_name)
#     df_voxel.to_csv(base_name + '_pointcloud.csv', header=None, index=False)
#     pd.DataFrame([0]*df_voxel.shape[0]).to_csv(base_name+'_labels.csv', header=None, index=False)
#     pd.DataFrame([0.]*6).T.to_csv(base_name+'_egomotion.csv', header=None, index=False)
#     print('single file runtime: {}'.format(time()-start_time))
    
#     return df_voxel


# In[ ]:


# world_dir = "E:/Datasets/DataHack/World/Train/vid_1/"


# In[ ]:


# scene_frames = glob(world_dir+"/*point*")[:10] + glob(world_dir+"/*point*")[20:30]
# cur_frame = glob(world_dir+"/*point*")[15]


# In[ ]:


# s_pcs = []
# for f in scene_frames:
#     s_pcs.append(pd.read_csv(f, header=None))
# df_scene_pcs = pd.concat(s_pcs, axis=0)


# In[ ]:


# %%time
# try:
#     os.mkdir("tmp/")
# except:
#     pass
# pc_to_grid(df_scene_pcs, 20, "tmp", "0000000")


# In[ ]:


# %%time
# df_cur_pc = pd.read_csv(cur_frame, header=None)
# pc_to_grid(df_cur_pc, 20, "tmp", "0000001")


# In[ ]:


# w = pd.read_csv("tmp/0000000_pointcloud.csv", header=None)
# c = pd.read_csv("tmp/0000001_pointcloud.csv", header=None)


# In[ ]:


# w['voxel_id'] = w.apply(lambda x: hash((x[0], x[1], x[2])), axis=1)
# c['voxel_id'] = c.apply(lambda x: hash((x[0], x[1], x[2])), axis=1)


# In[ ]:


# c_out = c[~c.voxel_id.isin(w.voxel_id)]


# In[ ]:


# c.shape, c_out.shape


# In[ ]:


# c_out = c_out.iloc[:, :4]
# c_out.head()


# In[ ]:


# try:
#     os.mkdir("tmp_out/")
# except:
#     pass
# pc_to_grid(c_out, 20, "tmp_out", "0000002")


# In[ ]:


# list_df_voxels = []
# for f in g


# In[ ]:


# def voxel_classification_for_frame(vid_dir, frame_idx):
#     df_pc_scene = ...
#     df_pc_frame = ...
#     df_voxel_scene = build_voxel(df_pc_scene)
#     df_voxel_frame = build_voxel()
    
    

