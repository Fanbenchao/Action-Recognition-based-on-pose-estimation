import scipy.io as scio
import numpy as np
data_path = u"E:\MPIIdataset\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat"
save_path = u'E:\Graduation Project\Codes\Burt Fan\data\mpii'
data = scio.loadmat(data_path)
annolist = data["RELEASE"]["annolist"]
img_train = data["RELEASE"]["img_train"]
mask_train = img_train[0][0][0]==1
mask_test = ~mask_train
train_valid = annolist[0][0][0][mask_train]
test = annolist[0][0][0][mask_test]
train_valid_process = []
lack_annorect = 0
lack_head_box = 0
lack_annopoints = 0
lack_scale = 0
lack_objpos = 0
lack_is_visible = 0
record = -1
error = []
for annot in train_valid:
    record +=1
    annot_str = str(annot.dtype)
    annorect_str = str(annot["annorect"].dtype)
    if 'annorect' not in annot_str:
        lack_annorect +=1
        error.append(record)
        continue
    if 'x1' not in annorect_str or 'x2' not in annorect_str:
        lack_head_box +=1
        error.append(record)
        continue
    if "annopoints" not in annorect_str:
        lack_annopoints +=1
        error.append(record)
        continue
    if "scale" not in annorect_str:
        lack_scale +=1
        error.append(record)
        continue
    if "objpos" not in annorect_str:
        lack_objpos +=1
        error.append(record)
        continue
    name = annot["image"][0]["name"][0][0]
    x1,y1 = annot["annorect"][0]['x1'],annot["annorect"][0]['y1']
    x2,y2 = annot["annorect"][0]['x2'],annot["annorect"][0]['y2']
    for person_num in range(len(x1)):
        if annot["annorect"][0]['scale'][person_num].shape != (1,1):
            lack_scale +=1
            continue
        if 'is_visible' not in str(annot['annorect'][0]['annopoints'][person_num]['point'][0][0].dtype):
            lack_is_visible +=1
            continue
        info_dic = {}
        info_dic['name'] = name+str(person_num)
        info_dic['head_box'] = [x1[person_num][0][0],y1[person_num][0][0],x2[person_num][0][0],y2[person_num][0][0]]
        info_dic['scale'] = annot["annorect"][0]['scale'][person_num][0][0]
        x,y = annot["annorect"][0]['objpos'][person_num][0]['x'][0][0][0],annot["annorect"][0]['objpos'][person_num][0]['y'][0][0][0]
        info_dic['objpos'] = [x,y]
        ids = []
        for joints_id in annot['annorect'][0]['annopoints'][person_num]['point'][0][0]['id'][0]:
            ids.append(joints_id[0][0])
        joints = [[-1,-1] for i in range(16)]
        is_visible = [-1 for i in range(16)]
        record_num = -1
        for index in ids:
            record_num +=1
            joints[index] = [annot['annorect'][0]['annopoints'][person_num]['point'][0][0]['x'][0][record_num][0][0],\
                             annot['annorect'][0]['annopoints'][person_num]['point'][0][0]['y'][0][record_num][0][0]]
            if  annot['annorect'][0]['annopoints'][person_num]['point'][0][0]['is_visible'][0][record_num].shape != (1,1):
                is_visible[index] = -1
            else:
                is_visible[index] = annot['annorect'][0]['annopoints'][person_num]['point'][0][0]['is_visible'][0][record_num][0][0]
        info_dic['joints_id'] = ids
        info_dic['joints'] = joints;
        info_dic['is_visible'] = is_visible 
        bounding_box = [[] for i in range(4)]
        joints_min = np.copy(joints)
        joints_max = np.copy(joints)
        for i in range(16):
            if joints_min[i][0] == -1:
                joints_min[i] = [1e5,1e5]
        bounding_box[0],bounding_box[1] = min(joints_min[:,0]),min(joints_min[:,1])
        bounding_box[2],bounding_box[3] = max(joints_max[:,0]),max(joints_max[:,1])
        info_dic['box'] = bounding_box
        train_valid_process.append(info_dic)
np.save(save_path+'\\train_valid_process.npy',train_valid_process)