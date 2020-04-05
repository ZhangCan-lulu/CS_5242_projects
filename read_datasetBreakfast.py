# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:58:28 2019

@author: fame
""" 
import os  
import torch
import numpy as np
import os.path 
import pandas as pd
 
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def generate_pre(split_load, actions_dict, GT_folder, DATA_folder):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
    all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake',
                 'salat']
    num_count = 0
    data_breakfast = []
    labels_breakfast = []
    submisson_gt, submisson,loc = [], [],[]
    ID,ID_pd= 0,0
    for content in content_all:
        file_ptr = open(GT_folder + content, 'r')
        curr_gt = file_ptr.read().split('\n')[:-1]

        file_prd = open(DATA_folder + content.split('.')[0], 'r')
        curr_prd = file_prd.read().split('\n')[-1].split(' ')
        data_breakfast.append(curr_prd)
        label_curr= []

        ini_st = actions_dict[curr_gt[0]]
        ini_pdst = actions_dict[curr_prd[0]]
        for iik in range(1, len(curr_gt)):
            cur_lb = actions_dict[curr_gt[iik]]
            cur_pdlb=actions_dict[curr_prd[iik]]
            if cur_lb != ini_st and cur_lb != 0:
                label_curr.append(iik)
                submisson_gt.append([ID, cur_lb])
                ID += 1
                ini_st = cur_lb
            if cur_lb != ini_st and cur_lb==0:
                label_curr.append(iik)
                # ini_st=0
                break
        if actions_dict[curr_gt[-1]]!=0:
            label_curr.append(len(curr_gt))


        loc.append(label_curr)

    for ind,loc_ind in enumerate(loc):
        start=loc_ind[0]
        brk_data=data_breakfast[ind]
        for iind in loc_ind[1:]:
            labels=brk_data[start:iind]
            labels_num=[actions_dict[da] for da in labels]
            prediction=np.argmax(np.bincount(labels_num))
            submisson.append([ID_pd, prediction])
            ID_pd += 1
            start=iind

    #
    # if cur_pdlb != ini_pdst and cur_pdlb != 0:
    #     submisson.append([ID_pd, cur_pdlb])
    #     ID_pd += 1
    #     ini_pdst = cur_pdlb

    #         label_curr_video.append(actions_dict[curr_gt[iik]])
    #     labels_breakfast.append(label_curr_video)
    #
    #
    # labels_uniq, labels_uniq_loc = get_label_bounds(labels_breakfast)
    submission = pd.DataFrame(submisson, columns=["Id", "Category"])
    submission.to_csv('submission.csv', index=False)
    submission_gt = pd.DataFrame(submisson_gt, columns=["Id", "Category"])
    submission_gt.to_csv('submission_gt.csv', index=False)

    acc=0
    for v_id,video in enumerate(submisson_gt):
        if video[1]==submisson[v_id][1]:
            acc+=1
    acc=acc/len(submisson)
    print("Acc is: {}".format(acc))
    # for lb in labels_uniq:
    #     num_count += len(lb)
    print("Finish Load the Test data!!!")

    return acc

def load_data(split_load, actions_dict, GT_folder, DATA_folder, datatype = 'training',):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]
    all_tasks = ['tea', 'cereals', 'coffee', 'friedegg', 'juice', 'milk', 'sandwich', 'scrambledegg', 'pancake', 'salat']
    num_count=0
    if datatype == 'training':
        data_breakfast = []
        labels_breakfast = []
        for content in content_all:
        
            file_ptr = open( GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]
            label_seq, length_seq = get_label_length_seq(curr_gt)

            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'
        
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            label_curr_video = []
            for iik in range(len(curr_gt)):
                label_curr_video.append( actions_dict[curr_gt[iik]] )
         
            data_breakfast.append(torch.tensor(curr_data,  dtype=torch.float64 ) )
            labels_breakfast.append(label_curr_video )
    
        labels_uniq, labels_uniq_loc = get_label_bounds(labels_breakfast)
        print("Finish Load the Training data and labels!!!")     
        return  data_breakfast, labels_uniq
    if datatype == 'test':
        data_breakfast = []
        labels_breakfast = []
        submisson_gt,submisson=[],[]
        ID=0
        for content in content_all:
            file_ptr = open(GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]

            label_curr_video = []
            ini_st=actions_dict[curr_gt[0]]
            for iik in range(1,len(curr_gt)):
                cur_lb=actions_dict[curr_gt[iik]]
                if cur_lb!=ini_st and cur_lb!=0:
                    submisson.append([ID,cur_lb])
                    ID+=1
                    ini_st=cur_lb

                label_curr_video.append(actions_dict[curr_gt[iik]])
            labels_breakfast.append(label_curr_video)

            # loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'
        
            # curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            
            # data_breakfast.append(torch.tensor(curr_data,  dtype=torch.float64 ) )
        labels_uniq, labels_uniq_loc = get_label_bounds(labels_breakfast)
        for lb in labels_uniq:
            num_count+=len(lb)
        print("Finish Load the Test data!!!")
        return data_breakfast


def get_label_bounds( data_labels):
    labels_uniq = []
    labels_uniq_loc = []
    for kki in range(0, len(data_labels) ):
        uniq_group, indc_group = get_label_length_seq(data_labels[kki])
        labels_uniq.append(uniq_group[1:-1])
        labels_uniq_loc.append(indc_group[1:-1])
    return labels_uniq, labels_uniq_loc

def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    return label_seq, length_seq


def get_maxpool_lstm_data(cData, indices):
    list_data = []
    for kkl in range(len(indices)-1):
        cur_start = indices[kkl]
        cur_end = indices[kkl+1]
        if cur_end > cur_start:
            list_data.append(torch.max(cData[cur_start:cur_end,:],
                                       0)[0].squeeze(0))
        else:
            list_data.append(torch.max(cData[cur_start:cur_end+1,:],
                                       0)[0].squeeze(0))
    list_data  =  torch.stack(list_data)
    return list_data

def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]

    actions_dict=dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

if __name__ == "__main__":
    COMP_PATH = ''
    split = 'training'
    #split = 'test'
    train_split =  os.path.join(COMP_PATH, 'splits/train.split1.bundle')
    test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle')
    GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/')
    DATA_folder =  os.path.join(COMP_PATH, 'data/')
    mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')
    
    
  
    actions_dict = read_mapping_dict(mapping_loc)
    if  split == 'training':
        data_feat, data_labels = load_data( test_split, actions_dict, GT_folder, DATA_folder, datatype = split)
    if  split == 'test':
        data_feat = load_data( test_split, actions_dict, GT_folder, DATA_folder, datatype = split)
    
        
 


