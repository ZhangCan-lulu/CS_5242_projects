
import torch
import numpy as np
import random
import torch.utils.data as data_loader

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict,coa_actions_dict, gt_path, features_path, sample_rate,device):
        self.list_of_examples = list()
        self.num_examples = 0
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.coa_actions_dict=coa_actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate


    def reset(self):
        self.index = 0

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        self.num_examples = len(self.list_of_examples)
        file_ptr.close()
        random.shuffle(self.list_of_examples)
    # def __getitem__(self, item):
    #     self.index=item

    def next_batch(self, batch_size, flag,coarse_on=True,boundary_on=False):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        # for re-loading target data
        if flag == 'target' and self.index == len(self.list_of_examples):
            self.reset()

        batch_input = []
        batch_target = []
        batch_coa_target=[]
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')  # dim: 2048 x frame#
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]  # ground truth (in words)
            classes = np.zeros(min(np.shape(features)[1], len(content)))  # ground truth (in indices)
            ##
            coarse_label = vid.split(".")[0].split("_")[-1]
            coarse_class=np.zeros(min(np.shape(features)[1], len(content)))
            ##
            seg_ind = []
            pre_logit, cur_logit = 0, 0
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
                coarse_logit=self.coa_actions_dict[coarse_label]
                coarse_class[i]=coarse_logit
                cur_logit = classes[i]
                if cur_logit != pre_logit:
                    seg_ind.append(i)
                    pre_logit = cur_logit
            if coarse_on:

                coarse_app = np.zeros(shape=[len(self.coa_actions_dict), features.shape[1]], dtype=np.float32)
                # if flag!="target":
                coarse_app[coarse_logit,:]=1
                features=np.concatenate((features,coarse_app),axis=0)

            if boundary_on:
                bound_app = np.zeros(shape=[1, features.shape[1]], dtype=np.float32)
                # if flag!="target":
                bound_app[:, np.array(seg_ind)] = 1
                features = np.concatenate((features, bound_app), axis=0)


            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

            batch_coa_target.append(coarse_class[::self.sample_rate])###

        num_gpus=1
        if torch.cuda.is_available():
            num_gpus=torch.cuda.device_count()



        length_of_sequences = list(map(len, batch_target))  # frame#
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # if different length, pad w/ zeros
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)#*(-100)
        batch_coa_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)# * (-100).cuda()#####
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)  # zero-padding for shorter videos
        for i in range(len(batch_input)):

            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            batch_coa_target_tensor[i, :np.shape(batch_coa_target[i])[0]] = torch.from_numpy(batch_coa_target[i])####
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        # for i in range(num_gpus):
        #
        #     seg_ind=len(batch_input)//num_gpus
        #     batch_input_tensor[:seg_ind,:,:].to(torch.device("cuda:0"))
        #     batch_input_tensor[seg_ind:,:,:].to(torch.device("cuda:1"))

        return batch_input_tensor, batch_target_tensor,batch_coa_target_tensor, mask
#

class train_generator(data_loader.Dataset):
    def __init__(self,num_classes, actions_dict,coa_actions_dict, gt_path, features_path,vid_file, sample_rate,coarse_on=True,boundary_on=True):
        self.list_of_examples = list()
        self.num_examples = 0
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.coa_actions_dict=coa_actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.vid_file=self.read_data(vid_file)
        self.coarse_on=coarse_on
        self.boundary_on=boundary_on

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        self.num_examples = len(self.list_of_examples)
        file_ptr.close()
        random.shuffle(self.list_of_examples)
    def __len__(self):
        return len(self.list_of_examples)

    def __getitem__(self, item):
        vid=self.list_of_examples[item]
        features = np.load(self.features_path + vid.split('.')[0] + '.npy')  # dim: 2048 x frame#
        file_ptr = open(self.gt_path + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]  # ground truth (in words)
        classes = np.zeros(min(np.shape(features)[1], len(content)))  # ground truth (in indices)
        ##
        coarse_label = vid.split(".")[0].split("_")[-1]
        coarse_class = np.zeros(min(np.shape(features)[1], len(content)))
        ##
        seg_ind=[]
        pre_logit,cur_logit=0,0
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
            coarse_logit = self.coa_actions_dict[coarse_label]
            coarse_class[i] = coarse_logit

            cur_logit=classes[i]
            if cur_logit!=pre_logit:
                seg_ind.append(i)
                pre_logit=cur_logit

        if self.coarse_on:
            coarse_app = np.zeros(shape=[len(self.coa_actions_dict), features.shape[1]], dtype=np.float32)

            coarse_app[coarse_logit, :] = 1
            features = np.concatenate((features, coarse_app), axis=0)
        if self.boundary_on:
            bound_app=np.zeros(shape=[1, features.shape[1]], dtype=np.float32)
            bound_app[:,np.array(seg_ind)]=1
            features = np.concatenate((features, bound_app), axis=0)

        return features,classes,coarse_class


    def collen_fn(self,batch):
        # features,logits=zip(*batch)
        batch_input = []
        batch_target = []
        batch_coa_target = []

        for features, logits,coarse_logits in zip(*batch):
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(logits[::self.sample_rate])
            batch_coa_target.append(coarse_logits[::self.sample_rate])  ###
        length_of_sequences = list(map(len,  batch_target))  # frame#
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)  # if different length, pad w/ zeros
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        batch_coa_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (
            -100)  #####
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences),
                           dtype=torch.float)  # zero-padding for shorter videos
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            batch_coa_target_tensor[i, :np.shape(batch_coa_target[i])[0]] = torch.from_numpy(batch_coa_target[i])  ####
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, batch_coa_target_tensor, mask


class test_generator(data_loader.Dataset):
    def __init__(self, num_classes, actions_dict, coa_actions_dict, gt_path, features_path, vid_file, sample_rate,
                 coarse_on=True, boundary_on=True):
        self.list_of_examples = list()
        self.num_examples = 0
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.coa_actions_dict = coa_actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.vid_file = self.read_data(vid_file)
        self.coarse_on = coarse_on
        self.boundary_on = boundary_on

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        self.num_examples = len(self.list_of_examples)
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def __len__(self):
        return len(self.list_of_examples)

    def __getitem__(self, item):
        vid = self.list_of_examples[item]
        features = np.load(self.features_path + vid.split('.')[0] + '.npy')  # dim: 2048 x frame#
        file_ptr = open(self.gt_path + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]  # ground truth (in words)
        classes = np.zeros(min(np.shape(features)[1], len(content)))  # ground truth (in indices)
        ##
        coarse_label = vid.split(".")[0].split("_")[-1]
        coarse_class = np.zeros(min(np.shape(features)[1], len(content)))
        ##
        seg_ind = []
        pre_logit, cur_logit = 0, 0
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
            coarse_logit = self.coa_actions_dict[coarse_label]
            coarse_class[i] = coarse_logit

            cur_logit = classes[i]
            if cur_logit != pre_logit:
                seg_ind.append(i)
                pre_logit = cur_logit

        if self.coarse_on:
            coarse_app = np.zeros(shape=[len(self.coa_actions_dict)+1, features.shape[1]], dtype=np.float32)

            coarse_app[coarse_logit, :] = 1
            features = np.concatenate((features, coarse_app), axis=0)
        # if self.boundary_on:
        #     bound_app = np.zeros(shape=[1, features.shape[1]], dtype=np.float32)
        #     bound_app[:, np.array(seg_ind)] = 1
        #     features = np.concatenate((features, bound_app), axis=0)

        return features, classes,coarse_class

    def collen_fn(self, batch):
        # features,logits=zip(*batch)
        batch_input = []
        batch_target = []
        batch_coa_target = []

        for features, logits,coarse_logits in zip(*batch):
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(logits[::self.sample_rate])
            batch_coa_target.append(coarse_logits[::self.sample_rate])  ###
        length_of_sequences = list(map(len, batch_target))  # frame#
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)  # if different length, pad w/ zeros
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        batch_coa_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (
            -100)  #####
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences),
                           dtype=torch.float)  # zero-padding for shorter videos
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            batch_coa_target_tensor[i, :np.shape(batch_coa_target[i])[0]] = torch.from_numpy(batch_coa_target[i])  ####
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, batch_coa_target_tensor, mask


