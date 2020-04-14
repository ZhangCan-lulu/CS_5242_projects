
import torch
import torch.nn as nn
import numpy as np



def predict(model, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, args):
    mapping_file = args.path_data + args.dataset + "/mapping.txt"  # mapping between classes & indices
    coarse_file = args.path_data + args.dataset + "/coarse_label"
    gt_path = args.path_data + args.dataset + "/groundTruth/"

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]  # list of classes
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    ###
    file_co_ptr = open(coarse_file, 'r')
    coarse_actions = file_co_ptr.read().split('\n')[:-1]  # list of classes
    file_co_ptr.close()
    coa_actions_dict = dict()
    for c_a in coarse_actions:
        coa_actions_dict[c_a.split()[1]] = int(c_a.split()[0])


    # collect arguments
    verbose = args.verbose
    use_best_model = args.use_best_model

    # multi-GPU
    # if args.multi_gpu and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        model.to(device)
        if use_best_model == 'source':
            # model.load_state_dict(torch.load(model_dir + "/acc_best_source.model"))
            if args.multi_gpu and torch.cuda.device_count() > 1:
                pretained_param = torch.load(model_dir + "/acc_best_source.model")
                model_dict = model.state_dict()
                updated_params = {k[7:]: v for k, v in pretained_param.items() if k[7:] in model_dict.keys()}
                model.load_state_dict(updated_params)
            else:
                model.load_state_dict(torch.load(model_dir + "/acc_best_source.model"))

            # model.load_state_dict(torch.load("models/breakfast/split_6/acc_best_target.model"))
        elif use_best_model == 'target':
            # model.load_state_dict(torch.load(model_dir + "/acc_best_target.model"))
            #model.load_state_dict(torch.load("models/breakfast/split_6new_noatt_coarse_onehot/epoch-25.model"))
            if args.multi_gpu and torch.cuda.device_count() > 1:
                pretained_param = torch.load(model_dir + "/acc_best_target.model")
                model_dict = model.state_dict()
                updated_params = {k[7:]: v for k, v in pretained_param.items() if k[7:] in model_dict.keys()}
                model.load_state_dict(updated_params)
            else:
                model.load_state_dict(torch.load(model_dir + "/acc_best_target.model"))
        else:
            if args.multi_gpu and torch.cuda.device_count() > 1:
                pretained_param = torch.load(model_dir + "/epoch-" + str(epoch) + ".model")
                model_dict = model.state_dict()
                updated_params = {k[7:]: v for k, v in pretained_param.items() if k[7:] in model_dict.keys()}
                model.load_state_dict(updated_params)
            else:
                model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]  # testing list
        file_ptr.close()
        for vid in list_of_vids:
            if verbose:
                print(vid)

            # features = np.load(features_path + vid.split('.')[0] + '.npy')
            # features = features[:, ::sample_rate]
            #
            # if args.use_onehot:
            #     coarse_app = np.zeros(shape=[len(coa_actions_dict), features.shape[1]], dtype=np.float32)
            #     coarse_label = vid.split(".")[0].split("_")[-1]
            #     coarse_logit = coa_actions_dict[coarse_label]
            #
            #     coarse_app[coarse_logit, :] = 1
            #     bound_app = np.zeros(shape=[1, features.shape[1]], dtype=np.float32)
            #     # if flag!="target":
            #
            #     # bound_app[:, np.array(seg_ind)] = 1
            #     features = np.concatenate((features, coarse_app), axis=0)
            #     features = np.concatenate((features, bound_app), axis=0)

            features = np.load(features_path + vid.split('.')[0] + '.npy')  # dim: 2048 x frame#
            file_ptr = open(gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]  # ground truth (in words)
            classes = np.zeros(min(np.shape(features)[1], len(content)))  # ground truth (in indices)
            ##
            coarse_label = vid.split(".")[0].split("_")[-1]
            coarse_class = np.zeros(min(np.shape(features)[1], len(content)))
            ##
            seg_ind = []
            pre_logit, cur_logit = 0, 0
            for i in range(len(classes)):
                classes[i] = actions_dict[content[i]]
                coarse_logit = coa_actions_dict[coarse_label]
                coarse_class[i] = coarse_logit
                cur_logit = classes[i]
                if cur_logit != pre_logit:
                    seg_ind.append(i)
                    pre_logit = cur_logit
            if args.use_onehot:
                coarse_app = np.zeros(shape=[len(coa_actions_dict), features.shape[1]], dtype=np.float32)
                # if flag!="target":
                coarse_app[coarse_logit, :] = 1
                features = np.concatenate((features, coarse_app), axis=0)

            if args.boundary_on:
                bound_app = np.zeros(shape=[1, features.shape[1]], dtype=np.float32)
                # if flag!="target":
                bound_app[:, np.array(seg_ind)] = 1
                features = np.concatenate((features, bound_app), axis=0)

            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            mask = torch.ones_like(input_x)
            predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ ,_,_= model(input_x, input_x, mask, mask, [0, 0], reverse=False)
            _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
            predicted = predicted.squeeze()
            recognition = []
            for i in range(predicted.size(0)):
                recognition = np.concatenate((recognition,
                    [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(results_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
