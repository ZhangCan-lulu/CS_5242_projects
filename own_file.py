import os
import glob

file_ptr =glob.glob("/home/zhangc/Downloads/segmentation_coarse/*")# open("/home/zhangc/data/preject_data/SSTDA_data/Datasets/action-segmentation/breakfast/splits/train.split5.bundle", 'r')
save_p=open("/home/zhangc/Downloads/segmentation_coarse/coarse_label",'w')

#actions=# = file_ptr.read().split('\n')[1:-1]  # list o# f classes

for ind,file in enumerate(file_ptr):
    save_p.write(str(ind)+' '+file.split("/")[-1])
    save_p.write("\n")


save_p.close()