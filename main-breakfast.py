import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')

args = parser.parse_args()
 
num_epochs = 120

lr = 0.0005
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    
if args.dataset == 'breakfast':
    lr = 0.0001

base_data_path = "/home/mw/input/breakfast64726472/breakfast/breakfast"

# 使用 os.path.join 构建路径
vid_list_file = os.path.join(base_data_path, "splits", f"train.split{args.split}.bundle")
vid_list_file_tst = os.path.join(base_data_path, "splits", f"test.split{args.split}.bundle")
features_path = os.path.join(base_data_path, "features")
gt_path = os.path.join(base_data_path, "groundTruth")
mapping_file = os.path.join(base_data_path, "mapping.txt")

# 确保目录路径以斜杠结尾
features_path = features_path + "/" if not features_path.endswith("/") else features_path
gt_path = gt_path + "/" if not gt_path.endswith("/") else gt_path

# 模型和结果目录（在本地）
model_dir = os.path.join(".", args.model_dir, args.dataset, f"split_{args.split}")
results_dir = os.path.join(".", args.result_dir, args.dataset, f"split_{args.split}")

# 创建输出目录
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 验证输入文件存在
required_files = [mapping_file, vid_list_file, vid_list_file_tst]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print("错误: 以下文件不存在:")
    for f in missing_files:
        print(f"  {f}")
    exit(1)

# 读取映射文件
with open(mapping_file, 'r') as file_ptr:
    actions = file_ptr.read().split('\n')[:-1]
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

