import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset,DataLoader

def create_purchase100(num_clients=10):
    np.random.seed(1234)

    dataset_path = "data/dataset_purchase"
    with open(dataset_path, "r") as f:
        purchase_dataset = f.readlines()
    x, y = [], []
    for datapoint in purchase_dataset:
        split = datapoint.rstrip().split(",")
        label = int(split[0]) - 1  # The first value is the label
        features = np.array(split[1:], dtype=np.float32)  # The next values are the features
        x.append(features)
        y.append(label)

    x = np.array(x, dtype=np.float32)
    y_onehot = OneHotEncoder(sparse=False).fit_transform(np.expand_dims(y, axis=1))

    xs,ys = [[] for _ in range(num_clients)],[[] for _ in range(num_clients)]
    for index in range(x.shape[0]):
        xs[y[index]%num_clients].append(x[index])
        ys[y[index]%num_clients].append(y_onehot[index])

    train_datasets_list = []
    for x_i,y_i in zip(xs,ys):
        x_i = np.array(x_i, dtype=np.float32)
        y_i = np.array(y_i, dtype=np.float32)
        train_datasets_list.append(train_test_split(x_i, y_i, test_size=0.25, random_state=1234))

    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    if not os.path.exists("./datasets/train"):
        os.mkdir("./datasets/train")
        
    for i,dataset in enumerate(train_datasets_list):
        if not os.path.exists("./datasets/train/%d"%(i)):
            os.mkdir("./datasets/train/%d"%(i))
        np.save("./datasets/train/%d/train_x.npy"%(i),dataset[0])
        np.save("./datasets/train/%d/train_y.npy"%(i),dataset[2])
        np.save("./datasets/train/%d/test_x.npy"%(i),dataset[1])
        np.save("./datasets/train/%d/test_y.npy"%(i),dataset[3])

def load_purchase100(file_path="./datasets/train/",dataset_type="train",rank=0,batch_size=32):
    x = np.load(file_path+"%d/%s_x.npy"%(rank,dataset_type))
    y = np.load(file_path+"%d/%s_y.npy"%(rank,dataset_type))
    dataset = NormalDataset(x,y)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return dataloader

class NormalDataset(Dataset):
    def __init__(self,x,y):
        super(NormalDataset,self).__init__()
        self.x = x
        self.y = y
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.x.shape[0]