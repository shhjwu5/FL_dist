from models import *
import torch.distributed as dist
from tqdm import tqdm

class ServerFedAvg():
    def __init__(self,args):
        self.rank = args["rank"]
        self.world_size = args["world_size"]
        self.train_type = args["train_type"]
        self.communication_type = args["communication_type"]
        self.model = args["Model"](model_type=self.train_type,communication_type=self.communication_type)
        self.loss_function = args["Loss_function"]
        self.num_clients = args["world_size"]-1

    def aggregate(self):
        self.param = flatten(self.model)
        self.param.zero_()
        self.loss = torch.tensor([0.0])
        self.accuracy = torch.tensor([0.0])
        self.param = torch.cat((self.loss,self.accuracy,self.param),dim=0)
        param = self.param.clone()
        for i in range(self.world_size-1):
            dist.recv(param,src=i)
            self.param.add_(param)
        # dist.reduce(self.param,dst=self.world_size-1)
        self.loss = self.param[0]
        self.accuracy = self.param[1]
        self.param.div_(self.world_size-1)
        unflatten(self.model,self.param[2:])

    def broadcast(self):
        self.param = flatten(self.model)
        for i in range(self.world_size-1):
            dist.send(self.param,dst=i)
        #dist.broadcast(self.param,src=self.world_size-1)

    def train(self,epochs):
        iterator = tqdm(range(epochs))
        for epoch in iterator:
            self.broadcast()
            self.aggregate()
            iterator.desc = f"Epoch {epoch:d} Loss {self.loss:.3f} Acc {self.accuracy:.3f}"