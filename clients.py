import torch
from torch.optim import Adam
from models import *
import torch.distributed as dist

class ClientFedAvg():
    def __init__(self,args):
        self.rank = args["rank"]
        self.world_size = args["world_size"]
        self.train_type = args["train_type"]
        self.communication_type = args["communication_type"]
        self.batch_size = args["batch_size"]
        self.device = args["device"]
        self.model = args["Model"](model_type=self.train_type,communication_type=self.communication_type).to(self.device)
        self.optimizer = args["Optimizer"](self.model.parameters())
        self.loss_function = args["Loss_function"]
        self.train_dataloader = args["Trainloader"]
        self.test_dataloader = args["Testloader"]

        self.n_test_samples = len(self.test_dataloader)*self.batch_size

    def train(self,epochs):
        for _ in range(epochs):
            self.broadcast()

            self.epoch_loss, self.acc = 0.0, 0.0
            for x,y in self.test_dataloader:
                x,y = x.to(self.device),y.to(self.device)
                with torch.no_grad():
                    y_pred = self.model(x)
                    loss = self.loss_function(y, y_pred)
                self.epoch_loss += loss.item()
                self.acc += torch.sum(y_pred.argmax(axis=1) == y.argmax(axis=1))
            self.acc /= self.n_test_samples
            self.epoch_loss /= self.n_test_samples

            for x,y in self.train_dataloader:
                x,y = x.to(self.device),y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_function(y, y_pred)
                loss.backward()
                self.optimizer.step()

            self.aggregate()

    def aggregate(self):
        param = flatten(self.model).clone().detach().cpu()
        self.epoch_loss=torch.tensor([self.epoch_loss])
        self.acc = torch.tensor([self.acc]).clone().detach().cpu()
        param = torch.cat((self.epoch_loss,self.acc,param),dim=0)
        dist.send(param,dst=self.world_size-1)

    def broadcast(self):
        param = flatten(self.model).clone().detach().cpu()
        dist.recv(param,src=self.world_size-1)
        unflatten(self.model,param.to(self.device))