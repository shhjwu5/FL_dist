import torch.nn as nn
import torch

class Multiply(nn.Module):
    def __init__(self):
        super(Multiply,self).__init__()
        self.weight = torch.rand((1,100))

    def forward(self,x):
        x.mul_(self.weight)
        return x
    
class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate,self).__init__()
        
    def forward(self,x):
        return torch.cat(x,dim=1)

class ExampleNet(nn.Module):
    def __init__(self,model_type,communication_type):
        super(ExampleNet,self).__init__()
        self.model_type = model_type
        self.communication_type = communication_type
        
        self.fe1 = nn.Sequential(
            nn.Linear(in_features=600, out_features=1024),
            nn.Tanh(),
        )
        self.fe2 = nn.Sequential(
            nn.Linear(in_features=1024,out_features=128),
            nn.Tanh(),
        )
        self.pd1 = nn.Sequential(
            nn.Linear(in_features=1024,out_features=100),
        )
        self.pd2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=100),
        )
        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )
        # self.concatenate = Concatenate()
        # self.mix = nn.Sequential(
        #     nn.Linear(in_features=200,out_features=100),
        # )
        self.multiply1 = Multiply()
        self.multiply2 = Multiply()
        
        if self.communication_type == "shallow":
            self.communication_layers = [self.fe1,self.pd1]
        elif self.communication_type == "deep":
            self.communication_layers = [self.fe1,self.fe2,self.pd2]
        else:
            self.communication_layers = [self.fe1,self.pd1,self.fe2,self.pd2]
        
    def forward(self,x,option=None):
        if option is not None:
            self.model_type = option
            
        if self.model_type=="mix":
            x = self.fe1(x)
            mid = self.pd1(x)
            x = self.fe2(x)
            out = self.pd2(x)
            mid = self.multiply1(mid)
            out = self.multiply2(out)
            # out = self.concatenate([out,mid])
            # out = self.mix(out)
            out = self.softmax(out)
            return out
        elif self.model_type=="shallow":
            x = self.fe1(x)
            x = self.pd1(x)
            x = self.softmax(x)
            return x
        elif self.model_type=="deep":
            x = self.fe1(x)
            x = self.fe2(x)
            x = self.pd2(x)
            x = self.softmax(x)
            return x
        
def flatten(model):
    embedding = []
    for child in model.communication_layers:
        for param in child.parameters():
            embedding.append(param.data.view(-1))
    return torch.cat(embedding)

def unflatten(model,embedding):
    pointer = 0
    embedding = embedding
    for child in model.communication_layers:
        for param in child.parameters():
            num_value = torch.prod(torch.LongTensor(list(param.size())))
            param.data = embedding[pointer:pointer+num_value].view(param.size())
            pointer+=num_value