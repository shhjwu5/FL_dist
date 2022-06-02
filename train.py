from servers import *
from clients import *
from models import *
from datasets import *

import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam
import torch
devices = [5,7]

def main():
    rank,world_size = init_dist()

    args = {
        "rank":rank,"world_size":world_size,"batch_size":32,
        "train_type":"deep","communication_type":"deep",
        "Model":ExampleNet,"Loss_function":nn.CrossEntropyLoss(),"Optimizer":Adam
    }
    epochs = 10
    
    if rank==world_size-1:
        agent = ServerFedAvg(args)
        agent.train(epochs)
    else:
        args["device"] = torch.device("cuda:%d"%(devices[rank%len(devices)]))
        args["Trainloader"] = load_purchase100(file_path="./datasets/train/",dataset_type="train",rank=args["rank"],batch_size=args["batch_size"])
        args["Testloader"] = load_purchase100(file_path="./datasets/train/",dataset_type="test",rank=args["rank"],batch_size=args["batch_size"])
        agent = ClientFedAvg(args)
        agent.train(epochs)

def init_dist():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return [rank, world_size]

if __name__=="__main__":
    main()