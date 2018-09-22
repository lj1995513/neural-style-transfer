
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        self.target = target.detach()
    def forward(self,input):
        self.loss = F.mse_loss(input,self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
    def forward(self,input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G,self.target)
        return input
    def gram_matrix(self,input):
        batch_size,feature_num,c,d = input.size()
        features = input.view(batch_size*feature_num,c*d)
        G = torch.mm(features,features.t())#A*A.T
        return G.div(batch_size*feature_num*c*d)

class Normalize(nn.Module):
    def __init__(self,mean,std):
        super(Normalize,self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
    def forward(self,input):
        return (input-self.mean)/self.std