import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class R_Net(nn.Module):
    
    def __init__(self, n_feature_m, n_feature_p):
        super().__init__()
        self.n_feature_m = n_feature_m
        self.n_feature_p = n_feature_p
        self.output_channel = 4
        
    
        self.m_net = nn.Sequential(
            nn.Linear(self.n_feature_m, int(math.sqrt(self.n_feature_m * self.n_feature_p * self.output_channel))),
            nn.ReLU()
        )

        self.p_net = nn.Sequential(
            nn.Linear(self.n_feature_p, int(math.sqrt(self.n_feature_m * self.n_feature_p * self.output_channel))),
            nn.ReLU()
        )
        self.o_net = nn.Linear(2 * int(math.sqrt(self.n_feature_m * self.n_feature_p * self.output_channel)), self.output_channel)
        
    def forward(self, x_n, x_p):
        h_n = self.m_net(x_n)
        h_p = self.p_net(x_p)
        # print(h_n.size())
        # print(h_p.size())
        output = F.log_softmax(self.o_net(torch.cat((h_n, h_p), 1)), dim=1)
        return output