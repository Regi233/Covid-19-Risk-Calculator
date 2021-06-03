
import os
import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



class CombinedDataset(Dataset):


    def __init__(self, csv_path, m_names, p_names):

        data_f = pd.read_csv(csv_path)

        data_f_m = data_f[m_names]
        data_f_p = data_f[p_names]
        data_f_y = data_f["Severity"]
        self.le = preprocessing.LabelEncoder()
        data_f_y = self.le.fit_transform(data_f_y)


        
        self.x_m = torch.from_numpy(data_f_m.to_numpy())
        self.x_p = torch.from_numpy(data_f_p.to_numpy())
        self.y = torch.from_numpy(data_f_y)

        self.mms_m = MinMaxScaler()
        self.mms_p = MinMaxScaler()

        self.x_m = self.mms_m.fit_transform(self.x_m)
        self.x_p = self.mms_p.fit_transform(self.x_p)

    def __len__(self):
        return len(self.x_m)

    def __getitem__(self, idx):
        return (self.x_m[idx], self.x_p[idx], self.y[idx])



if __name__ == "__main__":
    m_names = ["levulinate (4-oxovalerate)", 
        "1,2-dilinoleoyl-GPC (18:2/18:2)",
        "3-hydroxypyridine sulfate",
        "1-stearoyl-GPC (18:0)",
        "beta-sitosterol",
        "3-methyl catechol sulfate (1)",
        "6-bromotryptophan",
        "1-pentadecanoyl-GPC (15:0)*",
        "1-stearoyl-2-oleoyl-GPS (18:0/18:1)",
        "1-linoleoyl-GPE (18:2)*",
        "1-(1-enyl-palmitoyl)-GPC (P-16:0)*"]

    p_names = ["interleukin 6 (interferon, beta 2).2", "keratin 19",
              "chemokine (C-C motif) ligand 7", "leukemia inhibitory factor","TNF receptor-associated factor 2"]

    dataset = CombinedDataset(csv_path="./merge_df.csv", m_names=m_names, p_names=p_names)
    print(dataset[0][2])