import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dl model load
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=24, out_channels=24,
                                kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=24, hidden_size=64,
                             num_layers=5, batch_first=True)
        self.linear1 = nn.Linear(832, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.linear3(x)
        x = torch.flatten(x)
        return x

# normalization
def minmax(X, maximum, minimum):
    return (X - minimum) / (maximum - minimum), maximum, minimum

def reverse_minmax(X, maximum, minimum):
    return X * (maximum - minimum) + minimum

elecgrid = [    
    "SoccerField",         # 축구장
    "StudentCenter",       # 학생회관
    "CentralWarehouse",    # 중앙창고
    "AcademicBuilding",    # 학사과정
    "DasanBuilding",       # 다산빌딩
    "FacilitiesManagement",# 시설관리동
    "CollegeC",            # 대학C동
    "AnimalLab",           # 동물실험동
    "MainLibrary",         # 중앙도서관
    "LGLibrary",           # LG도서관
    "RenewableEnergy",     # 신재생에너지동
    "SamsungEnvBuilding",  # 삼성환경동
    "CentralResearchEquip",# 중앙연구기기센터
    "IndustryCollab",      # 산업협력관
    "DormB"                # 기숙사 B동
]

path = "./model_result/"
inference_dataset = np.array(pd.read_csv("./inference.csv"))
inference_value = {}

for idx, grid in enumerate(elecgrid):
    # ml model load
    with open(path + f"machinelearning/xgboost_model_{grid}.pickle", "rb") as f:
        ml = pickle.load(f)

    dl = CNNLSTM()
    dl.load_state_dict(torch.load(path + f"cnnlstm/cnn-lstm_model_{grid}.pt", map_location=torch.device("cpu")))

    # min, max load
    with open("gen_max.pickle", "rb") as f:
        maximum = pickle.load(f)

    with open("gen_min.pickle", "rb") as f:
        minimum = pickle.load(f)

    # normalization
    norm_dataset, _, _ = minmax(inference_dataset, maximum[0:14], minimum[0:14])

    ml_X = inference_dataset[:, 1:14]
    dl_X = np.array(norm_dataset[:, 1:14])

    ml_predict = ml.predict(ml_X)

    dl_predict = []
    for i in range(len(norm_dataset) - 24):
        pred = dl(torch.FloatTensor(dl_X[i:i+24]).unsqueeze(0)).item()
        dl_predict.append(pred)
    dl_predict = np.array(dl_predict)
    nonnorm_dl_predict = reverse_minmax(dl_predict, maximum[14+idx], minimum[14+idx])

    # for i in range(0, 11):
    #     total_predict = 0.7 * ml_predict[24:] + 0.3 * nonnorm_dl_predict
    #     inference_value[grid] = total_predict

    #     plt.subplot(11, 1, i+1)
    #     plt.plot(ml_predict[24:], label="ml")
    #     plt.plot(nonnorm_dl_predict, label="dl")
    #     plt.plot(total_predict, label="ensemble")
    #     plt.plot(target, label="target")
    #     plt.title(((target - total_predict) ** 2).mean())
    #     # plt.legend()
    # plt.show()
    total_predict = 0.7 * ml_predict[24:] + 0.3 * nonnorm_dl_predict
    total_predict = np.clip(total_predict, 0, None)
    inference_value[grid] = total_predict
#     plt.plot(ml_predict[24:], label="ml")
#     plt.plot(nonnorm_dl_predict, label="dl")
#     plt.plot(total_predict, label="ensemble")
#     plt.plot([
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1, 3.4, 11.4, 18.6, 14.8,
#     33.1, 23.2, 13.5, 24.8, 19.3, 11.8, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0
# ], label="target")
#     plt.xlabel("Time (h)")
#     plt.ylabel("Solar gen (kwh)")
#     plt.legend()
#     plt.show()

inference_dataframe = pd.DataFrame(inference_value)
inference_dataframe = inference_dataframe.fillna(0)
inference_dataframe["Total"] = inference_dataframe.sum(axis=1)
inference_dataframe.to_csv("./solargrid_inference_result.csv", index=False)