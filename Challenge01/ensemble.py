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
        # self.bn1 = nn.BatchNorm1d(24)
        self.lstm = nn.LSTM(input_size=24, hidden_size=64, 
                            num_layers=5, batch_first=True)
        self.linear1 = nn.Linear(576, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
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

building_name = ['HVM2', 'HVNM', 'HVEM', 'CentralPP', 'EEBuilding',
       'MSBuilding', 'LSBuilding', 'MEBuilding', 'LGLibrary', 'StartupB',
       'KumhoHall', 'CoolingPump123', 'CoolingPump456', 'MarriedAptE', 'Dorm9',
       'AdvancedLight', 'MSBuildingE', 'EEBuildingE', 'LSBuildingE',
       'MEBuildingE', 'LGLibraryE', 'CentralPPE', 'AdvancedLightE', 'HVNM1',
       'HVNM2', 'HighVoltageCapacitor', 'RenewableEnergyBuilding', 'CollegeB',
       'CollegeDormA', 'StudentUnion2', 'ScholarPP', 'FacultyApt', 'CollegeC',
       'CentralResearchEquipmentCenter', 'CollegeA', 'DasanBuilding',
       'IndustryCollabResearchBuilding', 'RenewableEnergyBuildingE',
       'CollegeBE', 'CollegeDormAE', 'ScholarPPE', 'StudentUnion2E',
       'FacultyAptE', 'CollegeCE', 'CentralResearchEquipmentCenterE',
       'CollegeAE', 'DasanBuildingE', 'IndustryCollabResearchBuildingE', 'SV2',
       'SV5', 'SV6']

path = "./model_result/"
inference_dataset = np.array(pd.read_csv("./inference.csv"))
inference_value = {}

for idx, building in enumerate(building_name):
    # ml model load
    with open(path + f"machinelearning/xgboost_model_{building}.pickle", "rb") as f:
        ml = pickle.load(f)

    dl = CNNLSTM()
    dl.load_state_dict(torch.load(path + f"cnnlstm/cnn-lstm_model_{building}.pt", map_location=torch.device("cpu")))

    # min, max load
    with open("elec_max.pickle", "rb") as f:
        maximum = pickle.load(f)

    with open("elec_min.pickle", "rb") as f:
        minimum = pickle.load(f)

    # normalization
    norm_dataset, _, _ = minmax(inference_dataset, maximum[:12], minimum[:12])

    ml_X = inference_dataset[:, 2:12]
    dl_X = np.array(np.concatenate((norm_dataset[:, 2:4], norm_dataset[:, 5:12]), axis=1))

    ml_predict = ml.predict(ml_X)

    dl_predict = []
    for i in range(len(norm_dataset) - 24):
        pred = dl(torch.FloatTensor(dl_X[i:i+24]).unsqueeze(0)).item()
        dl_predict.append(pred)
    dl_predict = np.array(dl_predict)
    nonnorm_dl_predict = reverse_minmax(dl_predict, maximum[12+idx], minimum[12+idx])

    total_predict = 0.1 * ml_predict[24:] + 0.9 * nonnorm_dl_predict
    inference_value[building] = total_predict

#     plt.plot(ml_predict[24:], label="ml")
#     plt.plot(nonnorm_dl_predict, label="dl")
#     plt.plot(total_predict, label="ensemble")
#     plt.plot([
#     136.3, 125.9, 120.0, 121.9, 115.5, 113.4, 119.7, 139.3,
#     232.4, 259.3, 262.0, 231.9, 226.2, 253.6, 310.4, 316.2,
#     257.2, 239.8, 214.3, 218.3, 213.2, 196.2, 143.6, 137.4
# ], label="target")
#     plt.xlabel("Time (h)")
#     plt.ylabel("Consume (kwh)")
#     plt.legend()
#     plt.title(building)
#     plt.show()


inference_dataframe = pd.DataFrame(inference_value)
inference_dataframe = inference_dataframe.fillna(0)
inference_dataframe["Total"] = inference_dataframe.sum(axis=1)
inference_dataframe.to_csv("./elec_inference_result.csv")