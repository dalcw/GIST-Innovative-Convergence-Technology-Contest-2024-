# requires
# numpy, pandas, holidays, xlrd, openpyxl

import os
import numpy as np
import pandas as pd
import holidays

# parameter
# 유효 전력 인덱스 - 1 (석사 일보)
extraction_idx_1 = [7, 18, 29, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117, 122, 127, 132, 137, 142, 147, 152]

# 각 건물 식별자 - 1 (석사 일보)
identifiers_1 = [
    "SV2",
    "SV5",
    "SV6",
    "HVM1",
    "HVM2",
    "HVNM",
    "HVEM",
    "CentralPP",
    "EEBuilding",
    "MSBuilding",
    "LSBuilding",
    "MEBuilding",
    "LGLibrary",
    "StartupB",
    "KumhoHall",
    "CoolingPump123",
    "CoolingPump456",
    "MarriedAptE",
    "Dorm9",
    "AdvancedLight",
    "MSBuildingE",
    "EEBuildingE",
    "LSBuildingE",
    "MEBuildingE",
    "LGLibraryE",
    "CentralPPE",
    "AdvancedLightE"
]

extraction_idx_2 = [7, 18, 29, 40, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98, 103, 108, 113, 118, 123, 128, 133, 138, 143, 148, 153, 158, 163, 168]

identifiers_2 = [
    "SV2",
    "SV5",
    "SV6",
    "SV7",
    "HVNM1",
    "HVNM2",
    "HighVoltageCapacitor",
    "RenewableEnergyBuilding",
    "CollegeB",
    "CollegeDormA",
    "StudentUnion2",
    "ScholarPP",
    "FacultyApt",
    "CollegeC",
    "CentralResearchEquipmentCenter",
    "CollegeA",
    "DasanBuilding",
    "IndustryCollabResearchBuilding",
    "RenewableEnergyBuildingE",
    "CollegeBE",
    "CollegeDormAE",
    "ScholarPPE",
    "StudentUnion2E",
    "FacultyAptE",
    "CollegeCE",
    "CentralResearchEquipmentCenterE",
    "CollegeAE",
    "DasanBuildingE",
    "IndustryCollabResearchBuildingE"
]


# 파일명
file_list = os.listdir("../../2023/전력일보/")
file_list.sort()
print(len(file_list))

# 전체 데이터 프레임들
total_df = []

# 첫 번째 파일은 .형식의 숨겨진 파일임 <- 상황에 따라 다를 수 있는 부분
for i in range(0, 61):
    # file read
    df1 = pd.read_excel("../../2023/전력일보/" + file_list[i]).iloc[9:, extraction_idx_1]
    df1 = df1.replace("-", 0).astype(float)

    df2 = pd.read_excel("../../2023/전력일보/" + file_list[i+61]).iloc[9:, extraction_idx_2]
    df2 = df2.replace("-", 0).astype(float)



    # 석사 + 학사 일보에서 겹치는 건물에 대해서는 평균을 낸다
    intersection_buildings = [0, 1, 2]
    intersection_identifiers = ["SV2", "SV5", "SV6"]
    merged_df = (df1.iloc[:, intersection_buildings] + df2.iloc[:, intersection_buildings]) / 2

    merged_df = pd.concat((df1.iloc[:, 4:], df2.iloc[:, 4:], merged_df), axis=1)
    merged_df.columns = identifiers_1[4:] + identifiers_2[4:] + intersection_identifiers
    total_df.append(merged_df)


# all merge
total_df = pd.concat(total_df, axis=0)

# weather data
weather_data = pd.read_csv("../../2023_weather.csv", encoding="cp949")

# 결측치 확인 및 보간
# weather_data["강수량(mm)"] = weather_data["강수량(mm)"].fillna(0)
# weather_data["일조(hr)"] = weather_data["일조(hr)"].fillna(0) 
# weather_data["일사(MJ/m2)"] = weather_data["일사(MJ/m2)"].fillna(0)
weather_data = weather_data.fillna(0)


# 공휴일의 특징을 추가함 <- 공유일에 따라서 전력 사용량이 다른 경향성을 보일 수 있음..!
# 아마..? -> 그런데.. 이런 곳은.. 주말이 없어서 장담은 못함
dates = pd.date_range(start="2023-07-01", end="2023-08-30")
kr_holidays = holidays.KR()

holiday_dates = []
for date in dates:
    if date.weekday() >= 5 or date in kr_holidays:
        holiday_dates.extend([1]*24)
    else: holiday_dates.extend([0]*24)

# 주말 or 공휴일 반영
weather_data["holidays"] = pd.Series(holiday_dates)

# 일시에서 시간 정보만 추출 (년 월, 일 정보는 크게 영향을 안 미칠 것임 <- 다른 기온적 요인으로 커버 가능)
weather_data["일시"] = pd.Series([int(date.split()[1].split(":")[0]) for date in weather_data["일시"]])

# 전체 통합 데이터 셋
full_df = pd.concat((weather_data.iloc[:, 2:].reset_index(drop=True), total_df.reset_index(drop=True)), axis=1)
full_df.to_csv("2023_train.csv")