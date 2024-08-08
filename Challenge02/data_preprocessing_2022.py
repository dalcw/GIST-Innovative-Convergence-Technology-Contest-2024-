import os
import numpy as np
import pandas as pd
import holidays

col = [
    "AirTempC",           # 기온(°C)
    "RainfallMm",         # 강수량(mm)
    "WindSpeedMs",        # 풍속(m/s)
    "WindDirection",      # 풍향(16방위)
    "HumidityPercent",    # 습도(%)
    "PressureHpa",        # 현지기압(hPa)
    "SunshineHours",      # 일조(hr)
    "SolarRadiationMJm2", # 일사(MJ/m2)
    "GroundTempC",        # 지면온도(°C)
    "HorizontalSurfaceTemp",  # 수평면
    "OutdoorTemp",            # 외기온도
    "InclinedSurfaceTemp",    # 경사면
    "ModuleTemp",             # 모듈온도
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

# 파일명
file_list = os.listdir("../../2022/태양광발전일보")
file_list.sort()

# list
total_df = []

for i in range(len(file_list)):
    df = pd.read_excel("../../2022/태양광발전일보/" + file_list[i])
    environment = df.iloc[5:-1, 1:5]
    solar_power_per_hour = df.iloc[5:-1, 6:-1:2]
    merge = pd.concat((environment, solar_power_per_hour), axis=1)
    total_df.append(merge)

total_df = pd.concat(total_df, axis=0)

weather_data = pd.read_csv("2022_weather.csv", encoding="cp949")
weather_data = weather_data.fillna(0)

full_df = pd.concat((weather_data.iloc[:, 3:].reset_index(drop=True), total_df.reset_index(drop=True)), axis=1)
full_df.columns = col
full_df.to_csv("2022_train.csv")