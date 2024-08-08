import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv("./2023_train.csv")
# df = df.iloc[-48:, :12]
# df = df.fillna(0)

# df.to_csv("inference.csv", index=False)

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
]

solar_data1 = pd.read_excel("../../2023/태양광일보/태양광 일보.gcf_2023-08-30_23-59-16.xls")
solar_data2 = pd.read_excel("../../2023/태양광일보/태양광 일보.gcf_2023-08-31_23-59-28.xls")
environment1 = solar_data1.iloc[5:-1, 1:5]
environment2 = solar_data2.iloc[5:-1, 1:5]
environment = pd.concat((environment1, environment2), axis=0)

# weather
weather_data = pd.read_csv("2023_weather.csv", encoding="cp949").iloc[-48:, 3:]
weather_data = weather_data.fillna(0)

full_df = pd.concat((weather_data.reset_index(drop=True), 
                     environment.reset_index(drop=True)), axis=1)
full_df.columns = col

full_df.to_csv("./inference.csv")