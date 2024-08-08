import pandas as pd

elec_usage = pd.read_csv("./elec_inference_result.csv")
sum_of_usage = elec_usage.iloc[:, [5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37]].sum(1).values

elec_usage["Valid_sum_of_value"] = sum_of_usage

elec_usage.to_csv("elec_inference_result.csv")