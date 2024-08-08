import cvxpy as cp # optimization library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# time setting
T = 24

# pv
pv_gen = pd.read_csv("./solargrid_inference_result.csv")
sum_of_gen = pv_gen.sum(1).values
# print(sum_of_gen)

# elec usage
elec_usage = pd.read_csv("./elec_inference_result.csv")
sum_of_usage = elec_usage.iloc[:, [5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37]].sum(1).values
# print(sum_of_usage)

# elec cost
cost_per_kWh = np.array([
    40.3,  # 00:00
    40.3,  # 01:00
    40.3,  # 02:00
    40.3,  # 03:00
    40.3,  # 04:00
    40.3,  # 05:00
    40.3,  # 06:00
    40.3,  # 07:00
    40.3,  # 08:00
    85.0,  # 09:00
    150.9, # 10:00
    150.9, # 11:00
    85.0,  # 12:00
    150.9, # 13:00
    150.9, # 14:00
    150.9, # 15:00
    150.9, # 16:00
    85.0,  # 17:00
    85.0,  # 18:00
    85.0,  # 19:00
    85.0,  # 20:00
    85.0,  # 21:00
    85.0,  # 22:00
    40.3   # 23:00
])

# BESS Constraint
P_bess_max = 250  # speed of bess
C_bess = 750  # cap of bess
rate_charge = 0.93
rate_discharge = 0.93
SOC_min = 0.2 * C_bess
SOC_max = 0.8 * C_bess
SOC_init = 0.5 * C_bess

# ramp rate
ramp_rate = 25  # kW/h (배터리 출력량(250kW)의 10%)

# optimization variable
P_charge = cp.Variable(T, nonneg=True)  # charge
P_discharge = cp.Variable(T, nonneg=True)  # discharge
SOC = cp.Variable(T+1, nonneg=True)  # SOC
P_grid = cp.Variable(T, nonneg=True)  # grid
charge_rate = cp.Variable(T, nonneg=True) # exclusivity charge
discharge_rate = cp.Variable(T, nonneg=True) # exclusivity discharge

# objective function
objective = cp.Minimize(cp.sum(cp.multiply(cost_per_kWh, P_grid)))

# constraint
constraints = [
    SOC[0] == SOC_init,
    SOC[T] == SOC_init
]

for t in range(T):
    # update
    constraints  += [
        # SOC update
        SOC[t+1] == SOC[t] + (P_charge[t] * rate_charge - P_discharge[t] / rate_discharge),

        # balance
        sum_of_usage[t] == sum_of_gen[t] + P_discharge[t] - P_charge[t] + P_grid[t],

        # charge and discharge constraint (e.g., SOC cap-, speed)
        P_charge[t] <= P_bess_max,
        P_discharge[t] <= P_bess_max,
        SOC[t+1] >= SOC_min,
        SOC[t+1] <= SOC_max,

        # exclusivity - charge or discharge
        charge_rate[t] + discharge_rate[t] <= 1,
        P_charge[t] == P_bess_max * charge_rate[t],
        P_discharge[t] == P_bess_max * discharge_rate[t]
    ]
    
    # 램프 레이트 제약 조건 수정
       
    if t > 0:
        constraints += [  # 충/방전량이 올라갈 때, 낮아질 때 고려
            P_charge[t] - P_charge[t-1] <= ramp_rate,
            P_charge[t-1] - P_charge[t] <= ramp_rate,
            P_discharge[t] - P_discharge[t-1] <= ramp_rate,
            P_discharge[t-1] - P_discharge[t] <= ramp_rate
        ]
        
# solve
problem = cp.Problem(objective, constraints)
problem.solve()

# print result
print("최적의 시간별 충전량 (kWh):", P_charge.value)
print("최적의 시간별 방전량 (kWh):", P_discharge.value)
print("최적의 SOC 상태 (kWh):", SOC.value)
print("최적의 외부 전력망 구매량 (kWh):", P_grid.value)
print("1 시간 동안의 충전 비율: ", charge_rate.value)
print("1 시간 동안의 방전 비율: ", discharge_rate.value)

print(f"최소화된 총 전기 요금 (원): {problem.value:.2f}", )
print(f"최적화 하지 않았을 때의 전기 요금 (원): {np.sum(sum_of_usage * cost_per_kWh):.2f}")

# graph
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.plot(P_grid.value, label="Optimal purchase(Grid)")
plt.xlabel("Time (h)")
plt.ylabel("kwh")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(P_charge.value, label="Optimal charge(SOC)")
plt.plot(P_discharge.value, label="Optimal discharge(SOC)")
plt.plot(SOC.value, label="Optimal SOC(SOC)")
plt.xlabel("Time (h)")
plt.ylabel("kwh")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(charge_rate.value, label="1 hour per charge rate")
plt.plot(discharge_rate.value, label="1 hour per discharge rate")
plt.plot(charge_rate.value + discharge_rate.value, label="charge_rate + discharge_rate")
plt.xlabel("Time (h)")
plt.ylabel("ratio")
plt.legend()

plt.show()