import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from scipy.optimize import linprog

# -----------------------------
# 1. Mock NLP extraction (parameter ranges)
# -----------------------------
param_ranges = {
    'temp': (70, 80),
    'time': (4, 6),
    'catalyst': (0.8, 1.2)
}

# -----------------------------
# 2. Generate synthetic process data
# -----------------------------
def generate_data(n=200):
    T = np.random.uniform(*param_ranges['temp'], n)
    R = np.random.uniform(*param_ranges['time'], n)
    C = np.random.uniform(*param_ranges['catalyst'], n)
    # Yield function (nonlinear peak with noise)
    Y = (
        90
        - (T - 75) ** 2 * 0.3
        - (R - 5) ** 2 * 2
        - (C - 1) ** 2 * 20
        + np.random.normal(0, 2, n)
    )
    return pd.DataFrame({'temp': T, 'time': R, 'catalyst': C, 'yield': Y})

# -----------------------------
# 3. Train ML surrogate (Random Forest)
# -----------------------------
data = generate_data(300)
X = data[['temp', 'time', 'catalyst']]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Surrogate RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# -----------------------------
# 4. Bayesian Optimization to get best candidate
# -----------------------------
def objective(params):
    T, R, C = params
    return -model.predict([[T, R, C]])[0]  # negative because gp_minimize minimizes

search_space = [
    param_ranges['temp'],
    param_ranges['time'],
    param_ranges['catalyst']
]

res = gp_minimize(objective, search_space, n_calls=30, random_state=42)
best_params = res.x
print("Bayesian best candidate:", best_params)

# -----------------------------
# 5. Monte Carlo Validation (variability check)
# -----------------------------
def monte_carlo_eval(params, n=1000):
    T, R, C = params
    T_sim = np.random.normal(T, 0.5, n)  # ±0.5°C variability
    R_sim = np.random.normal(R, 0.1, n)  # ±0.1h variability
    C_sim = np.random.normal(C, 0.05, n) # ±0.05 variability
    preds = model.predict(np.vstack([T_sim, R_sim, C_sim]).T)
    return preds

mc_preds = monte_carlo_eval(best_params)
print("Monte Carlo mean:", mc_preds.mean())
print("Monte Carlo std:", mc_preds.std())
print("Prob yield < 85%:", (mc_preds < 85).mean())

# -----------------------------
# 6. Local Refinement (grid search around candidate)
# -----------------------------
T0, R0, C0 = best_params
best_score = -1e9
best_local = None

for T in np.linspace(T0 - 1, T0 + 1, 5):
    for R in np.linspace(R0 - 0.2, R0 + 0.2, 5):
        for C in np.linspace(C0 - 0.05, C0 + 0.05, 5):
            y_pred = model.predict([[T, R, C]])[0]
            if y_pred > best_score:
                best_score = y_pred
                best_local = (T, R, C)

print("Refined best params:", best_local, "Pred yield:", best_score)

# -----------------------------
# 7. LPP: Resource allocation (maximize profit)
# -----------------------------
# Example: 2 product types using the optimal recipe
profit = [best_score * 10, best_score * 8]   # profit coefficients
A = [[1, 2], [2, 1]]                         # resource usage
b = [100, 80]                                # resource limits

res_lpp = linprog(c=[-p for p in profit], A_ub=A, b_ub=b, bounds=(0, None))
print("LPP optimal batches:", res_lpp.x)
print("Max profit:", -res_lpp.fun)
