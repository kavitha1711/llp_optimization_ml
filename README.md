# Process Optimization using ML, Bayesian Optimization, Monte Carlo & LPP

This project demonstrates a hybrid optimization pipeline combining:
- **Synthetic data generation** for a chemical process.
- **Random Forest surrogate model** to learn process yield.
- **Bayesian Optimization** to suggest best process parameters.
- **Monte Carlo Simulation** for robustness testing.
- **Local Grid Refinement** for fine-tuning.
- **Linear Programming (LPP)** for profit-maximizing resource allocation.

---

## 🚀 Steps Performed
1. **Generate Data** – simulate process yields with temperature, time, and catalyst.
2. **Train Surrogate Model** – fit Random Forest to approximate the true function.
3. **Bayesian Optimization** – explore parameter space for maximum yield.
4. **Monte Carlo Simulation** – evaluate robustness under real-world variability.
5. **Local Refinement** – fine-tune around best candidate.
6. **Linear Programming** – optimize resource allocation for profit.

---

## 📦 Installation
```bash
git clone https://github.com/your-username/process_optimization.git
cd process_optimization
pip install -r requirements.txt
```

---

## ▶️ Usage
```bash
python optimization.py
```

---

## 📊 Example Output
```
Surrogate RMSE: 2.05
Bayesian best candidate: [75.3, 5.02, 0.98]
Monte Carlo mean: 89.7
Monte Carlo std: 1.8
Prob yield < 85%: 0.12
Refined best params: (75.2, 5.01, 0.99) Pred yield: 90.1
LPP optimal batches: [20. 30.]
Max profit: 3100.5
```

---

## 🛠 Dependencies
- numpy
- pandas
- scikit-learn
- scikit-optimize
- scipy
