# rl-option-hedging: (this is what recruiters actually read). Include, in this order:

#CI badge
Problem in 3 sentences (what hedging is, why costs matter)
Results table + pnl_hist.png + the cost-vs-risk plot
Honest findings and limitations (GBM assumptions, Black-Scholes pricing, no real options data, one option type)
How to run: pip install -r requirements.txt, python train.py, python run_eval.py, pytest
Project structure
What you would do next (stochastic volatility, puts, discrete strikes)
C. Portfolio polish: pin the repo on your GitHub profile, add topics (reinforcement-learning, quantitative-finance, stable-baselines3), and write a short Medium post on what you learned.

Resume bullet template:

Built a PPO-based deep hedging agent in Gymnasium to hedge short call options under transaction costs; benchmarked against Black-Scholes delta hedging using std and CVaR(95%) on 2,000 simulated and real SPY paths; added pytest suite and GitHub Actions CI; deployed an interactive Streamlit demo