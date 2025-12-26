import yfinance as yf
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

def solve_portfolio_quantum(tickers, budget=2, risk_weight=0.5):
    """
    tickers: List of stock symbols
    budget: Number of stocks to select
    risk_weight: The 'q' factor (higher means more conservative)
    """
    # 1. Fetch Market Data
    data = yf.download(tickers, period="1y")['Close']
    returns = data.pct_change().dropna()
    mu = returns.mean().values  # Expected Returns
    sigma = returns.cov().values # Risk/Covariance Matrix
    
    # 2. Build the Quadratic Program
    qp = QuadraticProgram("QuantumPortfolio")
    for t in tickers:
        qp.binary_var(name=t)
        
    # Objective: Minimize (Risk * Weight) - Returns
    # This is the 'Cost Function' the quantum computer navigates
    qp.minimize(quadratic=(risk_weight * sigma), linear=-mu)
    
    # Constraint: Must select exactly the budget amount
    qp.linear_constraint(linear={t: 1 for t in tickers}, sense="==", rhs=budget)
    
    # 3. Setup the Quantum Algorithm (QAOA)
    sampler = StatevectorSampler() 
    optimizer = COBYLA(maxiter=100)
    qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    
    # 4. Solve
    qaoa_solver = MinimumEigenOptimizer(qaoa_mes)
    result = qaoa_solver.solve(qp)
    
    return result, data
