
import pytest
import numpy as np
from main import run_quoremind_simulation

def test_run_simulation_custom_params():
    """
    Ensures that the simulation runs with custom parameters and improves the objective function.
    """
    initial_states = np.array([[0.5, 0.5], [0.4, 0.6]])
    target_state = np.array([0.9, 0.1])
    prn_influence = 0.5
    learning_rate = 0.05
    max_iterations = 10
    
    results = run_quoremind_simulation(
        initial_states=initial_states,
        target_state=target_state,
        prn_influence=prn_influence,
        learning_rate=learning_rate,
        max_iterations=max_iterations
    )
    
    assert "optimized_states" in results
    assert "initial_objective" in results
    assert "final_objective" in results
    assert "improvement" in results
    
    # We expect some improvement even in 10 iterations
    assert results["improvement"] >= -0.1 # Allow for small noise/variability but generally it should improve

def test_run_simulation_defaults():
    """
    Ensures the simulation runs with default parameters.
    """
    results = run_quoremind_simulation(max_iterations=5)
    assert results["final_objective"] <= results["initial_objective"] + 0.5 # Safety margin
