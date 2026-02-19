
import numpy as np
import argparse
from quoremind import (
    QuantumNoiseCollapse,
    VonNeumannEntropy,
    lambda_doble_operator
)

def run_quoremind_simulation(
    initial_states=None,
    target_state=None,
    prn_influence=0.6,
    learning_rate=0.01,
    max_iterations=100,
    seed=42
):
    """
    Runs a QuoreMind simulation with the given parameters.
    """
    np.random.seed(seed)
    
    # 1. Default initial states if none provided
    if initial_states is None:
        initial_states = np.random.randn(5, 2)
    else:
        initial_states = np.array(initial_states)
        
    # 2. Default target state if none provided
    if target_state is None:
        target_state = np.array([0.8, 0.2])
    else:
        target_state = np.array(target_state)
        
    print(f"--- QuoreMind Simulation ---")
    print(f"PRN Influence:  {prn_influence}")
    print(f"Learning Rate:  {learning_rate}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Target State:   {target_state}")
    print("-" * 30)

    # 3. Create simulation system
    collapse_system = QuantumNoiseCollapse(prn_influence=prn_influence)
    
    # 4. Perform Optimization
    init_obj = collapse_system.objective_function_with_noise(
        initial_states.astype(np.float32), target_state, n_step=1
    )
    
    opt_states, final_obj = collapse_system.optimize_quantum_state(
        initial_states=initial_states,
        target_state=target_state,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
    )

    print(f"✓ Initial Objective: {init_obj:.6f}")
    print(f"✓ Final Objective:   {final_obj:.6f}")
    print(f"✓ Improvement:       {init_obj - final_obj:.6f}")
    print("-" * 30)
    
    return {
        "optimized_states": opt_states,
        "initial_objective": init_obj,
        "final_objective": final_obj,
        "improvement": init_obj - final_obj
    }

def main():
    parser = argparse.ArgumentParser(description="QuoreMind Parameter Configuration CLI")
    
    parser.add_argument("--prn", type=float, default=0.6, help="PRN Influence (0.0 to 1.0)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    parser.add_argument("--iterations", type=int, default=100, help="Maximum Iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--target", type=float, nargs=2, default=[0.8, 0.2], help="Target State (two floats)")
    
    args = parser.parse_args()
    
    run_quoremind_simulation(
        prn_influence=args.prn,
        learning_rate=args.lr,
        max_iterations=args.iterations,
        seed=args.seed,
        target_state=args.target
    )

if __name__ == "__main__":
    main()
