
import numpy as np
from main import run_quoremind_simulation

# Configuración solicitada por el usuario
PRN_INFLUENCE = 0.72
LEARNING_RATE = 0.01
MAX_ITERATIONS = 100
NUM_STATES = 8

# Definición de los targets (como vectores 4D basados en las matrices de momentos)
targets = [
    np.array([1, 6, 6, 1]),
    np.array([2, 5, 5, 2]),
    np.array([3, 4, 4, 3])
]

def run_experiment():
    print("=" * 80)
    print(f"🚀 Ejecutando Experimento QuoreMind: 8 Estados, PRN={PRN_INFLUENCE}")
    print("=" * 80)

    # Semilla para reproducibilidad
    np.random.seed(42)
    
    # Generar 8 estados iniciales de dimensión 4 (para coincidir con los targets)
    initial_states = np.random.randn(NUM_STATES, 4)

    results_all = []

    for i, target in enumerate(targets):
        print(f"\n🎯 Simulación {i+1} - Target: {target}")
        
        results = run_quoremind_simulation(
            initial_states=initial_states,
            target_state=target,
            prn_influence=PRN_INFLUENCE,
            learning_rate=LEARNING_RATE,
            max_iterations=MAX_ITERATIONS,
            seed=42
        )
        results_all.append(results)

    print("\n" + "=" * 80)
    print("✅ Experimento Completado")
    print("=" * 80)
    
    for i, res in enumerate(results_all):
        print(f"Simulación {i+1}: Mejora = {res['improvement']:.6f}")

if __name__ == "__main__":
    run_experiment()
