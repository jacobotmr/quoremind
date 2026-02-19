import sys
import os
import numpy as np
import pytest

# Añadir el directorio raíz al path para importar quoremind
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quoremind import QuantumBayesMahalanobis, EnhancedPRN

def test_mahalanobis_vectorized():
    """Valida que el cálculo optimizado con einsum sea correcto."""
    qb = QuantumBayesMahalanobis()
    
    # Generar datos de prueba
    np.random.seed(42)
    A = np.random.randn(20, 3)
    B = np.random.randn(10, 3)
    
    # Método manual (el original antes del cambio)
    inv_cov = qb._empirical_inv_cov(A)
    mean_A = np.mean(A, axis=0)
    diff = B - mean_A
    aux = diff @ inv_cov
    dist_manual = np.sqrt(np.einsum("ij,ij->i", aux, diff))
    
    # Método optimizado (la función actual en quoremind.py ya fue modificada)
    dist_opt = qb.compute_quantum_mahalanobis(A, B)
    
    np.testing.assert_allclose(dist_opt, dist_manual, rtol=1e-7)
    assert dist_opt.shape == (10,)

def test_enhanced_prn_mahalanobis():
    """Valida el cálculo de Mahalanobis dentro de EnhancedPRN."""
    prn = EnhancedPRN(influence=0.5)
    
    np.random.seed(42)
    states = np.random.randn(15, 2)
    probs = {"0": 0.5, "1": 0.5}
    
    # Método manual
    cov = np.cov(states, rowvar=False)
    inv_cov = np.linalg.pinv(np.atleast_2d(cov))
    mean_state = np.mean(states, axis=0)
    diff = states - mean_state
    aux = diff @ inv_cov
    dist_sq_manual = np.einsum("ij,ij->i", aux, diff)
    expected_mahal = float(np.mean(np.sqrt(dist_sq_manual)))
    
    entropy, actual_mahal = prn.record_quantum_noise(probs, states)
    
    assert np.isclose(actual_mahal, expected_mahal, rtol=1e-7)
    assert len(prn.mahalanobis_records) == 1

def test_mahalanobis_dimensions():
    """Valida que se lancen errores con dimensiones incompatibles."""
    qb = QuantumBayesMahalanobis()
    A = np.random.randn(10, 3)
    B = np.random.randn(5, 2) # Diferente dimensión feature
    
    with pytest.raises(ValueError, match="Las dimensiones de A y B deben coincidir."):
        qb.compute_quantum_mahalanobis(A, B)

if __name__ == "__main__":
    pytest.main([__file__])
