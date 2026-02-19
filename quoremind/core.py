"""
QuoreMind v1.0.0 — Sistema Metripléctico Cuántico-Bayesiano
Reescrito sin TensorFlow ni dependencias externas.
Dependencias: numpy, scipy (estándar científico)
"""

import numpy as np
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from typing import Tuple, List, Dict, Union, Any, Optional, Callable
import functools
import time
from dataclasses import dataclass, field

# ============================================================================
# DECORADORES Y UTILIDADES
# ============================================================================

def timer_decorator(func: Callable) -> Callable:
    """Decorador que mide el tiempo de ejecución."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️  {func.__name__} ejecutado en {time.time() - start:.4f}s")
        return result
    return wrapper


def validate_input_decorator(min_val: float = 0.0, max_val: float = 1.0) -> Callable:
    """Decorador que valida argumentos numéricos en un rango."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, arg in enumerate(args[1:], 1):
                if isinstance(arg, (int, float)) and not (min_val <= arg <= max_val):
                    raise ValueError(
                        f"Argumento {i} debe estar entre {min_val} y {max_val}. Valor: {arg}"
                    )
            for name, val in kwargs.items():
                if isinstance(val, (int, float)) and not (min_val <= val <= max_val):
                    raise ValueError(
                        f"Argumento '{name}' debe estar entre {min_val} y {max_val}. Valor: {val}"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# OPERADOR ÁUREO Y PARÁMETROS
# ============================================================================

def golden_ratio_operator(dimension: int, index: int = 0) -> np.ndarray:
    """
    Calcula el Operador Áureo (φ-operator) para modular fase cuasiperiódica.
    φ = (1 + √5) / 2 ≈ 1.618
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phases = np.array([phi ** (i + index) % (2 * np.pi) for i in range(dimension)])
    operator = np.exp(1j * phases)
    return operator / np.linalg.norm(operator)


def aureo_operator(n: int, phi: float = 1.6180339887) -> Tuple[float, float]:
    """Calcula paridad y fase del operador áureo Ô_n."""
    n_f = float(n)
    paridad   = np.cos(np.pi * n_f)
    fase_mod  = np.cos(np.pi * phi * n_f)
    return paridad, fase_mod


def lambda_doble_operator(
    state_vector: np.ndarray,
    hamiltonian: Callable,
    qubits: np.ndarray,
    golden_phase: int = 0,
) -> float:
    """
    λ_doble = (weighted_sum * aureo_component * hamiltonian_effects) / mean_qbits
    """
    amplitude     = np.max(np.abs(state_vector))
    weighted_sum  = np.sum(state_vector * amplitude)
    mean_qbits    = max(np.mean(qubits), 1e-6)

    aureo_op       = golden_ratio_operator(len(state_vector), golden_phase)
    aureo_comp     = np.abs(np.dot(aureo_op, state_vector))
    ham_effects    = np.mean([hamiltonian(q, 0.0) for q in qubits])

    return float(np.real(weighted_sum * aureo_comp * ham_effects / mean_qbits))


def calculate_cosines(entropy: float, prn_object: float) -> Tuple[float, float, float]:
    """Cosenos directores (x, y, z) para un vector 3D."""
    entropy    = max(entropy,    1e-6)
    prn_object = max(prn_object, 1e-6)
    mag  = np.sqrt(entropy ** 2 + prn_object ** 2 + 1.0)
    return entropy / mag, prn_object / mag, 1.0 / mag


# Observables globales
H = lambda q, p: 0.5 * p ** 2 + 0.5 * q ** 2
S = lambda q, p: -0.5 * np.log(q ** 2 + p ** 2 + 1e-6)


# ============================================================================
# ENTROPÍA VON NEUMANN
# ============================================================================

class VonNeumannEntropy:
    """Entropía de von Neumann para operadores de densidad cuánticos."""

    @staticmethod
    def compute_von_neumann_entropy(
        density_matrix: np.ndarray,
        epsilon: float = 1e-15,
        normalize: bool = True,
        method: str = "sigmoid",
    ) -> float:
        """S = -Tr(ρ log ρ)"""
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = np.clip(eigenvalues, epsilon, None)
        entropy     = -np.sum(eigenvalues * np.log2(eigenvalues))

        if not normalize:
            return float(entropy)

        dim         = len(eigenvalues)
        max_entropy = np.log2(dim)

        if method == "sigmoid":
            normalized = 1.0 / (1.0 + np.exp(-entropy))
        elif method == "tanh":
            normalized = (np.tanh(entropy / 2.0) + 1.0) / 2.0
        elif method == "log_compression":
            normalized = np.log(1.0 + entropy) / np.log(1.0 + max_entropy)
        elif method == "min_max":
            normalized = entropy / max_entropy
        elif method == "clamp":
            normalized = np.clip(entropy / max_entropy, 0.0, 1.0)
        else:
            raise ValueError(f"Método desconocido: {method}")

        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def density_matrix_from_state(state: np.ndarray) -> np.ndarray:
        """ρ = |ψ⟩⟨ψ|"""
        return np.outer(state, np.conj(state))

    @staticmethod
    def mixed_state_entropy(
        probabilities: List[float],
        density_matrices: List[np.ndarray],
    ) -> float:
        """Entropía de estado mixto: ρ = Σᵢ pᵢ ρᵢ"""
        rho = np.zeros_like(density_matrices[0], dtype=complex)
        for p, r in zip(probabilities, density_matrices):
            rho += p * r
        return VonNeumannEntropy.compute_von_neumann_entropy(rho)


# ============================================================================
# ESTRUCTURA SIMPLÉCTICA
# ============================================================================

class PoissonBrackets:
    """Estructura simpléctica con corchetes de Poisson {f, g}."""

    @staticmethod
    def _scalar(val: Union[np.ndarray, float]) -> float:
        return val.item() if isinstance(val, np.ndarray) else float(val)

    @staticmethod
    def poisson_bracket(
        f: Callable, g: Callable,
        q: np.ndarray, p: np.ndarray,
        eps: float = 1e-5,
    ) -> float:
        """{f, g} = (∂f/∂q)(∂g/∂p) - (∂f/∂p)(∂g/∂q)"""
        q, p = np.atleast_1d(q), np.atleast_1d(p)
        to_s  = PoissonBrackets._scalar

        df_dq = to_s((f(q + eps, p) - f(q - eps, p)) / (2 * eps))
        df_dp = to_s((f(q, p + eps) - f(q, p - eps)) / (2 * eps))
        dg_dq = to_s((g(q + eps, p) - g(q - eps, p)) / (2 * eps))
        dg_dp = to_s((g(q, p + eps) - g(q, p - eps)) / (2 * eps))

        return float(df_dq * dg_dp - df_dp * dg_dq)

    @staticmethod
    def liouville_evolution(
        H: Callable, f: Callable,
        q: np.ndarray, p: np.ndarray,
    ) -> float:
        """df/dt = {f, H}"""
        return PoissonBrackets.poisson_bracket(f, H, q, p)


# ============================================================================
# ESTRUCTURA METRIPLÉCTICA
# ============================================================================

class MetriplecticStructure:
    """Parte simpléctica + parte métrica (disipativa)."""

    @staticmethod
    def metriplectic_bracket(
        f: Callable, g: Callable,
        q: np.ndarray, p: np.ndarray,
        M: np.ndarray,
        eps: float = 1e-5,
    ) -> float:
        """[f, g] = {f, g} + ⟨∇f, M ∇g⟩"""
        q, p  = np.atleast_1d(q), np.atleast_1d(p)
        to_s  = PoissonBrackets._scalar

        poisson = PoissonBrackets.poisson_bracket(f, g, q, p, eps)

        grad_f = np.array([
            to_s((f(q + eps, p) - f(q - eps, p)) / (2 * eps)),
            to_s((f(q, p + eps) - f(q, p - eps)) / (2 * eps)),
        ])
        grad_g = np.array([
            to_s((g(q + eps, p) - g(q - eps, p)) / (2 * eps)),
            to_s((g(q, p + eps) - g(q, p - eps)) / (2 * eps)),
        ])

        metric = float(grad_f @ M @ grad_g)
        return float(poisson + metric)

    @staticmethod
    def metriplectic_evolution(
        H: Callable, S: Callable, f: Callable,
        q: np.ndarray, p: np.ndarray,
        M: np.ndarray,
        eps: float = 1e-5,
    ) -> float:
        """df/dt = {f, H} + [f, S]_M"""
        ham  = PoissonBrackets.poisson_bracket(f, H, q, p, eps)
        diss = MetriplecticStructure.metriplectic_bracket(f, S, q, p, M, eps)
        return float(ham + diss)


# ============================================================================
# OPTIMIZADOR ADAM (implementación pura en NumPy)
# ============================================================================

class AdamOptimizer:
    """Descenso de gradiente Adam sin dependencias externas."""

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.m       = None   # primer momento
        self.v       = None   # segundo momento
        self.t       = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Actualiza params con el gradiente dado."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m  = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v  = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def numerical_gradient(
    func: Callable[[np.ndarray], float],
    params: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Gradiente numérico por diferencias centradas."""
    grad = np.zeros_like(params)
    it   = np.nditer(params, flags=["multi_index"])
    while not it.finished:
        idx        = it.multi_index
        orig       = params[idx]
        params[idx] = orig + eps
        fp          = func(params)
        params[idx] = orig - eps
        fm          = func(params)
        params[idx] = orig
        grad[idx]   = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


# ============================================================================
# CONFIGURACIÓN Y LÓGICA BAYESIANA
# ============================================================================

@dataclass
class BayesLogicConfig:
    epsilon: float                  = 1e-6
    high_entropy_threshold: float   = 0.8
    high_coherence_threshold: float = 0.6
    action_threshold: float         = 0.5


class BayesLogic:
    """Lógica bayesiana para cálculo de probabilidades."""

    def __init__(self, config: Optional[BayesLogicConfig] = None):
        self.config = config or BayesLogicConfig()

    @validate_input_decorator(0.0, 1.0)
    def calculate_posterior_probability(
        self, prior_a: float, prior_b: float, conditional_b_given_a: float
    ) -> float:
        """P(A|B) = P(B|A)·P(A) / P(B)"""
        prior_b = max(prior_b, self.config.epsilon)
        return (conditional_b_given_a * prior_a) / prior_b

    @validate_input_decorator(0.0, 1.0)
    def calculate_conditional_probability(
        self, joint_probability: float, prior: float
    ) -> float:
        """P(X|Y) = P(X,Y) / P(Y)"""
        return joint_probability / max(prior, self.config.epsilon)

    def calculate_high_entropy_prior(self, entropy: float) -> float:
        return 0.3 if entropy > self.config.high_entropy_threshold else 0.1

    @validate_input_decorator(0.0, 1.0)
    def calculate_high_coherence_prior(self, coherence: float) -> float:
        return 0.6 if coherence > self.config.high_coherence_threshold else 0.2

    @validate_input_decorator(0.0, 1.0)
    def calculate_joint_probability(
        self, coherence: float, action: int, prn_influence: float
    ) -> float:
        if coherence > self.config.high_coherence_threshold:
            if action == 1:
                return prn_influence * 0.8 + (1 - prn_influence) * 0.2
            return prn_influence * 0.1 + (1 - prn_influence) * 0.7
        return 0.3

    @timer_decorator
    def calculate_probabilities_and_select_action(
        self,
        entropy: float,
        coherence: float,
        prn_influence: float,
        action: int,
    ) -> Dict[str, float]:
        # Normalizar si fuera de rango
        if entropy > 1.0:
            entropy = 1.0 / (1.0 + np.exp(-entropy))
        entropy      = float(np.clip(entropy, 0.0, 1.0))

        if coherence > 1.0:
            coherence = 1.0 / (1.0 + np.exp(-coherence))
        coherence    = float(np.clip(coherence, 0.0, 1.0))
        prn_influence = float(np.clip(prn_influence, 0.0, 1.0))

        he_prior = self.calculate_high_entropy_prior(entropy)
        hc_prior = self.calculate_high_coherence_prior(coherence)

        cond_b_a = (
            prn_influence * 0.7 + (1 - prn_influence) * 0.3
            if entropy > self.config.high_entropy_threshold
            else 0.2
        )

        posterior = self.calculate_posterior_probability(he_prior, hc_prior, cond_b_a)
        joint     = self.calculate_joint_probability(coherence, action, prn_influence)
        cond_act  = self.calculate_conditional_probability(joint, hc_prior)
        act       = 1 if cond_act > self.config.action_threshold else 0

        return {
            "action_to_take":             act,
            "high_entropy_prior":         he_prior,
            "high_coherence_prior":       hc_prior,
            "posterior_a_given_b":        posterior,
            "conditional_action_given_b": cond_act,
        }


# ============================================================================
# INTEGRACIÓN CUÁNTICO-BAYESIANA (sin sklearn ni TF)
# ============================================================================

class QuantumBayesMahalanobis(BayesLogic):
    """Combina lógica bayesiana con distancia de Mahalanobis en estados cuánticos."""

    @staticmethod
    def _empirical_inv_cov(data: np.ndarray) -> np.ndarray:
        """Inversa de la covarianza empírica (reemplaza EmpiricalCovariance)."""
        cov = np.cov(data, rowvar=False)
        # Asegurar que sea 2-D aunque data tenga 1 columna
        cov = np.atleast_2d(cov)
        try:
            return np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(cov)

    def compute_quantum_mahalanobis(
        self,
        quantum_states_A: np.ndarray,
        quantum_states_B: np.ndarray,
    ) -> np.ndarray:
        """Distancia de Mahalanobis vectorizada."""
        if quantum_states_A.ndim != 2 or quantum_states_B.ndim != 2:
            raise ValueError("Los estados deben ser matrices 2-D.")
        if quantum_states_A.shape[1] != quantum_states_B.shape[1]:
            raise ValueError("Las dimensiones de A y B deben coincidir.")

        inv_cov = self._empirical_inv_cov(quantum_states_A)
        mean_A  = np.mean(quantum_states_A, axis=0)
        diff    = quantum_states_B - mean_A
        return np.sqrt(np.einsum("ni,ij,nj->n", diff, inv_cov, diff))

    def quantum_cosine_projection(
        self,
        quantum_states: np.ndarray,
        entropy: float,
        coherence: float,
        n_step: int = 1,
    ) -> np.ndarray:
        """Proyecta estados cuánticos con cosenos + operador áureo."""
        if quantum_states.shape[1] != 2:
            raise ValueError("Se espera 'quantum_states' con 2 columnas.")

        cos_x, cos_y, cos_z = calculate_cosines(entropy, coherence)

        proj_A = quantum_states * np.array([cos_x, cos_y])
        proj_B = quantum_states * np.array([cos_x * cos_z, cos_y * cos_z])

        distances = self.compute_quantum_mahalanobis(proj_A, proj_B)

        paridad, _ = aureo_operator(n_step)
        return np.tanh(distances * paridad)

    def calculate_quantum_posterior_with_mahalanobis(
        self,
        quantum_states: np.ndarray,
        entropy: float,
        coherence: float,
    ):
        """Posterior bayesiana sobre proyecciones cuánticas."""
        projections = self.quantum_cosine_projection(quantum_states, entropy, coherence)

        # Covarianza escalar → traza/dim (reemplaza tfp.stats.covariance)
        cov_scalar  = float(np.var(projections))  # varianza = cov 1-D
        quantum_prior = cov_scalar                 # traza/1 = varianza misma

        prior_coh  = self.calculate_high_coherence_prior(coherence)
        joint      = self.calculate_joint_probability(coherence, 1, float(np.mean(projections)))
        cond       = self.calculate_conditional_probability(joint, quantum_prior)
        posterior  = self.calculate_posterior_probability(quantum_prior, prior_coh, cond)

        return posterior, projections

    def predict_quantum_state(
        self,
        quantum_states: np.ndarray,
        entropy: float,
        coherence: float,
    ):
        """Predice el siguiente estado cuántico."""
        posterior, projections = self.calculate_quantum_posterior_with_mahalanobis(
            quantum_states, entropy, coherence
        )
        next_state = np.sum(projections * posterior, axis=0)
        return next_state, posterior


# ============================================================================
# RUIDO PROBABILÍSTICO DE REFERENCIA (PRN)
# ============================================================================

class PRN:
    """Modelado del Ruido Probabilístico de Referencia."""

    def __init__(
        self,
        influence: float,
        algorithm_type: Optional[str] = None,
        **parameters,
    ):
        if not 0 <= influence <= 1:
            raise ValueError(f"Influencia debe estar en [0,1]. Valor: {influence}")
        self.influence      = influence
        self.algorithm_type = algorithm_type
        self.parameters     = parameters

    def adjust_influence(self, adjustment: float) -> None:
        new_inf = float(np.clip(self.influence + adjustment, 0.0, 1.0))
        if new_inf != self.influence + adjustment:
            print(f"⚠️  Influencia ajustada a {new_inf}")
        self.influence = new_inf

    def combine_with(self, other: "PRN", weight: float = 0.5) -> "PRN":
        if not 0 <= weight <= 1:
            raise ValueError(f"Peso debe estar en [0,1]. Valor: {weight}")
        combined_inf    = self.influence * weight + other.influence * (1 - weight)
        combined_params = {**self.parameters, **other.parameters}
        algorithm       = self.algorithm_type if weight >= 0.5 else other.algorithm_type
        return PRN(combined_inf, algorithm, **combined_params)

    def __str__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        algo   = f", algorithm={self.algorithm_type}" if self.algorithm_type else ""
        return f"PRN(influence={self.influence:.4f}{algo}{', ' + params if params else ''})"


class EnhancedPRN(PRN):
    """Extiende PRN para registrar distancias de Mahalanobis."""

    def __init__(self, influence: float = 0.5, algorithm_type: str = None, **parameters):
        super().__init__(influence, algorithm_type, **parameters)
        self.mahalanobis_records: List[float] = []

    def record_quantum_noise(
        self,
        probabilities: Dict[str, float],
        quantum_states: np.ndarray,
    ) -> Tuple[float, float]:
        """Registra ruido cuántico basado en Mahalanobis."""
        entropy    = self.calculate_entropy_from_probabilities(probabilities)
        cov        = np.cov(quantum_states, rowvar=False)
        inv_cov    = np.linalg.pinv(np.atleast_2d(cov))
        mean_state = np.mean(quantum_states, axis=0)

        diff    = quantum_states - mean_state
        dist_sq = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
        mahal   = float(np.mean(np.sqrt(dist_sq)))

        self.mahalanobis_records.append(mahal)
        return entropy, mahal

    @staticmethod
    def calculate_entropy_from_probabilities(probabilities: Dict[str, float]) -> float:
        """Entropía de Shannon."""
        probs = np.array(list(probabilities.values()))
        probs = probs[probs >= 0]
        s     = np.sum(probs)
        probs = probs / s if s > 0 else np.ones_like(probs) / len(probs)
        probs = np.clip(probs, 1e-15, 1.0)
        return float(-np.sum(probs * np.log2(probs)))


# ============================================================================
# COLAPSO DE ONDA CON ESTRUCTURA METRIPLÉCTICA
# ============================================================================

class QuantumNoiseCollapse(QuantumBayesMahalanobis):
    """Simula colapso de onda cuántico con estructura metripléctica."""

    def __init__(self, prn_influence: float = 0.5):
        super().__init__()
        self.prn          = EnhancedPRN(influence=prn_influence)
        self.aureo_n_step = 1

    @timer_decorator
    def simulate_wave_collapse_metriplectic(
        self,
        quantum_states: np.ndarray,
        density_matrix: np.ndarray,
        prn_influence: float,
        previous_action: int,
        M: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Simula colapso de onda con estructura metripléctica."""
        if M is None:
            M = np.eye(2) * 0.1

        # 1. Entropía von Neumann
        vn_ent = VonNeumannEntropy.compute_von_neumann_entropy(
            density_matrix, normalize=True, method="sigmoid"
        )

        # 2. Ruido cuántico
        mags = np.array([np.linalg.norm(s) for s in quantum_states])
        sm   = np.sum(mags)
        norm_mags = mags / sm if sm > 0 else np.ones_like(mags) / len(mags)
        probs     = {str(i): p for i, p in enumerate(norm_mags)}
        entropy, mahal_mean = self.prn.record_quantum_noise(probs, quantum_states)

        # Normalizar
        ent_n   = float(np.clip(
            1.0 / (1.0 + np.exp(-entropy)) if entropy > 1.0 else entropy, 0.0, 1.0
        ))
        mahal_n = float(np.clip(1.0 / (1.0 + np.exp(-mahal_mean)), 0.0, 1.0))

        # 3. Cosenos directores
        cos_x, cos_y, cos_z = calculate_cosines(ent_n, mahal_n)

        # 4. Coherencia
        coherence = np.exp(-mahal_n) * (cos_x + cos_y + cos_z) / 3.0

        # 5. Lógica bayesiana
        bayes = self.calculate_probabilities_and_select_action(
            entropy=ent_n,
            coherence=coherence,
            prn_influence=prn_influence,
            action=previous_action,
        )

        # 6. Proyección cuántica
        projected = self.quantum_cosine_projection(
            quantum_states, ent_n, coherence, n_step=self.aureo_n_step
        )

        # 7. Evolución metripléctica
        q_val = np.array([ent_n])
        p_val = np.array([coherence])
        metro = MetriplecticStructure.metriplectic_evolution(H, S, H, q_val, p_val, M)

        # 8. Estado colapsado final
        collapsed = float(np.sum(projected * float(bayes["action_to_take"]) * self.aureo_n_step))
        self.aureo_n_step += 1

        return {
            "collapsed_state":              collapsed,
            "action":                       bayes["action_to_take"],
            "shannon_entropy":              entropy,
            "shannon_entropy_normalized":   ent_n,
            "von_neumann_entropy":          vn_ent,
            "coherence":                    coherence,
            "mahalanobis_distance":         mahal_mean,
            "mahalanobis_normalized":       mahal_n,
            "cosines":                      (cos_x, cos_y, cos_z),
            "metriplectic_evolution":       metro,
            "bayesian_posterior":           bayes["posterior_a_given_b"],
        }

    @timer_decorator
    def objective_function_with_noise(
        self,
        quantum_states: np.ndarray,
        target_state: np.ndarray,
        entropy_weight: float = 1.0,
        n_step: int = 1,
    ) -> float:
        """Función objetivo con ruido (implementación NumPy pura)."""
        paridad, fase_mod = aureo_operator(n_step)
        target_mod = target_state.astype(np.float32) * float(fase_mod)

        fidelity = float(np.abs(np.sum(quantum_states * target_mod)) ** 2)

        mags     = np.array([np.linalg.norm(s) for s in quantum_states])
        sm       = np.sum(mags)
        norm_mag = mags / sm if sm > 0 else np.ones_like(mags) / len(mags)
        probs    = {str(i): p for i, p in enumerate(norm_mag)}
        entropy, mahal = self.prn.record_quantum_noise(probs, quantum_states)

        mahal_clipped  = np.clip(mahal, 0, 1000.0)
        mahal_term     = 1.0 - np.exp(-mahal_clipped)
        entropy_clipped = np.clip(entropy, 0, 1000.0)
        paridad_term   = float(np.abs(np.mean(quantum_states) - paridad))

        return float(
            (1.0 - fidelity)
            + entropy_weight * entropy_clipped
            + mahal_term
            + 0.5 * paridad_term
        )

    @timer_decorator
    def optimize_quantum_state(
        self,
        initial_states: np.ndarray,
        target_state: np.ndarray,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> Tuple[np.ndarray, float]:
        """Optimiza estados cuánticos con Adam (NumPy puro)."""
        params       = initial_states.astype(np.float64).copy()
        optimizer    = AdamOptimizer(lr=learning_rate)
        best_obj     = float("inf")
        best_params  = params.copy()

        for i in range(1, max_iterations + 1):
            obj_fn = lambda p: self.objective_function_with_noise(p, target_state, n_step=i)
            grads  = numerical_gradient(obj_fn, params)

            if np.any(np.isnan(grads)):
                print(f"⚠️  Deteniendo en iteración {i} por gradiente inválido")
                break

            params = optimizer.step(params, grads)
            obj    = obj_fn(params)

            if obj < best_obj:
                best_obj    = obj
                best_params = params.copy()

        return best_params, float(best_obj)


# ============================================================================
# DEMO INTEGRADA
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("⚛️  QuoreMind v1.0.0: Sistema Metripléctico Cuántico-Bayesiano")
    print("   (NumPy puro — sin TensorFlow ni dependencias externas)")
    print("=" * 80)

    # 1. Matriz de densidad
    state          = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    density_matrix = VonNeumannEntropy.density_matrix_from_state(state)

    # 2. Estados cuánticos
    np.random.seed(42)
    quantum_states = np.random.randn(10, 2)

    # 3. Sistema y matriz métrica
    collapse_system = QuantumNoiseCollapse(prn_influence=0.6)
    M = np.array([[0.1, 0.0], [0.0, 0.1]])

    # ── PARTE 1: Colapso Metripléctico ──────────────────────────────────────
    print("\n📊 1. ANÁLISIS CON ESTRUCTURA METRIPLÉCTICA (Ciclo 1)")
    print("-" * 80)

    res = collapse_system.simulate_wave_collapse_metriplectic(
        quantum_states=quantum_states,
        density_matrix=density_matrix,
        prn_influence=0.6,
        previous_action=1,
        M=M,
    )

    print(f"✓ Estado colapsado:      {res['collapsed_state']:.6f}")
    print(f"✓ Acción tomada:         {res['action']}")
    print(f"✓ Entropía Shannon(norm):{res['shannon_entropy_normalized']:.6f}")
    print(f"✓ Entropía von Neumann:  {res['von_neumann_entropy']:.6f}")
    print(f"✓ Coherencia:            {res['coherence']:.6f}")
    print(f"✓ Mahalanobis (norm):    {res['mahalanobis_normalized']:.6f}")
    print(f"✓ Evolución metriplect.: {res['metriplectic_evolution']:.6f}")
    print(f"✓ Posterior bayesiana:   {res['bayesian_posterior']:.6f}")
    cx, cy, cz = res["cosines"]
    print(f"✓ Cosenos (x,y,z):       ({cx:.4f}, {cy:.4f}, {cz:.4f})")

    # ── PARTE 2: Corchetes de Poisson ────────────────────────────────────────
    print("\n🔄 2. ANÁLISIS DE CORCHETES DE POISSON")
    print("-" * 80)

    ent_v = res["shannon_entropy_normalized"]
    coh_v = res["coherence"]
    q_t   = np.array([ent_v])
    p_t   = np.array([coh_v])

    x_obs = lambda q, p: q
    p_obs = lambda q, p: p

    pb_xh = PoissonBrackets.poisson_bracket(x_obs, H, q_t, p_t)
    pb_ph = PoissonBrackets.poisson_bracket(p_obs, H, q_t, p_t)
    df_dt = PoissonBrackets.liouville_evolution(H, x_obs, q_t, p_t)

    print(f"✓ {{x, H}}: {pb_xh:.6f}  (esperado p ≈ {coh_v:.6f})")
    print(f"✓ {{p, H}}: {pb_ph:.6f}  (esperado -q ≈ {-ent_v:.6f})")
    print(f"✓ dx/dt = {{x, H}}: {df_dt:.6f}")

    # ── PARTE 3: Metripléctica detallada ─────────────────────────────────────
    print("\n⚙️  3. ESTRUCTURA METRIPLÉCTICA DETALLADA")
    print("-" * 80)

    mb_hs = MetriplecticStructure.metriplectic_bracket(H, S, q_t, p_t, M)
    dh_dt = MetriplecticStructure.metriplectic_evolution(H, S, H, q_t, p_t, M)
    ds_dt = MetriplecticStructure.metriplectic_evolution(H, S, S, q_t, p_t, M)

    print(f"✓ [H, S]_M (metriplexico): {mb_hs:.6f}")
    print(f"✓ dH/dt (rev+disipativa):  {dh_dt:.6f}")
    print(f"✓ dS/dt (prod. entropía):  {ds_dt:.6f}")

    # ── PARTE 4: Optimización ────────────────────────────────────────────────
    print("\n📈 4. OPTIMIZACIÓN DE ESTADOS CUÁNTICOS (Adam NumPy)")
    print("-" * 80)

    target_state   = np.array([0.8, 0.2])
    initial_states = np.random.randn(5, 2)

    init_obj = collapse_system.objective_function_with_noise(
        initial_states.astype(np.float32), target_state, n_step=1
    )
    opt_states, final_obj = collapse_system.optimize_quantum_state(
        initial_states=initial_states,
        target_state=target_state,
        max_iterations=50,
        learning_rate=0.01,
    )

    print(f"✓ Objetivo inicial: {init_obj:.6f}")
    print(f"✓ Objetivo final:   {final_obj:.6f}")
    print(f"✓ Mejora:           {init_obj - final_obj:.6f}")

    # ── PARTE 5: Evolución temporal ──────────────────────────────────────────
    print("\n🔁 5. EVOLUCIÓN TEMPORAL (5 ciclos)")
    print("-" * 80)

    cur = quantum_states.copy()
    for cycle in range(5):
        r = collapse_system.simulate_wave_collapse_metriplectic(
            quantum_states=cur,
            density_matrix=density_matrix,
            prn_influence=0.6,
            previous_action=1 if cycle % 2 == 0 else 0,
            M=M,
        )
        print(f"\n  Ciclo {cycle + 1}:")
        print(f"    von Neumann:         {r['von_neumann_entropy']:.4f}")
        print(f"    Shannon (norm):      {r['shannon_entropy_normalized']:.4f}")
        print(f"    Mahalanobis (norm):  {r['mahalanobis_normalized']:.4f}")
        print(f"    Evol. metriplectic.: {r['metriplectic_evolution']:.4f}")
        print(f"    Acción:              {r['action']}")
        cur = cur + np.random.randn(*cur.shape) * 0.01

    # ── PARTE 6: Operador Áureo y λ_doble ───────────────────────────────────
    print("\n" + "=" * 80)
    print("🔷 6. OPERADOR ÁUREO Y PARÁMETRO λ_DOBLE")
    print("=" * 80)

    init_idx = np.array([[0.1 * i, 0.1 * (i + 1)] for i in range(10)], dtype=np.float32)
    H_local  = lambda q, p: 0.5 * p ** 2 + 0.5 * q ** 2
    qubits   = np.linspace(0.1, 1.0, 10)

    print("\n🟢 λ_doble — Estados Legítimos:")
    lam_leg = [lambda_doble_operator(s, H_local, qubits, golden_phase=0) for s in init_idx]
    for i, lam in enumerate(lam_leg[:3]):
        print(f"   Estado {i}: λ_doble = {lam:.6f}")
    print(f"   ... (10 total)")
    mu_leg, sigma_leg = np.mean(lam_leg), np.std(lam_leg)
    print(f"\n   Media:     {mu_leg:.6f}")
    print(f"   Desv. Est: {sigma_leg:.6f}")

    print("\n🔴 λ_doble — Estados Anómalos:")
    anom_states = np.random.randn(5, 2) * 5.0 + np.array([2.0, 2.0])
    lam_anom    = [lambda_doble_operator(s, H_local, qubits, golden_phase=1) for s in anom_states]
    for i, lam in enumerate(lam_anom):
        print(f"   Estado {i}: λ_doble = {lam:.6f}")

    threshold   = mu_leg + 3 * sigma_leg
    detected    = sum(lam > threshold for lam in lam_anom)
    print(f"\n⚠️  Umbral 3σ:              {threshold:.6f}")
    print(f"✓  Anomalías detectadas:    {detected}/{len(anom_states)}")

    # ── RESUMEN ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 RESUMEN FINAL")
    print("=" * 80)
    print(f"""
✓ Operador Áureo (φ-operator): 10 estados — modulación cuasiperiódica activa
✓ λ_doble:   umbral={threshold:.4f} | sensibilidad={detected/len(lam_anom)*100:.1f}%
✓ Poisson:   {{x,H}}={pb_xh:.4f} | {{p,H}}={pb_ph:.4f}
✓ Metriplectic.: [H,S]_M={mb_hs:.4f} | dS/dt={ds_dt:.4f}
✓ Bayesiana: posterior={res['bayesian_posterior']:.4f} | acción={'ACEPTAR' if res['action'] else 'BLOQUEAR'}
✓ Optimización: mejora={init_obj - final_obj:.4f} en 50 iteraciones
""")
print("=" * 80)
print("✅ QuoreMind v1.0.0 — Análisis completo (sin TensorFlow)")
print("=" * 80)