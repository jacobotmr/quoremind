# QuoreMind v1.0.0

### Sistema Metripléctico Cuántico-Bayesiano

QuoreMind es un framework de lógica bayesiana avanzado que integra estructuras metriplécticas y operadores cuánticos para el modelado de sistemas de información dinámicos. Diseñado bajo el rigor de **El Mandato Metriplético**, el sistema garantiza estabilidad numérica y coherencia física mediante la competencia entre términos conservativos y disipativos.

## 🌌 Fundamentos Físicos

Siguiendo el "Mandato Metriplético", QuoreMind define explícitamente la dinámica del sistema mediante dos corchetes ortogonales:

1. **Componente Simpléctica**: Generada por un Hamiltoniano $H$ (Energía) para movimientos reversibles.
    * `d_symp = {u, H}` (Estructura de Poisson).
2. **Componente Métrica**: Generada por un Potencial de Disipación $S$ (Entropía) para relajación funcional.
    * `d_metr = [u, S]` (Potencial disipativo).

### Ecuación Maestra de Evolución

$$ \frac{df}{dt} = \{f, H\} + [f, S]_M $$

Donde $\{, \}$ representa el corchete de Poisson y $[, ]_M$ representa la interacción métrica disipativa mediada por la matriz métrica $M$.

## 🛠️ Características Principales

* **Estructura Metripléctica**: Simulación de evolución temporal combinando entropía y energía.
* **Operador Áureo ($O_n$)**: Modulación de fase cuasiperiódica mediante la razón áurea ($\phi \approx 1.618$) para evitar singularidades y estructurar el vacío de información.
* **Pre-Análisis de Mahalanobis**: Uso de la distancia de Mahalanobis vectorizada para evaluar la consistencia de estados cuánticos.
* **Lógica Bayesiana Cuántica**: Motor de inferencia para el cálculo de probabilidades posteriores $P(A|B)$ sobre estados colapsados.
* **Optimización Adam (NumPy Puro)**: Algoritmo de optimización de primer orden implementado sin dependencias externas pesadas (TensorFlow/PyTorch).

## 🧬 Analogía Rigurosa

QuoreMind implementa el **Nivel 3 de Isomorfismo Físico Operacional**, permitiendo la transferencia de intuición entre:
* **Dinámica de Fluidos**: Viscosidad e Inercia.
* **Información Cuántica**: Decoherencia (Lindblad) y Dinámica Unitaria (Schrödinger).

## 🚀 Instalación y Uso

### Instalación

Asegúrate de tener instaladas las dependencias científicas base:

```bash
pip install numpy scipy
```

### Ejecutar Demo Integrada

El archivo `quoremind.py` contiene una demo completa de optimización y evolución metripléctica:

```bash
python quoremind.py
```

## 🧪 Verificación (Pytest)

La integridad del sistema se valida mediante pruebas de reversibilidad y límites asintóticos:

```bash
pytest tests/
```

---
**Autor:** Jacobo Tlacaelel Mina Rodriguez.
**Diseño:** Basado en principios de simetría estructural y física teórica.
