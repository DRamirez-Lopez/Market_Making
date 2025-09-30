# Market_Making
Repositorio para el semillero de IA aplicada a trading algorítmico

# Documentación — Agente de Market Making con RL

**Stack:** Gymnasium · Stable-Baselines3 (PPO) · Python 3.10+

**Archivo principal:** `avellaneda_stoikov_gym_env.py`

---

## 0) Resumen ejecutivo (paso a paso breve)
1. **Crear entorno** (Conda recomendado):
   ```bash
   conda create -n mmrl python=3.12 -y
   conda activate mmrl
   python -m pip install --upgrade pip
   pip install gymnasium numpy stable-baselines3
   pip install "torch>=2.2,<3.0" --index-url https://download.pytorch.org/whl/cpu
   ```
2. **Guardar** `avellaneda_stoikov_gym_env.py` en la carpeta de trabajo.
3. **Entrenar** (siempre con el Python del entorno):
   ```bash
   python .\avellaneda_stoikov_gym_env.py
   ```
4. **Ver resultados**: la consola imprimirá métricas de SB3 durante el entrenamiento y, al final, un **rollout de evaluación** con **Total reward** y **Final PnL**.
5. **Modificar parámetros** del entorno (volatilidad, intensidades, penalización de inventario) y de PPO para experimentar con el desempeño.

## 1) Objetivo y diseño del agente
Construir un **market maker** que cotiza **bid/ask** y gestiona inventario en un libro de órdenes, maximizando PnL con control de riesgo. El entorno implementa una versión didáctica del modelo **Avellaneda–Stoikov**:

- **Precio de reserva**: $ r_t = s_t - q_t\,\gamma\,\sigma^2\,(T-t) $, que desplaza las cotizaciones para reducir inventario.
- **Llegadas de órdenes**: procesos de Poisson con **intensidad exponencial** decreciente con la distancia a mid: $\lambda(\delta) = A e^{-k\delta}$.
- **Acción del agente**: $ (\text{skew},\ \text{half_spread}) $ (ambos en dólares, con ticks).
- **Recompensa**: $ \Delta\text{PnL} - \text{inv_penalty}\cdot q_t^2\cdot dt $ (shaping por inventario).

> **Nota:** Es un simulador base para investigación y docencia. Para uso productivo, ver 7.

---

## 2) Estructura del entorno (Gymnasium)
- **Observación** (vector de 5):
  1) `mid_price_norm`, 2) `inventory_norm`, 3) `time_remaining`, 4) `last_halfspread_norm`, 5) `last_skew_norm`.
- **Acción**: `Box(2,)` → `(skew, half_spread)` con límites y tick.
- **Dinámica del precio**: Browniano aritmético $ s_{t+dt} = s_t + \sigma\sqrt{dt} Z $.
- **Ejecuciones**: probabilidad de fill por lado en $[t, t+dt] ≈  1 - e^{-\lambda(\delta)\,dt} $. Un fill por lado por paso (unitario, configurable).
- **Terminal**: `t >= T` o `|q| >= max_inv`.
- **`info`**: `mid_price`, `cash`, `inventory`, `bid`, `ask`, `pnl`, `time`.

---

## 3) Parámetros clave
### 3.1 Configuración del entorno (`EnvConfig`)
| Parámetro | Significado | Valor por defecto |
|---|---|---|
| `T` | Horizonte del episodio (días de mercado) | `1.0` |
| `dt` | Paso temporal (fracción del día) | `1/390/10` (≈ 6 seg) |
| `sigma` | Volatilidad diaria (abs) | `2.0` |
| `s0` | Mid inicial | `100.0` |
| `gamma` | Aversión al riesgo (AS) | `0.1` |
| `A` | Nivel base de llegadas | `140.0` |
| `k` | Pendiente de intensidad vs. distancia | `1.5` |
| `tick_size` | Tamaño de tick | `0.01` |
| `max_inv` | Límite de inventario | `50` |
| `inv_penalty` | Penalización de inventario | `0.02` |
| `max_half_spread` | Cota superior del half-spread | `1.5` |
| `max_skew` | Cota superior del skew | `1.5` |
| `fill_size` | Tamaño por fill | `1` |

**Guidelines de tuning:**
- Aumenta `sigma` → más riesgo direccional; el agente debería ampliar spreads y recentrar más agresivamente.
- Aumenta `A` o reduce `k` → más fills a igual distancia; spreads pueden estrecharse.
- Sube `gamma` y/o `inv_penalty` → el agente prioriza descargar inventario (menor varianza de PnL, quizá menor retorno).

### 3.2 Hiperparámetros PPO (SB3)
| Parámetro | Valor | Comentario |
|---|---:|---|
| `n_steps` | 2048 | Trajectorias por update (↑ → gradientes más estables) |
| `batch_size` | 256 | Tamaño de lote para SGD |
| `gamma` | 0.999 | Horizonte largo (adecuado para PnL acumulado) |
| `gae_lambda` | 0.95 | Generalized Advantage Estimation |
| `learning_rate` | 3e-4 | Tasa de aprendizaje |
| `clip_range` | 0.2 | Clipping de PPO |
| `ent_coef` | 0.01 | Entropía → fomenta exploración |

---

## 4) Cómo se construye el agente (paso a paso)
1. **Modelar microestructura mínima**: define mid-price (Browniano), función de intensidades y constraints (tick, inventario, tamaños).
2. **Definir la observación**: normaliza precio/inventario y añade features de control (tiempo restante, acción anterior) para estabilidad.
3. **Definir la acción**: `(skew, half_spread)` acotados; redondea a tick y asegura `ask - bid ≥ 2*tick_size`.
4. **Cálculo de cotizaciones**: centra en **precio de reserva** y aplica `(skew, half_spread)` → `bid`, `ask`.
5. **Simular ejecuciones**: aproxima rellenos por lado con Poisson y prob. de fill en `dt`; actualiza `cash`/`q`.
6. **Evolución del precio**: avanza `s` con \(\sigma\sqrt{dt}Z\).
7. **Recompensa**: PnL marcado a mercado y penalización por inventario.
8. **Término del episodio**: por tiempo o límite de inventario.
9. **Entrenamiento PPO**: env + vectorizado (si quieres), ciclo `learn()` con millones de timesteps.
10. **Evaluación**: rollout determinístico → `Total reward`, `Final PnL`; repetir en múltiples seeds/episodios.

---

## 5) Uso e interpretación
### 5.1 Ejecución
```bash
python .\avellaneda_stoikov_gym_env.py
```
- Verás logs periódicos de SB3 (`ep_len_mean`, `ep_rew_mean`, `explained_variance`, `entropy_loss`, etc.).
- Al final: `Episode finished. Total reward: ...  Final PnL: ...`.

### 5.2 Métricas a vigilar
- **`ep_rew_mean`**: hacia ↑ con entrenamiento (ojo si se estanca en negativo).
- **`explained_variance`**: ↑ indica que la red de valor está aprendiendo la dinámica de recompensas.
- **`approx_kl`** y **`clip_fraction`**: cambios de política razonables (ni 0 ni demasiado altos).
- **`std`** (Gauss policy): debería bajar paulatinamente (menos exploración).

---

## 6) Baselines y experimentos
- **Heurística AS** (spread cerrado-forma):
  \[ \delta_b = \gamma q\sigma^2(T-t) + \frac{1}{\gamma}\ln\big(1+\tfrac{\gamma}{k}\big),\quad
     \delta_a = -\gamma q\sigma^2(T-t) + \frac{1}{\gamma}\ln\big(1+\tfrac{\gamma}{k}\big) \]
  Comparar RL vs. esta política y vs. **estrategia simétrica** (mismo spread centrado en mid).
- **Ablation**: variar `gamma`, `inv_penalty`, `A`, `k` y medir impacto en PnL/volatilidad de inventario.
- **Curriculum**: empezar con baja `sigma`/alto `A` e ir endureciendo.

---

## 7) Hacia un entorno más realista
Para pruebas serias:
- **Fees y rebates** (maker/taker), **latencia** (place/cancel), **fills parciales y cola FIFO**.
- **Replay L2/L3**: reproducir días históricos con prioridad de cola y barridos; evaluar por regímenes (alta/baja vol).
- **Marcación**: mid vs. last vs. VWAP; *slippage* si cruzas.
- **Risk checks**: límites de notional, `|q|`, kill-switch por drawdown.

> Reemplaza las \(\lambda(\delta)\) teóricas por **tablas empíricas** de \(P(\text{fill}|\delta, \text{spread}, \text{vol}, t)\).

---

## 8) Buenas prácticas
- **Semillas** y múltiples rollouts para medias/intervalos de confianza.
- **Walk-forward** por fechas;
- **Normalización de features** estable (evitar fugas de información).
- **Logging estructurado** (TensorBoard / CSV) + trazas de trades.

---

## 9) Solución de problemas frecuentes (Windows/Conda)
- `ModuleNotFoundError: gymnasium` → ejecutar siempre con `python` del entorno, no `py`.
- `Python < 3.8` → crear entorno nuevo `python=3.10`.
- Torch en CPU/GPU → instalar wheel acorde a tu hardware.

---

## 10) Snippets útiles
**Cambiar parámetros del entorno** en el ejemplo de `__main__`:
```python
from avellaneda_stoikov_gym_env import EnvConfig, make_env
cfg = EnvConfig(sigma=3.0, A=200.0, k=1.2, inv_penalty=0.03, max_inv=30)
env = make_env(cfg, seed=42)
```

**Guardar y cargar el modelo PPO**:
```python
model.save("ppo_mm.zip")
# ...
from stable_baselines3 import PPO
model = PPO.load("ppo_mm.zip", env=env, device="auto")
```

---

## 11) Referencias sugeridas
- Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a limit order book*.
- Ho, T., & Stoll, H. (1981, 1983). *Optimal dealer pricing*.
- O’Hara, M. (1995). *Market Microstructure Theory*.

---

**Contacto/Notas**: Ajusta los parámetros a tu activo y franja horaria. Para “production”, prioriza el §7 (realismo) y una batería de tests fuera de muestra.

