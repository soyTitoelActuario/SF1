import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import skew, kurtosis
import scipy.optimize as op 

API_KEY = "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"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

def preguntar_groq(pregunta):
    headers = {"Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"}
    data = {"model": "llama-3.3-70b-versatile","messages": [{"role": "user", "content": pregunta}]}
    try:
        resp = requests.post(API_URL, headers=headers, json=data)
        respuesta = resp.json()
        if "error" in respuesta:
            return f"Error de API: {respuesta['error']}"
        return respuesta["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error al conectar con Groq: {str(e)}"

def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def calcular_metricas(df):
    returns = df.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def calcular_rendimientos_portafolio(returns, weights, tickers=None):
    if isinstance(weights, pd.Series):
        w = weights
    else:
        if tickers is None:
            raise ValueError(
                "Si weights no es pd.Series, debes pasar tickers para alinear"
            )
        w = pd.Series(weights, index=tickers)
    w = w.reindex(returns.columns).fillna(0.0)
    return (returns * w).sum(axis=1)

def calcular_rendimiento_ventana(returns, window):
    if len(returns) < window:
        return np.nan
    return (1 + returns.iloc[-window:]).prod() - 1

def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

def calcular_var_cvar_ventana(returns, window):
    if len(returns) < window:
        return np.nan, np.nan
    window_returns = returns.iloc[-window:]
    return calcular_var_cvar(window_returns)

def calcular_max_drawdown(cumulative_returns):
    if isinstance(cumulative_returns, pd.DataFrame):
        return cumulative_returns.apply(calcular_max_drawdown, axis=0)
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / running_max) - 1
    max_drawdown = drawdowns.min()    
    return max_drawdown

def calcular_max_drawdown_ventana(cumulative_returns, window):
    if len(cumulative_returns) < window:
        return np.nan
    window_data = cumulative_returns.iloc[-window:]
    return calcular_max_drawdown(window_data)

def calcular_skewness(returns):
    if isinstance(returns, pd.DataFrame):
        return returns.apply(skew, axis=0)
    return skew(returns.dropna())
    

def calcular_kurtosis(returns):
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda x: kurtosis(x.dropna(), fisher=False), axis=0)
    return kurtosis(returns.dropna(), fisher=False)

def calcular_beta_vs_sp500(returns_portafolio, start_date, end_date):
    # 'SPY' es el ETF que sigue al índice S&P 500
    datos_spy = obtener_datos_acciones(['SPY'], start_date, end_date)
    returns_spy = datos_spy.pct_change().dropna()
    if isinstance(returns_spy, pd.DataFrame):
        returns_spy = returns_spy.iloc[:, 0]
    df_aligned = pd.concat([returns_portafolio, returns_spy], axis=1).dropna()
    # Definimos X (Mercado) e Y (Portafolio)
    y_port = df_aligned.iloc[:, 0] # Tu portafolio
    x_spy  = df_aligned.iloc[:, 1] # S&P 500
    covariance_matrix = np.cov(y_port, x_spy)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]    
    return beta

rf = 0.045  # tasa libre de riesgo anual (4.5%)
periods_per_year = 252  # número típico de días de trading en un año
rf_daily = (1 + rf)**(1/periods_per_year) - 1  # tasa libre de riesgo diaria

# -------------------------------------------
# 3. Sharpe Ratio
# -------------------------------------------
def sharpe_ratio(r, rf_rate=rf_daily, periods=252):
    """
    r: Serie de retornos del activo o portafolio
    rf_rate: tasa libre de riesgo (por periodo)
    periods: número de periodos en un año (252 = diario)
    """
    excess = r - rf_rate             # retornos en exceso = R_p - R_f
    mean_excess = np.mean(excess)    # E[R_p - R_f]
    std_excess = np.std(excess)      # desviación estándar σ(R_p - R_f)
    sharpe = np.sqrt(periods) * mean_excess / std_excess
    return sharpe

# -------------------------------------------
# 4. Sortino Ratio
# -------------------------------------------
def sortino_ratio(r, rf_rate=rf_daily, target=0, periods=252):
    """
    r: retornos del portafolio
    rf_rate: tasa libre de riesgo
    target: retorno mínimo aceptable (T)
    """
    excess = r - rf_rate                  # retornos sobre libre de riesgo
    downside = r[r < target] - target     # solo pérdidas respecto al objetivo
    downside_std = np.sqrt(np.mean(downside**2))  # desviación a la baja σ_d
    mean_excess = np.mean(excess)
    sortino = np.sqrt(periods) * mean_excess / downside_std
    return sortino

# -------------------------------------------
# 5. Treynor Ratio
# -------------------------------------------
def treynor_ratio(r, benchmark, rf_rate=rf_daily, periods=252):
    """
    r: retornos del portafolio
    benchmark: retornos del índice de referencia (p.ej. S&P500)
    rf_rate: tasa libre de riesgo
    """
    excess_r = r - rf_rate           # exceso del portafolio
    excess_b = benchmark - rf_rate   # exceso del benchmark
    beta = np.cov(excess_r, excess_b)[0,1] / np.var(excess_b)  # covarianza / varianza
    treynor = np.mean(excess_r) * periods / beta
    return treynor

# -------------------------------------------
# 6. Information Ratio (IR)
# -------------------------------------------
def information_ratio(r, benchmark, periods=252):
    """
    r: retornos del portafolio
    benchmark: retornos del índice de referencia
    """
    active = r - benchmark           # diferencia activa (alpha)
    ir = np.sqrt(periods) * np.mean(active) / np.std(active)  # media / volatilidad del tracking error
    return ir

# -------------------------------------------
# 7. Calmar Ratio
# -------------------------------------------
def calmar_ratio(r, periods=252):
    """
    r: retornos del portafolio
    """
    cumulative = (1 + r).cumprod()           # curva de capital
    running_max = cumulative.cummax()        # máximo acumulado
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())             # máxima caída (máx. drawdown)
    cagr = (cumulative[-1])**(periods / len(r)) - 1  # tasa compuesta anual (CAGR)
    calmar = cagr / max_dd
    return calmar


# =====================================================================
# OPTIMIZACIÓN 1: Portafolio de mínima volatilidad
# =====================================================================
def minima_varianza(expected_returns, cov_matrix):
    """
    Minimiza la volatilidad del portafolio.
    minimize( σp(w) )
    """
    n = len(expected_returns)
    
    # Función que calcula rendimiento y volatilidad del portafolio
    def portfolio_performance(w):
        ret = np.dot(expected_returns, w)  # rp = w' * μ
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))  # σp = sqrt(w' * Σ * w)
        return ret, vol
    
    x0 = np.ones(n)/n      # Condición inicial: pesos iguales
    bounds = tuple((0,1) for _ in range(n))  # Pesos entre 0 y 1 (no short)
    constraints = ({'type':'eq','fun':lambda w: np.sum(w)-1})  # Suma de pesos = 1

    # Minimizamos solo la volatilidad:
    # minimize( σp(w) )
    result = op.minimize(
        lambda w: portfolio_performance(w)[1],
        x0, 
        method='SLSQP',  # <-- IMPORTANTE: especificar el método
        constraints=constraints, 
        bounds=bounds
    )
    return result.x, portfolio_performance(result.x)

# =====================================================================
# OPTIMIZACIÓN 2: Portafolio de máximo Sharpe ratio
# =====================================================================
def maximo_sharpe(expected_returns, cov_matrix, risk_free=0.0):
    """
    Maximiza el Sharpe Ratio usando la transformación de variables de Cornuejols y Tutuncu (2006).
    
    El problema original:
        maximize (r - rf) / σ
    
    Se transforma en un problema convexo:
        minimize y' * Σ * y
        sujeto a: (μ - rf)' * y = 1
                  sum(y) >= 0
                  y >= 0
    
    Donde y = w / ((μ - rf)' * w), y luego w = y / sum(y)
    
    :param expected_returns: retornos esperados anualizados
    :param cov_matrix: matriz de covarianza anualizada
    :param risk_free: tasa libre de riesgo (default 0.0)
    :return: pesos óptimos y métricas (rendimiento, volatilidad)
    """
    if not isinstance(risk_free, (int, float)):
        raise ValueError("risk_free_rate debe ser numérico")
    
    if np.max(expected_returns) <= risk_free:
        raise ValueError(
            "al menos uno de los activos debe tener un retorno esperado mayor a la tasa libre de riesgo"
        )
    
    n = len(expected_returns)
    
    # Función que calcula rendimiento y volatilidad del portafolio
    def portfolio_performance(w):
        ret = np.dot(expected_returns, w)  # rp = w' * μ
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))  # σp = sqrt(w' * Σ * w)
        return ret, vol
    
    # Función objetivo: y' * Σ * y (en términos de la variable transformada)
    def objective(y):
        return np.dot(y.T, np.dot(cov_matrix, y))
    
    # Retornos excedentes
    excess_returns = expected_returns - risk_free
    
    # Condición inicial
    x0 = np.ones(n) / n
    
    # Bounds: y >= 0
    bounds = tuple((0, None) for _ in range(n))
    
    # Restricciones:
    # 1. (μ - rf)' * y = 1
    # 2. No necesitamos sum(y) >= 0 porque ya tenemos y >= 0
    constraints = (
        {'type': 'eq', 'fun': lambda y: np.dot(excess_returns, y) - 1}
    )
    
    # Resolver problema transformado
    result = op.minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'maxiter': 1000, 'ftol': 1e-12}
    )
    
    if not result.success:
        raise ValueError("La optimización no convergió para max_sharpe")
    
    # Transformación inversa: w = y / sum(y)
    y_optimal = result.x
    w_optimal = y_optimal / np.sum(y_optimal)
    
    # Redondear para evitar errores numéricos
    w_optimal = np.round(w_optimal, 16)
    
    return w_optimal, portfolio_performance(w_optimal)

# =====================================================================
# OPTIMIZACIÓN 3: Portafolio Markowitz (Rendimiento Objetivo)
# =====================================================================
def markowitz_rendimiento_objetivo(expected_returns, cov_matrix, target_return):
    """
    Portafolio de Markowitz: minimiza volatilidad dado un rendimiento objetivo.
    
    minimize( σp²(w) )  <-- VARIANZA, no volatilidad
    sujeto a:
        - rp(w) >= target_return
        - sum(w) = 1
        - w >= 0
    
    :param expected_returns: retornos esperados anualizados
    :param cov_matrix: matriz de covarianza anualizada
    :param target_return: rendimiento objetivo anualizado
    :return: pesos óptimos y métricas (rendimiento, volatilidad)
    """
    n = len(expected_returns)
    
    # Función que calcula rendimiento y volatilidad del portafolio
    def portfolio_performance(w):
        ret = np.dot(expected_returns, w)  # rp = w' * μ
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))  # σp = sqrt(w' * Σ * w)
        return ret, vol
    
    # Función que calcula la VARIANZA del portafolio
    def portfolio_variance(w):
        return np.dot(w.T, np.dot(cov_matrix, w))  # σp² = w' * Σ * w
    
    # Verificar que el rendimiento objetivo sea alcanzable
    max_return = np.max(expected_returns)
    if target_return > max_return:
        raise ValueError(
            f"El rendimiento objetivo ({target_return:.2%}) excede el máximo retorno posible ({max_return:.2%})"
        )
    
    x0 = np.ones(n)/n      # Condición inicial: pesos iguales
    bounds = tuple((0,1) for _ in range(n))  # Pesos entre 0 y 1 (no short)
    
    # Restricciones:
    # 1. Suma de pesos = 1
    # 2. Rendimiento >= target_return
    constraints = (
        {'type':'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type':'ineq', 'fun': lambda w: np.dot(expected_returns, w) - target_return}
    )

    # Minimizamos la VARIANZA sujeto a las restricciones
    result = op.minimize(
        portfolio_variance,  # <-- CAMBIO: minimizar varianza en lugar de volatilidad
        x0, 
        method='SLSQP',
        constraints=constraints, 
        bounds=bounds,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )
    
    if not result.success:
        raise ValueError(f"No se pudo encontrar un portafolio con rendimiento >= {target_return:.2%}")
    
    return result.x, portfolio_performance(result.x)

def market_implied_prior_returns( market_caps,risk_aversion,cov_matrix,risk_free_rate=0.0):
    """
    Calcula los retornos implícitos de equilibrio del mercado (π).

    Fórmula:
        π = δ · Σ · w_mkt + r_f

    donde:
        δ  = coeficiente de aversión al riesgo
        Σ  = matriz de covarianza
        w_mkt = pesos de mercado (capitalización relativa)

    Estos retornos representan el "prior" del modelo Black–Litterman.
    """
    mcaps = pd.Series(market_caps)
    if isinstance(cov_matrix, pd.DataFrame):
        mcaps = mcaps.reindex(cov_matrix.columns)
        if mcaps.isna().any():
            raise ValueError("market_caps no coincide con los activos del universo")
        Sigma = cov_matrix
    else:
        Sigma = pd.DataFrame(cov_matrix, index=mcaps.index, columns=mcaps.index)
    # Pesos de mercado
    w_mkt = mcaps / mcaps.sum()
    pi = risk_aversion * Sigma.dot(w_mkt) + risk_free_rate
    return pi
 
def parse_absolute_views(tickers, absolute_views):
    """
    Convierte views absolutas en matrices P y Q.

    Ejemplo:
        absolute_views = {"AAPL": 0.12, "MSFT": 0.10}

    Significa:
        E[R_AAPL] = 12%
        E[R_MSFT] = 10%
    """
    tickers = list(tickers)
    K = len(absolute_views)
    N = len(tickers)
    P = np.zeros((K, N))
    Q = np.zeros((K, 1))
    for i, (asset, view_return) in enumerate(absolute_views.items()):
        idx = tickers.index(asset)
        P[i, idx] = 1.0
        Q[i, 0] = view_return

    return P, Q

def default_omega(cov_matrix, P, tau):
    """
    Construye la matriz Omega usando el método de He–Litterman.

    Omega_k = tau · Var(view_k)

    Es decir, la incertidumbre de cada view es proporcional
    a la varianza del portafolio asociado a esa view.
    """

    return np.diag(np.diag(tau * P @ cov_matrix @ P.T))

def black_litterman_returns(cov_matrix,pi,P,Q,omega=None,tau=0.05):
    """
    Calcula los retornos posteriores Black–Litterman (μ_BL).

    Fórmula:
    """

    # Normalización de tipos
    if isinstance(cov_matrix, pd.DataFrame):
        Sigma = cov_matrix.values
        tickers = cov_matrix.columns
    else:
        Sigma = cov_matrix
        tickers = None

    pi = np.asarray(pi).reshape(-1, 1)
    Q = np.asarray(Q).reshape(-1, 1)

    if omega is None:
        omega = default_omega(Sigma, P, tau)

    # Componentes intermedios
    tauSigma = tau * Sigma
    tauSigma_Pt = tauSigma @ P.T
    A = P @ tauSigma_Pt + omega
    b = Q - P @ pi

    # Resolver sistema lineal
    x = np.linalg.solve(A, b)
    mu_bl = pi + tauSigma_Pt @ x

    if tickers is not None:
        return pd.Series(mu_bl.flatten(), index=tickers)
    return mu_bl.flatten()

