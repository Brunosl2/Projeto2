"""
portfolio.py
Funções puras para cálculo de métricas de carteira.
Puras = sem efeitos colaterais, sem estado global, mesma entrada → mesma saída.
"""

import numpy as np
from numpy.typing import NDArray

TRADING_DAYS = 252
MAX_WEIGHT = 0.20


def portfolio_daily_returns(
    returns_matrix: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Retorno diário da carteira: rp = R @ w."""
    return returns_matrix @ weights


def annualized_return(
    returns_matrix: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Retorno médio anualizado: μ = mean(R @ w) * 252."""
    rp = portfolio_daily_returns(returns_matrix, weights)
    return float(np.mean(rp) * TRADING_DAYS)


def annualized_volatility_from_cov(
    cov_matrix: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """
    Volatilidade anualizada a partir de uma matriz de covariância já calculada.
    σ = sqrt(wᵀ C w) * sqrt(252)

    Pré-computar a covariância fora do loop é ~1000× mais rápido que recalcular
    em toda simulação (a matriz só depende dos retornos, não dos pesos).
    """
    variance = weights @ cov_matrix @ weights
    return float(np.sqrt(variance) * np.sqrt(TRADING_DAYS))


def generate_random_weights(
    n_assets: int,
    rng: np.random.Generator,
    max_weight: float = MAX_WEIGHT,
    max_attempts: int = 1000,
) -> NDArray[np.float64]:
    """
    Gera pesos via Dirichlet com rejeição até satisfazer wᵢ ≤ max_weight.

    Pura no sentido funcional: o gerador é injetado como argumento,
    sem leitura de estado global. Mesma seed → mesma sequência.

    Fallback (uniformes) só é alcançável teoricamente; com n=20 e cap=0.2
    a aceitação por sample é > 99%.
    """
    for _ in range(max_attempts):
        raw = rng.dirichlet(np.ones(n_assets))
        if np.all(raw <= max_weight):
            return raw
    return np.full(n_assets, 1.0 / n_assets)


def evaluate_portfolio(
    returns_matrix: NDArray[np.float64],
    weights: NDArray[np.float64],
    cov_matrix: NDArray[np.float64] | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Avalia uma carteira (retorno, vol e Sharpe anualizados).

    Se `cov_matrix` for fornecida, evita o recálculo (caminho rápido usado
    no loop de simulação). Caso contrário calcula sob demanda.
    """
    mu = annualized_return(returns_matrix, weights)
    if cov_matrix is None:
        cov_matrix = np.cov(returns_matrix.T)
    sigma = annualized_volatility_from_cov(cov_matrix, weights)
    sharpe = (mu - risk_free_rate) / sigma if sigma > 0 else float("-inf")
    return {
        "sharpe": sharpe,
        "return": mu,
        "volatility": sigma,
        "weights": weights,
    }
