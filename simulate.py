"""
simulate.py
Motor de simulação Monte Carlo paralelizado.

Pipeline funcional:
    combinações
      |> parMap simulate_combination   (multiprocessing.Pool.imap_unordered)
      |> reduce _compare_portfolios    (acha o de maior Sharpe)

Sobre paralelismo e pureza:
- A matriz de retornos e a lista de tickers são constantes do problema.
  Em vez de reenviá-las em cada task (custo enorme de pickle), elas são
  publicadas uma única vez por worker via Pool(initializer=...).
- `simulate_combination` permanece referencialmente transparente: para a
  mesma tupla (combo, n_simulations, seed) e a mesma matriz inicializada,
  produz sempre a mesma saída — não há mutação de estado compartilhado.
"""

import json
import os
import time
import numpy as np
import multiprocessing as mp
from itertools import combinations
from functools import reduce
from typing import NamedTuple

from portfolio import (
    generate_random_weights,
    evaluate_portfolio,
)

N_SIMULATIONS = 10_000
N_ASSETS_TOTAL = 30
# Requisito relaxado pelo professor: "carteiras com 25 ativos ou mais".
# Varremos todos os tamanhos em [MIN_ASSETS_SELECT, N_ASSETS_TOTAL].
MIN_ASSETS_SELECT = 25

# Estado por-worker preenchido por _init_worker. Não é mutável após init.
_RETURNS: np.ndarray | None = None
_TICKERS: list[str] | None = None


class BestPortfolio(NamedTuple):
    """Resultado imutável de uma combinação (seguro entre processos)."""
    tickers: tuple
    weights: np.ndarray
    sharpe: float
    ret: float
    volatility: float


def _init_worker(returns_matrix: np.ndarray, tickers: list[str]) -> None:
    """Roda uma única vez por processo filho — publica os dados do problema."""
    global _RETURNS, _TICKERS
    _RETURNS = returns_matrix
    _TICKERS = tickers


def simulate_combination(args: tuple) -> BestPortfolio:
    """
    Sorteia n_simulations carteiras para a combinação dada e devolve a melhor.

    Args:
        args: (combo_indices, n_simulations, seed)
    """
    combo_indices, n_simulations, seed = args
    sub_returns = _RETURNS[:, list(combo_indices)]
    combo_tickers = tuple(_TICKERS[i] for i in combo_indices)
    n_assets = len(combo_indices)

    # Covariância depende só dos retornos — calcula uma vez por combinação.
    cov_matrix = np.cov(sub_returns.T)
    rng = np.random.default_rng(seed)

    best_sharpe = float("-inf")
    best_result = None

    for _ in range(n_simulations):
        weights = generate_random_weights(n_assets, rng)
        result = evaluate_portfolio(sub_returns, weights, cov_matrix=cov_matrix)
        if result["sharpe"] > best_sharpe:
            best_sharpe = result["sharpe"]
            best_result = result

    if best_result is None:
        equal_w = np.full(n_assets, 1.0 / n_assets)
        best_result = evaluate_portfolio(sub_returns, equal_w, cov_matrix=cov_matrix)

    return BestPortfolio(
        tickers=combo_tickers,
        weights=best_result["weights"],
        sharpe=best_result["sharpe"],
        ret=best_result["return"],
        volatility=best_result["volatility"],
    )


def _compare_portfolios(a: BestPortfolio, b: BestPortfolio) -> BestPortfolio:
    """Redução pura: vence o de maior Sharpe."""
    return a if a.sharpe >= b.sharpe else b


_EMPTY_PORTFOLIO = BestPortfolio(
    tickers=(),
    weights=np.array([]),
    sharpe=float("-inf"),
    ret=float("nan"),
    volatility=float("nan"),
)


def _all_combinations():
    """Itera todas as combinações com tamanho em [MIN_ASSETS_SELECT, N_ASSETS_TOTAL]."""
    for k in range(MIN_ASSETS_SELECT, N_ASSETS_TOTAL + 1):
        yield from combinations(range(N_ASSETS_TOTAL), k)


def _build_args(
    n_simulations: int,
    max_combinations: int | None,
    base_seed: int,
):
    """Gera lazy a tupla de argumentos (combo, n_sims, seed) para cada combinação."""
    for i, combo in enumerate(_all_combinations()):
        if max_combinations is not None and i >= max_combinations:
            break
        yield (combo, n_simulations, base_seed + i)


def _count_combos(max_combinations: int | None) -> int:
    from math import comb
    total = sum(comb(N_ASSETS_TOTAL, k)
                for k in range(MIN_ASSETS_SELECT, N_ASSETS_TOTAL + 1))
    return total if max_combinations is None else min(total, max_combinations)


def run_parallel(
    returns_matrix: np.ndarray,
    tickers: list[str],
    n_simulations: int = N_SIMULATIONS,
    n_workers: int | None = None,
    max_combinations: int | None = None,
    base_seed: int = 42,
    chunksize: int = 50,
) -> BestPortfolio:
    """parMap simulate_combination |> reduce _compare_portfolios."""
    n_combos = _count_combos(max_combinations)
    workers = n_workers or mp.cpu_count()
    print(f"Combinações: {n_combos:,} | Sims/comb: {n_simulations:,} "
          f"| Workers: {workers}")

    args_iter = _build_args(n_simulations, max_combinations, base_seed)

    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(returns_matrix, tickers),
    ) as pool:
        results = pool.imap_unordered(simulate_combination, args_iter, chunksize=chunksize)
        return reduce(_compare_portfolios, results, _EMPTY_PORTFOLIO)


def run_sequential(
    returns_matrix: np.ndarray,
    tickers: list[str],
    n_simulations: int = N_SIMULATIONS,
    max_combinations: int | None = None,
    base_seed: int = 42,
) -> BestPortfolio:
    """Versão single-process — usada apenas no benchmark de comparação."""
    _init_worker(returns_matrix, tickers)
    args_iter = _build_args(n_simulations, max_combinations, base_seed)
    return reduce(_compare_portfolios, map(simulate_combination, args_iter), _EMPTY_PORTFOLIO)


# ── Checkpointing ─────────────────────────────────────────────────────────────
# Útil em runs longos (~horas/dias). Salva o melhor encontrado + quantas
# combinações já foram processadas, para retomar de onde parou em caso de
# crash, falta de luz, ou Ctrl-C.

def _serialize_portfolio(p: BestPortfolio) -> dict:
    return {
        "tickers": list(p.tickers),
        "weights": p.weights.tolist() if isinstance(p.weights, np.ndarray) else list(p.weights),
        "sharpe": p.sharpe,
        "ret": p.ret,
        "volatility": p.volatility,
    }


def _deserialize_portfolio(d: dict) -> BestPortfolio:
    return BestPortfolio(
        tickers=tuple(d["tickers"]),
        weights=np.array(d["weights"]),
        sharpe=d["sharpe"],
        ret=d["ret"],
        volatility=d["volatility"],
    )


def _save_checkpoint(path: str, best: BestPortfolio, n_done: int, n_total: int) -> None:
    """Escreve checkpoint atomicamente (write tmp → rename)."""
    data = {
        "n_done": n_done,
        "n_total": n_total,
        "best": _serialize_portfolio(best),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _load_checkpoint(path: str) -> tuple[BestPortfolio, int] | tuple[None, int]:
    """Lê checkpoint. Retorna (best, n_done) ou (None, 0) se não existir."""
    if not os.path.exists(path):
        return None, 0
    with open(path) as f:
        data = json.load(f)
    return _deserialize_portfolio(data["best"]), data["n_done"]


def run_parallel_checkpointed(
    returns_matrix: np.ndarray,
    tickers: list[str],
    n_simulations: int = N_SIMULATIONS,
    n_workers: int | None = None,
    max_combinations: int | None = None,
    base_seed: int = 42,
    chunksize: int = 50,
    checkpoint_path: str = "checkpoint.json",
    save_every: int = 20_000,
    resume: bool = True,
) -> BestPortfolio:
    """
    Mesma lógica de `run_parallel`, mas salva progresso em disco para runs
    longos. Se `checkpoint_path` existir e `resume=True`, retoma de onde parou.

    Funções paralelizadas continuam puras: o checkpointing acontece só no
    consumer (processo principal), que coleta os resultados do `imap_unordered`.
    """
    n_total = _count_combos(max_combinations)

    best_so_far: BestPortfolio = _EMPTY_PORTFOLIO
    skip = 0
    if resume:
        loaded, skip = _load_checkpoint(checkpoint_path)
        if loaded is not None:
            best_so_far = loaded
            print(f"Retomando do checkpoint: {skip:,}/{n_total:,} combos "
                  f"({skip/n_total*100:.2f}%) | melhor Sharpe={best_so_far.sharpe:.4f}")
            if skip >= n_total:
                print("Checkpoint indica corrida completa — nada a fazer.")
                return best_so_far

    # Gera args pulando os primeiros `skip` (seeds determinísticas garantem
    # que pular o índice i é equivalente a tê-lo processado antes).
    def _args_iter():
        for i, combo in enumerate(_all_combinations()):
            if i < skip:
                continue
            if max_combinations is not None and i >= max_combinations:
                break
            yield (combo, n_simulations, base_seed + i)

    workers = n_workers or mp.cpu_count()
    print(f"Combinações: {n_total:,} (a fazer: {n_total - skip:,}) "
          f"| Sims/comb: {n_simulations:,} | Workers: {workers}")
    print(f"Checkpoint a cada {save_every:,} combos → {checkpoint_path}")

    n_done = skip
    last_save = n_done
    started_at = time.time()

    try:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(returns_matrix, tickers),
        ) as pool:
            for result in pool.imap_unordered(
                simulate_combination, _args_iter(), chunksize=chunksize
            ):
                if result.sharpe > best_so_far.sharpe:
                    best_so_far = result
                n_done += 1

                if n_done - last_save >= save_every:
                    _save_checkpoint(checkpoint_path, best_so_far, n_done, n_total)
                    last_save = n_done
                    elapsed = time.time() - started_at
                    rate = (n_done - skip) / elapsed if elapsed > 0 else 0.0
                    eta_h = (n_total - n_done) / rate / 3600 if rate > 0 else float("inf")
                    print(f"  {n_done:>10,}/{n_total:,}  ({n_done/n_total*100:5.2f}%)  "
                          f"Sharpe={best_so_far.sharpe:.4f}  "
                          f"rate={rate:,.0f}/s  ETA={eta_h:.2f}h")
    except KeyboardInterrupt:
        print(f"\nInterrompido em {n_done:,}/{n_total:,}. Salvando checkpoint…")
        _save_checkpoint(checkpoint_path, best_so_far, n_done, n_total)
        raise

    _save_checkpoint(checkpoint_path, best_so_far, n_done, n_total)
    return best_so_far
