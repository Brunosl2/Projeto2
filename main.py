"""
main.py
Pipeline principal do otimizador de carteiras.

Executa:
1. Leitura do CSV de retornos (dow30_returns.csv)
2. Otimização paralelizada (2º semestre 2025)
3. Benchmark paralelo vs sequencial (5 execuções cada)
4. Exibição e salvamento dos resultados
"""

import time
import json
import numpy as np
import multiprocessing as mp

from fetch_data import load_returns, get_tickers
from simulate import (
    run_parallel,
    run_parallel_checkpointed,
    run_sequential,
    N_SIMULATIONS,
)


# ── Configurações ─────────────────────────────────────────────────────────────
BENCHMARK_COMBINATIONS = 200  # combinações usadas no benchmark de velocidade
BENCHMARK_RUNS         = 5    # execuções para cada modo (paralelo e sequencial)
BENCHMARK_SIMS         = 1_000

# Espaço de busca: carteiras com 25–30 ativos = 174.437 combinações.
# Cabe na varredura completa (~2h com N_SIMULATIONS=10.000).
# Defina um valor menor para uma amostra rápida.
MAX_COMBINATIONS = None

CHECKPOINT_PATH = "checkpoint.json"
CHECKPOINT_EVERY = 20_000  # combos entre saves (1 save ≈ 1-2 min na full run)


def format_result(result, tickers: list[str]) -> dict:
    """Função pura: serializa BestPortfolio em dict legível."""
    weights_dict = {
        ticker: float(w)
        for ticker, w in zip(result.tickers, result.weights)
    }
    return {
        "tickers": list(result.tickers),
        "weights": weights_dict,
        "sharpe_ratio": round(result.sharpe, 4),
        "annualized_return_%": round(result.ret * 100, 2),
        "annualized_volatility_%": round(result.volatility * 100, 2),
    }


def run_benchmark(
    returns_matrix: np.ndarray,
    tickers: list[str],
) -> dict:
    """Benchmark: executa paralelo e sequencial N vezes e coleta tempos."""
    parallel_times   = []
    sequential_times = []

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {BENCHMARK_RUNS} execuções, {BENCHMARK_COMBINATIONS} combinações cada")
    print(f"{'='*60}")

    for i in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        run_parallel(returns_matrix, tickers,
                     n_simulations=BENCHMARK_SIMS,
                     max_combinations=BENCHMARK_COMBINATIONS,
                     base_seed=i)
        parallel_times.append(time.perf_counter() - t0)
        print(f"  Run {i+1} paralelo:    {parallel_times[-1]:.3f}s")

        t0 = time.perf_counter()
        run_sequential(returns_matrix, tickers,
                       n_simulations=BENCHMARK_SIMS,
                       max_combinations=BENCHMARK_COMBINATIONS,
                       base_seed=i)
        sequential_times.append(time.perf_counter() - t0)
        print(f"  Run {i+1} sequencial:  {sequential_times[-1]:.3f}s")

    speedup = np.mean(sequential_times) / np.mean(parallel_times)
    return {
        "parallel_times_s":   [round(t, 4) for t in parallel_times],
        "sequential_times_s": [round(t, 4) for t in sequential_times],
        "parallel_mean_s":    round(np.mean(parallel_times), 4),
        "sequential_mean_s":  round(np.mean(sequential_times), 4),
        "speedup_x":          round(speedup, 2),
        "n_workers":          mp.cpu_count(),
    }


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 1. Carregar dados ──────────────────────────────────────────────────
    print_section("1. CARREGANDO DADOS")
    returns_df  = load_returns("dow30_returns.csv")
    tickers     = get_tickers(returns_df)
    returns_np  = returns_df.values
    print(f"  Shape: {returns_np.shape[0]} dias × {returns_np.shape[1]} ações")
    print(f"  Tickers: {tickers}")

    # ── 2. Otimização principal ────────────────────────────────────────────
    print_section("2. OTIMIZAÇÃO PARALELIZADA")
    print(f"  Combinações: {'todas (174k, sizes 25–30)' if MAX_COMBINATIONS is None else MAX_COMBINATIONS}")
    print(f"  Simulações por combinação: {N_SIMULATIONS:,}")

    t_start = time.perf_counter()
    best = run_parallel_checkpointed(
        returns_np,
        tickers,
        n_simulations=N_SIMULATIONS,
        max_combinations=MAX_COMBINATIONS,
        checkpoint_path=CHECKPOINT_PATH,
        save_every=CHECKPOINT_EVERY,
        resume=True,
    )
    t_total = time.perf_counter() - t_start
    print(f"  Tempo total: {t_total:.1f}s")

    # ── 3. Resultados ──────────────────────────────────────────────────────
    print_section("3. MELHOR CARTEIRA ENCONTRADA")
    result_dict = format_result(best, tickers)
    print(f"  Ativos: {result_dict['tickers']}")
    print(f"  Sharpe Ratio:       {result_dict['sharpe_ratio']}")
    print(f"  Retorno anual:      {result_dict['annualized_return_%']}%")
    print(f"  Volatilidade anual: {result_dict['annualized_volatility_%']}%")
    print("\n  Alocação:")
    for ticker, w in sorted(result_dict["weights"].items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 50)
        print(f"    {ticker:5s}  {w*100:5.1f}%  {bar}")

    # ── 4. Benchmark ───────────────────────────────────────────────────────
    print_section("4. BENCHMARK: PARALELO vs SEQUENCIAL")
    benchmark = run_benchmark(returns_np, tickers)
    print(f"\n  Média paralelo:    {benchmark['parallel_mean_s']}s")
    print(f"  Média sequencial:  {benchmark['sequential_mean_s']}s")
    print(f"  Speedup:           {benchmark['speedup_x']}x  ({benchmark['n_workers']} workers)")

    # ── 5. Salvar JSON ─────────────────────────────────────────────────────
    print_section("5. SALVANDO RESULTADOS")
    output = {
        "best_portfolio": result_dict,
        "benchmark":      benchmark,
        "optimization_time_s": round(t_total, 2),
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  Salvo em results.json")

    print_section("CONCLUÍDO")


if __name__ == "__main__":
    mp.freeze_support()
    main()