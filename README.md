# Portfolio Optimizer — Dow Jones (Sharpe Maximization)

Projeto 2 da disciplina de **Programação Funcional** — Insper 2026-1.

Otimização de carteiras via simulação Monte Carlo paralelizada, usando princípios de **programação funcional** (funções puras, `map`/`reduce`, sem estado mutável compartilhado).

---

## Objetivo

Encontrar a alocação entre **25 ou mais das 30 ações do índice Dow Jones** que **maximiza o Sharpe Ratio**, sujeito a:

- Carteira *long-only*: `wᵢ ≥ 0`
- Restrição de concentração: `wᵢ ≤ 0.20`
- Soma dos pesos: `Σwᵢ = 1`

Janela de dados: **2º semestre de 2025** (01/07/2025 – 31/12/2025), 126 pregões.

> O enunciado original pedia 20 de 30 (~30M combinações × 1M de sims = inviável computacionalmente). O professor relaxou para **25 ou mais ativos**, o que reduz o espaço para 174.437 combinações.

---

## Metodologia

### Métricas

Retorno anualizado da carteira:

```text
μ = mean(R @ w) × 252
```

Volatilidade anualizada:

```text
σ = sqrt(wᵀ · C · w) × sqrt(252)
```

Sharpe Ratio (com `r_free = 0`, indiferente para fins de ranking):

```text
SR = (μ − r_free) / σ
```

### Estratégia de busca (força bruta)

1. Gera todas as combinações de tamanho `k ∈ {25, 26, …, 30}` entre os 30 ativos: **174.437 combinações** no total
   - C(30,25) = 142.506
   - C(30,26) = 27.405
   - C(30,27) = 4.060
   - C(30,28) = 435
   - C(30,29) = 30
   - C(30,30) = 1
2. Para cada combinação, sorteia `N_SIMULATIONS` vetores de pesos via **Dirichlet(1,…,1)** com **rejeição** quando algum `wᵢ > 0.20`
3. Mantém a carteira de maior Sharpe encontrada (redução por Sharpe máximo)

### Pipeline funcional

```text
combinações
  |> parMap simulate_combination   # multiprocessing.Pool.imap_unordered
  |> reduce  _compare_portfolios   # functools.reduce
```

- **Funções puras** (`portfolio.py`): mesma entrada → mesma saída, sem estado global
- **Determinismo**: cada combinação `i` recebe `seed = base_seed + i`; resultado é o mesmo independentemente da ordem em que `imap_unordered` devolve
- **Imutabilidade**: o resultado de cada combinação é um `NamedTuple` (`BestPortfolio`)
- **Inicialização única por worker**: a matriz de retornos é publicada em cada processo via `Pool(initializer=…)`, evitando reenviá-la em cada task. Os workers só leem essa matriz — não há mutação compartilhada
- **Pré-cálculo da covariância**: a matriz de covariância depende só dos retornos (não dos pesos) e é calculada **uma vez por combinação**, fora do laço de simulação

---

## Estrutura

```text
files/
├── fetch_data.py        # Carrega CSV de retornos
├── portfolio.py         # Funções puras: retorno, volatilidade, Sharpe
├── simulate.py          # Monte Carlo paralelizado
├── main.py              # Pipeline principal + benchmark
├── dow30_returns.csv    # Retornos diários (jul–dez 2025), 126 × 30
├── requirements.txt
└── README.md
```

---

## Instalação

Requer **Python 3.11+**.

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Como executar

```bash
python main.py
```

O script:

1. Carrega `dow30_returns.csv`
2. Roda a otimização paralelizada (parâmetros em `main.py`)
3. Exibe a melhor carteira encontrada (ativos, pesos, retorno, vol, Sharpe)
4. Roda o benchmark **paralelo vs sequencial** (5 execuções de cada)
5. Salva tudo em `results.json`

### Configuração — `main.py`

| Variável | Padrão | Significado |
| --- | --- | --- |
| `MAX_COMBINATIONS` | `None` | Combinações usadas na otimização. `None` = todas as 174.437 |
| `N_SIMULATIONS` (em `simulate.py`) | `10_000` | Carteiras aleatórias por combinação |
| `BENCHMARK_COMBINATIONS` | `200` | Combinações usadas no benchmark |
| `BENCHMARK_RUNS` | `5` | Execuções por modo no benchmark |
| `BENCHMARK_SIMS` | `1_000` | Simulações por combinação no benchmark |

> **Sobre a varredura:** com 174k combinações × 10k simulações, o run completo termina em ~1–2h numa máquina típica. Para iterar mais rápido, defina `MAX_COMBINATIONS` para um valor menor (ex. `5_000`). A amostragem é determinística (sempre as primeiras `MAX_COMBINATIONS` na ordem `size=25, 26, …, 30`).

### Runs longos: checkpoint + resume

O script salva **checkpoints** automaticamente em `checkpoint.json` a cada `CHECKPOINT_EVERY` combinações. Se o processo morrer (queda de luz, Ctrl-C, kernel panic), basta rodar `python main.py` de novo — ele detecta o checkpoint e retoma de onde parou. Para começar do zero, `rm checkpoint.json`.

Receita prática Linux para deixar a máquina trabalhando sem dormir e independente do terminal:

```bash
tmux new -s opt
systemd-inhibit --what=sleep:idle --why="portfolio optimization" \
  python main.py 2>&1 | tee run.log
# sair do tmux sem matar:  Ctrl-b, d
# reentrar:                tmux attach -t opt
```

---

## Resultado obtido

Run completa (todas as 174.437 combinações × 10.000 simulações) executada em **~73 minutos** numa máquina Linux com 16 workers (i9, 16 threads). Resultados salvos em `results.json`.

### Melhor carteira encontrada

| Métrica | Valor |
| --- | --- |
| **Sharpe Ratio** | **4.7511** |
| Retorno anualizado | 47.26% |
| Volatilidade anualizada | 9.95% |
| Tamanho da carteira | 25 ativos |
| Tempo de otimização | 4.403,9 s (~1h13) |

### Alocação ótima

A carteira tem 25 ativos, mas a massa está concentrada em ~6 nomes (somam 79%). Os outros 19 são "lastro" para satisfazer a restrição de 25+ ativos.

| Ranking | Ativo | Peso |
| --- | --- | --- |
| 1 | JNJ | 19.50% |
| 2 | CAT | 15.12% |
| 3 | AAPL | 14.20% |
| 4 | MRK | 14.09% |
| 5 | CVX | 10.43% |
| 6 | WMT | 6.06% |
| 7 | MSFT | 3.07% |
| 8 | TRV | 2.96% |
| 9 | KO | 2.05% |
| 10 | NVDA | 1.75% |
| 11–25 | (15 ativos) | < 1.6% cada |

JNJ está no teto da restrição (`wᵢ ≤ 0.20`), o que indica que sem o cap o ótimo concentraria ainda mais nesse ativo. O tamanho ótimo da carteira foi o **menor permitido (25)**, conforme esperado: aumentar o número de ativos só piora o Sharpe quando o cap por ativo é constante.

> **Observação:** Sharpe ~4.75 é um valor inflado típico de **backtest in-sample** (otimizando e medindo na mesma janela de 126 dias). Em out-of-sample real, o número cai significativamente.

### Benchmark — paralelo vs sequencial

5 execuções de cada modo, 200 combinações × 1.000 simulações cada (configuração leve para o benchmark; a otimização principal usa 10.000 sims).

| Modo | Tempos (s) | Média (s) |
| --- | --- | --- |
| Paralelo (16 workers) | 0.99 / 0.83 / 0.81 / 0.83 / 0.80 | **0.853** |
| Sequencial (1 worker) | 3.03 / 3.01 / 2.98 / 2.96 / 2.94 | **2.985** |

**Speedup: 3.5×** com 16 workers.

O speedup ficou abaixo do "ideal" (esperaria 8–12× com 16 workers físicos) porque:

1. **Tasks pequenas** no benchmark (200 combos × 1k sims) — overhead de IPC do `multiprocessing` domina
2. **Hyperthreading** — a máquina tem 8 cores físicos × 2 threads SMT; aceleração real escala com cores físicos, não threads
3. **Memory bandwidth** — operações Numpy compartilham o mesmo barramento de memória entre processos
4. Na otimização real (10× mais sims por combo), o speedup efetivo é maior porque o overhead é amortizado.

### Conteúdo de `results.json`

```json
{
  "best_portfolio": {
    "tickers": ["AAPL", "AMGN", ..., "WMT"],
    "weights": {"JNJ": 0.195, "CAT": 0.151, ...},
    "sharpe_ratio": 4.7511,
    "annualized_return_%": 47.26,
    "annualized_volatility_%": 9.95
  },
  "benchmark": {
    "parallel_times_s":   [0.9923, 0.8299, 0.8123, 0.8316, 0.799],
    "sequential_times_s": [3.0322, 3.011, 2.9804, 2.9638, 2.9365],
    "parallel_mean_s":    0.853,
    "sequential_mean_s":  2.9848,
    "speedup_x":          3.5,
    "n_workers":          16
  },
  "optimization_time_s": 4403.86
}
```

---

## Itens da rubrica

| Item | Status |
| --- | --- |
| Linguagem multi-paradigma com elementos funcionais | Python (funções puras, `map`/`reduce`, imutabilidade) |
| Paralelização | `multiprocessing.Pool.imap_unordered` |
| Funções paralelizadas puras | Sim — sem mutação de estado compartilhado |
| README com instalação/execução | Este arquivo |
| **Opcional:** benchmark paralelo vs sequencial (≥ 5 runs) | Implementado |

---

## Por que Programação Funcional aqui?

- **Funções puras eliminam condições de corrida** → paralelismo seguro sem locks
- **`map` + `reduce`** expressam diretamente o pipeline (sortear → avaliar → escolher o melhor)
- **Imutabilidade** (`NamedTuple`) garante que workers não interfiram entre si
- **Determinismo via injeção do gerador aleatório**: passar `np.random.Generator` como argumento (em vez de usar `np.random.seed` global) mantém a função pura

---

## Dependências

| Pacote | Uso |
| --- | --- |
| `numpy` | Álgebra linear (retorno, covariância, Dirichlet) |
| `pandas` | Leitura do CSV |

---

## Autor

Projeto individual — Bruno Locatelli —  Programação Funcional — Insper 2026-1.
