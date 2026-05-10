"""
fetch_data.py
Lê os retornos diários do CSV fornecido pelo professor.
Funções puras: recebem caminho, retornam dados sem efeitos colaterais.
"""

import pandas as pd

CSV_PATH = "dow30_returns.csv"


def load_returns(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Função pura: lê o CSV de retornos diários.

    O arquivo não tem coluna de data — as linhas são dias consecutivos
    do 2º semestre de 2025 (01/07/2025 a 31/12/2025).

    Args:
        path: caminho para o arquivo CSV

    Returns:
        DataFrame (n_dias x n_ativos) com retornos diários
    """
    returns = pd.read_csv(path, header=0)
    if returns.empty:
        raise ValueError(f"CSV vazio ou não encontrado: {path}")
    return returns


def get_tickers(returns: pd.DataFrame) -> list[str]:
    """
    Função pura: extrai a lista de tickers do DataFrame.

    Args:
        returns: DataFrame de retornos

    Returns:
        lista de strings com os nomes das colunas (tickers)
    """
    return list(returns.columns)


if __name__ == "__main__":
    returns = load_returns()
    tickers = get_tickers(returns)
    print(f"Dados carregados: {returns.shape[0]} dias, {returns.shape[1]} ações")
    print(f"Tickers: {tickers}")
    print(returns.head())