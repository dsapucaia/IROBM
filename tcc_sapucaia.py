# -*- coding: utf-8 -*-
"""TCC_Sapucaia.ipynb

##*Título: Correlação de Pearson*

## 1- Acesso ao Drive

Edite aqui para localizar o diretório adequado em seu drive.
"""

from google.colab import drive
drive.mount('/content/drive/')

import os
os.chdir('/content/drive/MyDrive/LFSR/Monografia Final Diego Sapucaia')

!ls

"""## 2- Leitura do conjunto de dados"""

# @title 2.1- Dados Geoespaciais
import pandas as pd

ibge = pd.read_excel('ibge.xlsx')

ibge

# @title 2.2- Dados Estatísticos
import pandas as pd

cbmerj = pd.read_excel('cbmerj.xlsx')

cbmerj

# @title 2.3- IROBM calculados
import pandas as pd

irobm_cal = pd.read_excel('IROBM_CAL.xlsx')

irobm_cal

"""## 3- União de dados"""

# prompt: Juntar os dataframes ibge e cbmerj respeitando a coluna municípios

# Merge dos dataframes
uniao = pd.merge(ibge, cbmerj, on='MUNICÍPIOS', how='inner')
uniao

"""##4- Normalização de dados"""

# prompt: Criar um novo dataframe que preserva as linas entre 0 e 91.
# normalizar as colunas de 2 em diante, dividindo-as por suas somas.

# Criando um novo DataFrame com as linhas de 0 a 91
normal = uniao.iloc[0:92].copy()

# Normalizando as colunas de 2 em diante
for col in normal.columns[1: ]:
    if pd.api.types.is_any_real_numeric_dtype(normal[col]):  # Verifica se a coluna é numérica
        col_sum = normal[col].sum()
        if col_sum != 0:  # Evita divisão por zero
            normal[col] = normal[col] / col_sum
        else:
          print(f"A soma da coluna {col} é zero. A normalização não foi realizada.")
    else:
        print(f"A coluna {col} não é numérica. A normalização não foi realizada.")

# Exibindo o novo DataFrame
normal

"""##5- Visualização de dados, outliers e coeficientes de correlação"""

# @title Definição de rotinas reutilizáveis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def computeRansac(X, y, label = 'Distância'):
  # Aplicando o Consenso de Amostras Aleatórias (RANSAC)
  ransac = RANSACRegressor(random_state=0)
  ransac.fit(X.values, y.values)

  # Identificando máscaras para separar valores discrepantes (outliers)
  mask = dict(inliers = ransac.inlier_mask_)
  mask['outliers'] = np.logical_not(mask['inliers'])
  outliers = normal[mask['outliers']]

  # Distância vertical de cada ponto para a reta modelada
  distances = np.abs(y[mask['outliers']] - ransac.predict(X[mask['outliers']].values))

  # Adiciona a distância ao DataFrame de outliers
  outliers.insert(len(outliers.columns), label, distances)

  # Ordena os outliers pela distância em ordem decrescente
  sorted_outliers = outliers.sort_values(by=[label], ascending=False)

  return ransac, mask, sorted_outliers

def computePearson(X, Y, mask = None):
  # Obtem e retorna a correlação de Pearson com ou sem máscara
  if mask is None:
    correlation, p_value = pearsonr(X, Y)
  else:
    correlation, p_value = pearsonr(X[mask], Y[mask])
  return correlation, p_value

def plotRegressionAndCorrelation(ransac, X, y, corr, mask, key = 'inliers', axes = ['X', 'y'], ax = None):
  # Obtendo a linha de modelda pela regressão no intervalo de dados
  if type(mask) == dict:
    line_X = np.linspace(X[X.columns[0]].min(), X[X.columns[0]].max(),10)[:, np.newaxis]
  else:
    line_X = np.linspace(X[mask][X.columns[0]].min(), X[mask][X.columns[0]].max(),10)[:, np.newaxis]
  line_y_ransac = ransac.predict(line_X)

  # Plotando os dados classificados e modelo de regressão linear
  colors = {'inliers': 'blue', 'outliers': 'red', 'line': 'green'}
  markers = {'inliers': '.', 'outliers': 'x', 'line': '-'}
  if ax is None:
    plt.figure(figsize=(10, 6))
    if type(mask) == dict:
      for key in mask:
        plt.scatter(X[mask[key]][X.columns[0]], y[mask[key]], color=colors[key], label=f'Conjunto dos {key} ({len(y[mask[key]])} elementos)', marker=markers[key])
    else:
      plt.scatter(X[mask][X.columns[0]], y[mask], color=colors[key], label=f'Conjunto dos {key} ({len(y[mask])} elementos)', marker=markers[key])
    plt.plot(line_X, line_y_ransac, color=colors['line'], label=f'Regressão ($p$ = {corr:.4f})')

    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.title(f'{axes[0]} vs. {axes[1]}')
    plt.legend()
    plt.show()
    return
  if type(mask) == dict:
    for key in mask:
      ax.scatter(X[mask[key]][X.columns[0]], y[mask[key]], color=colors[key], label=f'Conjunto dos {key} ({len(y[mask[key]])} elementos)', marker=markers[key])
  else:
    ax.scatter(X[mask][X.columns[0]], y[mask], color=colors[key], label=f'Conjunto dos {key} ({len(y[mask])} elementos)', marker=markers[key])
  ax.plot(line_X, line_y_ransac, color=colors['line'], label=f'Regressão ($p$ = {corr:.4f})')
  ax.set_xlabel(axes[0])
  ax.set_ylabel(axes[1])
  if ax and type(mask) == dict:
    ax.legend()

"""### 5.1- ATT vs Veículos"""

X, y = 'VEÍCULOS', 'ATT'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.2- APH vs População"""

X, y = 'POPULAÇÃO', 'APH'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.3- Incêndio Urbano vs Área Urbana"""

X, y = 'ÁREA URBANA', 'INCENDIO URBANO'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.4- Salvamento vs População"""

X, y = 'POPULAÇÃO', 'SALVAMENTO'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.5- Salvamento vs Área Urbana"""

X, y = 'ÁREA URBANA', 'SALVAMENTO'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.6- Salvamento vs Área Territorial"""

X, y = 'ÁREA TERRITORIAL', 'SALVAMENTO'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.7- Incêndio Florestal vs População"""

X, y = 'POPULAÇÃO', 'INCÊNDIO FLORESTAL'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.8- Incêndio Florestal vs Área Urbana"""

X, y = 'ÁREA URBANA', 'INCÊNDIO FLORESTAL'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.9- Incêndio Florestal vs Área Territorial"""

X, y = 'ÁREA TERRITORIAL', 'INCÊNDIO FLORESTAL'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.10- Incêndio Florestal vs Área Rural"""

X, y = 'ÁREA RURAL', 'INCÊNDIO FLORESTAL'

ransac, mask, outliers = computeRansac(normal[[X]], normal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(normal[X], normal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(normal[X], normal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, normal[[X]], normal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.11- IROBM DADOS GEOESPACIAIS vs IROBM DADOS ESTATÍSTICOS"""

X, y = 'IROBM - GEO', 'IROBM - EST'

ransac, mask, outliers = computeRansac(irobm_cal[[X]], irobm_cal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(irobm_cal[X], irobm_cal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(irobm_cal[X], irobm_cal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, irobm_cal[[X]], irobm_cal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, irobm_cal[[X]], irobm_cal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.12- IROBM DADOS GEOESPACIAIS vs IROBM COM ATT"""

X, y = 'IROBM - GEO', 'IROBM - ATT'

ransac, mask, outliers = computeRansac(irobm_cal[[X]], irobm_cal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(irobm_cal[X], irobm_cal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(irobm_cal[X], irobm_cal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, irobm_cal[[X]], irobm_cal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, irobm_cal[[X]], irobm_cal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.13- IROBM DADOS GEOESPACIAIS vs IROBM COM INCÊNDIO URBANO"""

X, y = 'IROBM - GEO', 'IROBM - IU'

ransac, mask, outliers = computeRansac(irobm_cal[[X]], irobm_cal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(irobm_cal[X], irobm_cal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(irobm_cal[X], irobm_cal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, irobm_cal[[X]], irobm_cal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, irobm_cal[[X]], irobm_cal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()

"""### 5.14- IROBM DADOS GEOESPACIAIS vs IROBM COM ATT E INCÊNDIO URBANO"""

X, y = 'IROBM - GEO', 'IROBM - ATT e IU'

ransac, mask, outliers = computeRansac(irobm_cal[[X]], irobm_cal[y])
print(f"{len(outliers)} outliers ordenados por distância (regressão {X} vs {y})\n")
print(outliers[['MUNICÍPIOS', 'Distância']])

correlation_with_outliers, _ = computePearson(irobm_cal[X], irobm_cal[y])
print(f"\nCorrelação com outliers: {correlation_with_outliers}")

correlation_without_outliers, _ = computePearson(irobm_cal[X], irobm_cal[y], mask['inliers'])
print(f"Correlação sem outliers: {correlation_without_outliers}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'{X} vs {y}')
plotRegressionAndCorrelation(ransac, irobm_cal[[X]], irobm_cal[y], correlation_with_outliers, mask, axes = [X, y], ax=axs[0])
plotRegressionAndCorrelation(ransac, irobm_cal[[X]], irobm_cal[y], correlation_without_outliers, mask['inliers'], axes = [X, y], ax=axs[1])
plt.show()