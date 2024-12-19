import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import questionary
from questionary import Validator, ValidationError
from tabulate import tabulate

# Constantes para extensões de arquivo e nomes de colunas
XLS = ".xls"
CSV = ".csv"
XLSX = ".xlsx"
REGIAO = "Região"

# Definição de emojis para uso no script
EMOJI_SUCCESS = "✅"
EMOJI_ERROR = "❌"
EMOJI_WARNING = "⚠️"
EMOJI_QUESTION = "🤔"
EMOJI_INFO = "ℹ️"
EMOJI_REMOVAL = "🗑️"
EMOJI_DONE = "✔️"

# Configuração do logging
logging.basicConfig(
    filename='irobm_calculator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class NumberValidator(Validator):
    def validate(self, document):
        try:
            value = float(document.text)
            if value < 0:
                raise ValueError
        except ValueError:
            raise ValidationError(
                message="Por favor, insira um número válido (não negativo).",
                cursor_position=len(document.text)
            )


class IROBMCalculator:
    def __init__(self):
        self.df = pd.DataFrame()
        self.parametros = {}

    @staticmethod
    def padronizar_regiao(df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza a coluna 'Região' para maiúsculas e remove espaços em branco.
        """
        if REGIAO in df.columns:
            df[REGIAO] = df[REGIAO].astype(str).str.strip().str.upper()
        return df

    def carregar_e_visualizar_dados(self, arquivo, planilha, header_row):
        df = self.carregar_dados(arquivo, planilha, header_row)
        self.exibir_primeiras_linhas(df)
        return df

    def validar_nome_parametro(self, nome: str) -> bool:
        return len(nome.strip()) > 0 and nome.strip().upper() not in (p.upper() for p in self.parametros)

    def atualizar_dataframe_com_parametro(self, parametro_df_clean, nome_parametro):
        # Padronizar a coluna "Região"
        parametro_df_clean = self.padronizar_regiao(parametro_df_clean)

        # Padronizar a coluna "Região" no DataFrame principal
        self.df = self.padronizar_regiao(self.df)

        # Adicionar o parâmetro ao DataFrame principal com base na coluna "Região"
        self.df = self.df.merge(parametro_df_clean, on=REGIAO, how='left')

        # Renomear a coluna do parâmetro para o nome fornecido
        self.df.rename(columns={parametro_df_clean.columns[1]: nome_parametro}, inplace=True)

        # Tratar valores ausentes
        self.df.fillna({nome_parametro: 0}, inplace=True)

    def normalizar_coluna(self, coluna: str):
        soma_coluna = self.df[coluna].sum()
        if soma_coluna != 0:
            self.df[coluna] = self.df[coluna] / soma_coluna
            print(f"{EMOJI_SUCCESS} Dados da coluna '{coluna}' normalizados com sucesso.\n")
            logging.info(f"Coluna '{coluna}' normalizada. Soma: {soma_coluna}")
        else:
            print(f"{EMOJI_WARNING} A soma dos valores da coluna '{coluna}' é zero. Normalização não realizada.\n")
            logging.warning(f"Tentativa de normalizar a coluna '{coluna}' com soma zero.")

    def validar_coluna_numerica(self, coluna: pd.Series) -> bool:
        """
        Verifica se todos os valores na coluna são numéricos.
        """
        coerced = pd.to_numeric(coluna, errors='coerce')
        nao_numericos = coerced.isna() & ~coluna.isna()
        return not nao_numericos.any()

    def limpar_coluna_numerica(self, coluna: pd.Series) -> pd.Series:
        """
        Tenta limpar os dados não numéricos da coluna com base na escolha do usuário.
        """
        # Identificar valores não numéricos
        coerced = pd.to_numeric(coluna, errors='coerce')
        nao_numericos = coerced.isna() & ~coluna.isna()

        # Opção 1: Remover linhas com valores inválidos
        # Opção 2: Substituir valores inválidos por NaN
        # Opção 3: Remover caracteres não numéricos
        escolha_limpeza = questionary.select(
            f"{EMOJI_QUESTION} Como deseja limpar os dados não numéricos?",
            choices=[
                "Remover linhas com valores inválidos",
                "Substituir valores inválidos por NaN",
                "Remover caracteres não numéricos"
            ]
        ).ask()

        if escolha_limpeza == "Remover linhas com valores inválidos":
            linhas_removidas = nao_numericos.sum()
            coluna = coluna.drop(index=nao_numericos[nao_numericos].index)
            print(f"{EMOJI_SUCCESS} {linhas_removidas} linhas removidas.\n")
            logging.info(f"{linhas_removidas} linhas removidas da coluna para limpeza.")
        elif escolha_limpeza == "Substituir valores inválidos por NaN":
            coluna = coerced
            print(f"{EMOJI_SUCCESS} Valores inválidos substituídos por NaN.\n")
            logging.info("Valores inválidos na coluna substituídos por NaN.")
        elif escolha_limpeza == "Remover caracteres não numéricos":
            # Utilizar expressão regular para remover caracteres não numéricos, exceto ponto decimal
            coluna = coluna.astype(str).apply(lambda x: re.sub(r'[^\d\.]', '', x))
            # Tentar converter para numérico novamente
            coluna = pd.to_numeric(coluna, errors='coerce')
            # Verificar se ainda há valores não numéricos
            if not self.validar_coluna_numerica(coluna):
                print(
                    f"{EMOJI_ERROR} A limpeza não conseguiu remover todos os caracteres não numéricos. Operação cancelada.\n")
                logging.error("A limpeza não conseguiu remover todos os caracteres não numéricos da coluna.")
                return coluna  # Retorna a coluna com valores ainda inválidos
            else:
                print(f"{EMOJI_SUCCESS} Caracteres não numéricos removidos com sucesso.\n")
                logging.info("Caracteres não numéricos removidos da coluna.")

        # Exibir amostra dos dados limpos
        print("\n=== Amostra dos Dados Limpados ===\n")
        print(tabulate(coluna.dropna().head(5).to_frame(name=coluna.name), headers='keys', tablefmt='grid'))
        print("\n")
        logging.info(f"Amostra dos dados limpos da coluna '{coluna.name}' exibida para o usuário.")

        return coluna

    def acrescentar_area_de_estudo(self):
        print(f"\n{EMOJI_INFO} === Acrescentar Área de Estudo ===\n")
        arquivo = self.selecionar_arquivo("ibge.xlsx")
        planilha = self.selecionar_planilha(arquivo)
        self.carregar_e_visualizar_dados(arquivo, planilha, header_row=None)

        linha_inicio = self.selecionar_linha_inicio()
        areas_df = self.carregar_e_visualizar_dados(arquivo, planilha, header_row=linha_inicio)

        colunas = list(areas_df.columns)
        coluna_selecionada = self.selecionar_coluna(
            colunas,
            f"{EMOJI_INFO} Selecione a coluna que contém as áreas de estudo:"
        )

        if not coluna_selecionada:
            print(f"{EMOJI_WARNING} Nenhuma coluna selecionada. Operação cancelada.\n")
            logging.info("Operação cancelada: Nenhuma coluna de área de estudo selecionada.")
            return

        unique_values = areas_df[coluna_selecionada].dropna().unique().tolist()

        valores_para_remover = questionary.checkbox(
            f"{EMOJI_QUESTION} Selecione os valores que deseja remover da coluna '{coluna_selecionada}':",
            choices=unique_values
        ).ask()

        if valores_para_remover:
            num_removidos = areas_df[coluna_selecionada].isin(valores_para_remover).sum()
            areas_df.loc[areas_df[coluna_selecionada].isin(valores_para_remover), coluna_selecionada] = np.nan
            areas_df.dropna(inplace=True)
            print(f"{EMOJI_SUCCESS} {num_removidos} valores removidos da coluna '{coluna_selecionada}'.\n")
            logging.info(f"{num_removidos} valores removidos da coluna '{coluna_selecionada}'.")
        else:
            print(f"{EMOJI_INFO} Nenhum valor foi removido.\n")
            logging.info("Nenhum valor removido da coluna de área de estudo.")

        areas_df.rename(columns={coluna_selecionada: REGIAO}, inplace=True)
        areas_df = self.padronizar_regiao(areas_df)

        if REGIAO in self.df.columns:
            self.df = pd.concat([self.df, areas_df[REGIAO]], ignore_index=True).drop_duplicates().reset_index(drop=True)
        else:
            self.df[REGIAO] = areas_df[REGIAO]

        print(f"{EMOJI_SUCCESS} Área de estudo '{REGIAO}' adicionada com sucesso ao DataFrame.\n")
        logging.info(f"Área de estudo '{REGIAO}' adicionada ao DataFrame.")

    def acrescentar_parametro(self):
        print(f"\n{EMOJI_INFO} === Acrescentar um Parâmetro ===\n")

        nome = self._obter_nome_parametro()
        if not nome:
            return

        arquivo, planilha = self._selecionar_arquivo_e_planilha()
        if not arquivo:
            return

        self.carregar_e_visualizar_dados(arquivo, planilha, header_row=None)

        linha_inicio = self.selecionar_linha_inicio()
        parametro_df = self.carregar_e_visualizar_dados(arquivo, planilha, header_row=linha_inicio)

        coluna_dados, coluna_referencia = self._selecionar_colunas_parametro(parametro_df)
        if not coluna_dados or not coluna_referencia:
            return

        peso = self._obter_peso_parametro(nome)
        if peso is None:
            return

        if not self._verificar_regiao_existe():
            return

        parametro_df_clean = self._preparar_parametro_df(parametro_df, coluna_referencia, coluna_dados)
        if parametro_df_clean is None:
            return

        if not self.validar_coluna_numerica(parametro_df_clean[coluna_dados]):
            if not self._tratar_dados_nao_numericos(parametro_df_clean, coluna_dados):
                return

        self.atualizar_dataframe_com_parametro(parametro_df_clean, nome)

        self._processar_normalizacao(nome)

        self.parametros[nome] = peso
        print(f"{EMOJI_SUCCESS} Parâmetro '{nome}' adicionado com sucesso.\n")
        logging.info(f"Parâmetro '{nome}' adicionado com peso {peso}.")

        self._exibir_amostra_parametro(nome)

    def _obter_nome_parametro(self):
        nome = questionary.text(
            f"{EMOJI_QUESTION} Digite o nome do parâmetro:",
            validate=lambda text: self.validar_nome_parametro(text) or "Nome inválido ou já existe."
        ).ask()

        if not nome or not nome.strip():
            print(f"{EMOJI_WARNING} Nenhum nome fornecido. Operação cancelada.\n")
            logging.info("Operação cancelada: Nenhum nome de parâmetro fornecido.")
            return None

        nome = nome.strip()

        if not self.validar_nome_parametro(nome):
            print(f"{EMOJI_ERROR} Erro: Um parâmetro com o nome '{nome}' já existe.\n")
            logging.error(f"Falha ao adicionar parâmetro: Nome '{nome}' já existe.")
            return None

        return nome

    def _selecionar_arquivo_e_planilha(self):
        try:
            arquivo = self.selecionar_arquivo("ibge.xlsx")
            planilha = self.selecionar_planilha(arquivo)
            return arquivo, planilha
        except Exception as e:
            print(f"{EMOJI_ERROR} Erro ao selecionar arquivo ou planilha: {e}\n")
            logging.error(f"Erro ao selecionar arquivo ou planilha: {e}")
            return None, None

    def _selecionar_colunas_parametro(self, parametro_df):
        colunas = list(parametro_df.columns)
        coluna_dados = self.selecionar_coluna(
            colunas,
            "Selecione a coluna que contém os dados do parâmetro:"
        )

        # Selecionar a coluna de referência para alinhamento com "Região"
        print(f"{EMOJI_INFO} Selecione a coluna de referência para alinhar com '{REGIAO}':")
        coluna_referencia = self.selecionar_coluna(
            colunas,
            f"Selecione a coluna de referência (correspondente a '{REGIAO}' no DataFrame principal):"
        )

        if not coluna_referencia:
            print(f"{EMOJI_WARNING} Nenhuma coluna de referência selecionada. Operação cancelada.\n")
            logging.info("Operação cancelada: Nenhuma coluna de referência selecionada.")
            return None, None

        return coluna_dados, coluna_referencia

    def _obter_peso_parametro(self, nome):
        peso = questionary.text(
            f"{EMOJI_QUESTION} Digite o peso para o parâmetro '{nome}':",
            validate=NumberValidator
        ).ask()

        if not peso or not peso.strip():
            print(f"{EMOJI_WARNING} Nenhum peso fornecido. Operação cancelada.\n")
            logging.info("Operação cancelada: Nenhum peso fornecido para o parâmetro.")
            return None

        return float(peso)

    def _verificar_regiao_existe(self):
        if REGIAO not in self.df.columns:
            print(
                f"{EMOJI_ERROR} Erro: A coluna '{REGIAO}' não está presente no DataFrame. Adicione uma área de estudo primeiro.\n")
            logging.error(f"Falha ao adicionar parâmetro: Coluna '{REGIAO}' ausente.")
            return False
        return True

    def _preparar_parametro_df(self, parametro_df, coluna_referencia, coluna_dados):
        parametro_df_clean = parametro_df[[coluna_referencia, coluna_dados]].dropna(
            subset=[coluna_referencia, coluna_dados])
        parametro_df_clean.rename(columns={coluna_referencia: REGIAO}, inplace=True)
        return parametro_df_clean

    def _tratar_dados_nao_numericos(self, parametro_df_clean, coluna_dados):
        print(f"{EMOJI_WARNING} A coluna '{coluna_dados}' contém valores não numéricos.")
        logging.warning(f"Coluna '{coluna_dados}' contém valores não numéricos.")

        # Mostrar uma amostra dos dados não numéricos
        amostra_nao_numerica = parametro_df_clean.loc[
                                   pd.to_numeric(parametro_df_clean[coluna_dados], errors='coerce').isna(),
                                   coluna_dados
                               ].unique().tolist()[:5]
        print("Exemplo de valores inválidos:", amostra_nao_numerica)

        limpar = questionary.confirm(
            f"{EMOJI_QUESTION} Deseja tentar limpar os dados não numéricos da coluna '{coluna_dados}'?",
            default=True
        ).ask()

        if limpar:
            parametro_df_clean[coluna_dados] = self.limpar_coluna_numerica(parametro_df_clean[coluna_dados])
            # Verificar novamente se há valores não numéricos após a limpeza
            if not self.validar_coluna_numerica(parametro_df_clean[coluna_dados]):
                print(
                    f"{EMOJI_ERROR} A limpeza não conseguiu remover todos os valores não numéricos. Operação cancelada.\n")
                logging.error(
                    f"A limpeza não conseguiu remover todos os valores não numéricos da coluna '{coluna_dados}'.")
                return False
            return True
        else:
            print(f"{EMOJI_INFO} Dados não foram limpos. Operação cancelada.\n")
            logging.info("Operação cancelada: Dados não foram limpos.")
            return False

    def _processar_normalizacao(self, nome):
        # Perguntar se o usuário deseja normalizar os dados do parâmetro
        normalizar = questionary.confirm(
            f"{EMOJI_QUESTION} Deseja normalizar os dados do parâmetro '{nome}'?",
            default=False
        ).ask()

        if normalizar:
            self.normalizar_coluna(nome)
        else:
            print(f"{EMOJI_INFO} Dados do parâmetro '{nome}' não foram normalizados.\n")
            logging.info(f"Dados do parâmetro '{nome}' não foram normalizados.")

    def _exibir_amostra_parametro(self, nome):
        # Exibir amostra dos dados limpos
        print("\n=== Amostra dos Dados Limpos do Parâmetro ===\n")
        print(tabulate(self.df[[REGIAO, nome]].head(5), headers='keys', tablefmt='grid'))
        print("\n")
        logging.info(f"Amostra dos dados limpos do parâmetro '{nome}' exibida para o usuário.")

    def calcular_irobm(self):
        print(f"\n{EMOJI_INFO} === Calcular IROBM ===\n")
        if self.df.empty:
            print(f"{EMOJI_WARNING} Nenhum dado carregado para calcular o IROBM.\n")
            logging.warning("Tentativa de calcular IROBM com DataFrame vazio.")
            return

        if not self.parametros:
            print(f"{EMOJI_WARNING} Nenhum parâmetro adicionado para o cálculo do IROBM.\n")
            logging.warning("Tentativa de calcular IROBM sem parâmetros adicionados.")
            return

        # Verificar se todas as colunas dos parâmetros estão presentes no DataFrame
        parametros_faltando = [p for p in self.parametros if p not in self.df.columns]
        if parametros_faltando:
            for parametro in parametros_faltando:
                print(f"{EMOJI_WARNING} O parâmetro '{parametro}' não está presente no DataFrame.\n")
                logging.warning(f"Parâmetro '{parametro}' ausente no DataFrame durante o processamento.")
            return

        # Criar uma matriz dos parâmetros
        parametros_df = self.df[list(self.parametros.keys())].copy()

        # Verificar e tratar valores ausentes
        if parametros_df.isnull().values.any():
            print(f"{EMOJI_WARNING} Existem valores ausentes nos parâmetros. Substituindo por zero para o cálculo.\n")
            logging.warning("Valores ausentes encontrados nos parâmetros. Substituindo por zero.")
            parametros_df.fillna(0, inplace=True)

        # Criar um vetor de pesos
        pesos = np.array(list(self.parametros.values())).reshape((len(self.parametros), 1))

        # Calcular IROBM (multiplicação matricial)
        irobm = parametros_df.values @ pesos

        # Adicionar ou atualizar a coluna IROBM no DataFrame
        self.df['IROBM'] = irobm

        # Informar o usuário sobre o cálculo
        print(f"{EMOJI_SUCCESS} IROBM calculado e adicionado/atualizado no DataFrame com sucesso.\n")
        print(
            f"{EMOJI_INFO} Para visualizar a tabela atualizada, selecione a opção '4. Visualizar a tabela' no menu.\n")
        logging.info("IROBM calculado e adicionado/atualizado no DataFrame.")

    def remover_parametro(self):
        if not self.parametros:
            print(f"{EMOJI_WARNING} Nenhum parâmetro para remover.\n")
            logging.info("Tentativa de remover parâmetro sem parâmetros existentes.")
            return

        print(f"\n{EMOJI_INFO} === Remover um Parâmetro ===\n")
        nomes = list(self.parametros.keys())
        escolha = questionary.select(
            f"{EMOJI_QUESTION} Selecione o parâmetro a ser removido:",
            choices=nomes
        ).ask()

        if escolha in self.df.columns:
            self.df.drop(columns=[escolha], inplace=True)
            del self.parametros[escolha]
            print(f"{EMOJI_SUCCESS} Parâmetro '{escolha}' removido com sucesso.\n")
            logging.info(f"Parâmetro '{escolha}' removido do DataFrame e dicionário de parâmetros.")
        else:
            print(f"{EMOJI_ERROR} Erro: O parâmetro '{escolha}' não existe no DataFrame.\n")
            logging.error(f"Falha ao remover parâmetro '{escolha}': Não encontrado no DataFrame.")

    def visualizar_tabela(self):
        print(f"\n{EMOJI_INFO} === Visualizar Tabela ===\n")
        if self.df.empty:
            print(f"{EMOJI_WARNING} Nenhum dado foi carregado até o momento.\n")
            logging.info("Tentativa de visualizar tabela com DataFrame vazio.")
            return

        # Estilizar e exibir a tabela usando tabulate
        print(tabulate(self.df, headers='keys', tablefmt='grid'))
        print("\n")
        logging.info("Tabela visualizada pelo usuário.")

    def salvar_arquivo(self):
        if self.df.empty:
            print(f"{EMOJI_WARNING} Nenhum dado para salvar.\n")
            logging.info("Tentativa de salvar arquivo com DataFrame vazio.")
            return

        salvar = questionary.confirm(
            f"{EMOJI_QUESTION} Deseja salvar os dados?",
            default=True
        ).ask()

        if salvar:
            nome_arquivo = questionary.text(
                f"{EMOJI_QUESTION} Digite o nome do arquivo (sem extensão):",
                validate=lambda text: len(text.strip()) > 0 or "Nome inválido."
            ).ask()

            if not nome_arquivo or not nome_arquivo.strip():
                print(f"{EMOJI_WARNING} Nome do arquivo inválido. Operação cancelada.\n")
                logging.info("Operação de salvar arquivo cancelada: Nome inválido.")
                return

            nome_arquivo = nome_arquivo.strip()

            formato = questionary.select(
                f"{EMOJI_QUESTION} Escolha o formato do arquivo:",
                choices=["CSV", "XLSX"]
            ).ask()

            nome_completo = f"{nome_arquivo}.{formato.lower()}"

            try:
                if formato == "CSV":
                    self.df.to_csv(nome_completo, index=False)
                else:
                    self.df.to_excel(nome_completo, index=False)
                print(f"{EMOJI_SUCCESS} Dados salvos em '{nome_completo}'.\n")
                logging.info(f"Dados salvos em '{nome_completo}'.")
            except Exception as e:
                print(f"{EMOJI_ERROR} Erro ao salvar o arquivo '{nome_completo}': {e}\n")
                logging.error(f"Erro ao salvar o arquivo '{nome_completo}': {e}")
        else:
            print(f"{EMOJI_INFO} Os dados não foram salvos.\n")
            logging.info("Usuário optou por não salvar os dados.")

    def editar_nome_parametro(self):
        if not self.parametros:
            print(f"{EMOJI_WARNING} Nenhum parâmetro para editar.\n")
            logging.info("Tentativa de editar nome de parâmetro sem parâmetros existentes.")
            return

        print(f"\n{EMOJI_INFO} === Editar Nomes dos Parâmetros ===\n")
        nomes = list(self.parametros.keys())
        escolha = questionary.select(
            f"{EMOJI_QUESTION} Selecione o parâmetro cujo nome deseja editar:",
            choices=nomes
        ).ask()

        novo_nome = questionary.text(
            f"{EMOJI_QUESTION} Digite o novo nome para o parâmetro '{escolha}':",
            validate=lambda text: self.validar_nome_parametro(text) or "Nome inválido ou já existe."
        ).ask()

        if not novo_nome or not novo_nome.strip():
            print(f"{EMOJI_WARNING} Nenhum novo nome fornecido. Operação cancelada.\n")
            logging.info("Operação cancelada: Nenhum novo nome fornecido para o parâmetro.")
            return

        novo_nome = novo_nome.strip()

        if not self.validar_nome_parametro(novo_nome):
            print(f"{EMOJI_ERROR} Erro: Um parâmetro com o nome '{novo_nome}' já existe.\n")
            logging.error(f"Falha ao renomear parâmetro: Nome '{novo_nome}' já existe.")
            return

        # Renomear a coluna no DataFrame
        self.df.rename(columns={escolha: novo_nome}, inplace=True)
        # Atualizar o dicionário de parâmetros
        self.parametros[novo_nome] = self.parametros.pop(escolha)
        print(f"{EMOJI_SUCCESS} Nome do parâmetro atualizado para '{novo_nome}'.\n")
        logging.info(f"Parâmetro '{escolha}' renomeado para '{novo_nome}'.")

    def editar_peso_parametro(self):
        if not self.parametros:
            print(f"{EMOJI_WARNING} Nenhum parâmetro para editar.\n")
            logging.info("Tentativa de editar peso de parâmetro sem parâmetros existentes.")
            return

        print(f"\n{EMOJI_INFO} === Editar Peso dos Parâmetros ===\n")
        # Mostrar os pesos atuais e a soma
        print("Pesos atuais dos parâmetros:")
        total_pesos = 0
        for nome, peso in self.parametros.items():
            print(f"- {nome}: {peso}")
            total_pesos += peso
        print(f"Soma atual dos pesos: {total_pesos}\n")

        nomes = list(self.parametros.keys())
        escolha = questionary.select(
            f"{EMOJI_QUESTION} Selecione o parâmetro cujo peso deseja editar:",
            choices=nomes
        ).ask()

        novo_peso = questionary.text(
            f"{EMOJI_QUESTION} Digite o novo peso para o parâmetro '{escolha}':",
            validate=NumberValidator
        ).ask()

        if not novo_peso or not novo_peso.strip():
            print(f"{EMOJI_WARNING} Nenhum novo peso fornecido. Operação cancelada.\n")
            logging.info("Operação cancelada: Nenhum novo peso fornecido para o parâmetro.")
            return

        novo_peso = float(novo_peso)

        # Atualizar o peso no dicionário
        self.parametros[escolha] = novo_peso
        print(f"{EMOJI_SUCCESS} Peso do parâmetro '{escolha}' atualizado para {novo_peso}.\n")
        logging.info(f"Peso do parâmetro '{escolha}' atualizado para {novo_peso}.")

        # Mostrar os pesos atualizados e a nova soma
        print("Pesos atualizados dos parâmetros:")
        total_pesos = 0
        for nome, peso in self.parametros.items():
            print(f"- {nome}: {peso}")
            total_pesos += peso
        print(f"Soma atual dos pesos: {total_pesos}\n")
        logging.info(f"Somatório dos pesos atualizado: {total_pesos}")

    @staticmethod
    def sair():
        print(f"\n{EMOJI_DONE} Saindo do programa. Até mais!\n")
        logging.info("Programa encerrado pelo usuário.")
        sys.exit(0)

    def processar_dados(self):
        if self.df.empty:
            print(f"{EMOJI_WARNING} Nenhum dado para processar.\n")
            logging.warning("Tentativa de processar dados com DataFrame vazio.")
            return

        if len(self.parametros) == 0:
            print(f"{EMOJI_WARNING} Nenhum parâmetro adicionado para o cálculo do IROBM.\n")
            logging.warning("Tentativa de calcular IROBM sem parâmetros adicionados.")
            return

        # Verificar se todas as colunas necessárias estão presentes
        for nome in self.parametros.keys():
            if nome not in self.df.columns:
                print(f"{EMOJI_WARNING} O parâmetro '{nome}' não está presente no DataFrame.\n")
                logging.warning(f"Parâmetro '{nome}' ausente no DataFrame durante o processamento.")
                return

        # Normalizar os dados
        colunas_dados = list(self.parametros.keys())
        dados_normalizados = self.df[colunas_dados].div(self.df[colunas_dados].sum())

        # Calcular IROBM
        pesos = np.array(list(self.parametros.values())).reshape((len(self.parametros), 1))
        irobm = dados_normalizados.values @ pesos

        # Adicionar IROBM ao DataFrame
        self.df['IROBM'] = irobm

        # Verificar soma de IROBM
        soma_irobm = irobm.sum()
        print(f"\n{EMOJI_INFO} Soma dos valores de IROBM: {soma_irobm:.4f}")
        if np.isclose(soma_irobm, 1, atol=1e-2) or np.isclose(soma_irobm, 100, atol=1e-1):
            print(f"{EMOJI_SUCCESS} A soma dos valores de IROBM está correta.\n")
            logging.info(f"Soma dos valores de IROBM está correta: {soma_irobm}")
        else:
            print(f"{EMOJI_WARNING} A soma dos valores de IROBM NÃO está correta.\n")
            logging.warning(f"Soma dos valores de IROBM não está correta: {soma_irobm}")

    def menu(self):
        opcoes = {
            "1. Acrescentar área de estudo": self.acrescentar_area_de_estudo,
            "2. Acrescentar um parâmetro": self.acrescentar_parametro,
            "3. Remover um parâmetro": self.remover_parametro,
            "4. Visualizar a tabela": self.visualizar_tabela,
            "5. Salvar arquivo": self.salvar_arquivo,
            "6. Editar nomes dos parâmetros": self.editar_nome_parametro,
            "7. Editar peso dos parâmetros": self.editar_peso_parametro,
            "8. Calcular IROBM": self.calcular_irobm,
            "9. Sair": self.sair  # "Sair" como última opção
        }

        while True:
            escolha = questionary.select(
                f"{EMOJI_QUESTION} Selecione uma opção:",
                choices=list(opcoes.keys())
            ).ask()

            funcao = opcoes.get(escolha)
            if funcao:
                funcao()
            else:
                print(f"{EMOJI_ERROR} Opção inválida. Tente novamente.\n")
                logging.error(f"Opção inválida selecionada: {escolha}")

    # Funções auxiliares

    @staticmethod
    def selecionar_coluna(colunas, mensagem):
        """
        Apresenta uma lista de colunas para o usuário selecionar uma.
        """
        escolha = questionary.select(
            mensagem,
            choices=colunas
        ).ask()
        return escolha

    @staticmethod
    def selecionar_arquivo(default_arquivo="ibge.xlsx", tipos_validos=None):
        """
        Solicita ao usuário o caminho do arquivo de entrada (CSV ou Excel).
        """
        if tipos_validos is None:
            tipos_validos = [CSV, XLSX, XLS]
        while True:
            arquivo = questionary.text(
                f"{EMOJI_QUESTION} Digite o caminho do arquivo de entrada:",
                default=default_arquivo
            ).ask()
            if not arquivo or not arquivo.strip():
                print(f"{EMOJI_ERROR} Erro: O nome do arquivo não pode estar vazio.\n")
                continue
            arquivo = arquivo.strip('"').strip("'")
            caminho = Path(arquivo)
            if not caminho.is_file():
                print(f"{EMOJI_ERROR} Erro: O arquivo '{arquivo}' não foi encontrado. Tente novamente.\n")
                continue
            if caminho.suffix.lower() not in tipos_validos:
                tipos_str = ", ".join(tipos_validos)
                print(f"{EMOJI_ERROR} Erro: O arquivo deve ser do tipo {tipos_str}.\n")
                continue
            return caminho

    @staticmethod
    def selecionar_planilha(arquivo):
        """
        Se o arquivo for Excel com múltiplas planilhas, permite ao usuário selecionar uma.
        Retorna o nome da planilha selecionada.
        """
        if arquivo.suffix.lower() in [XLSX, XLS]:
            try:
                excel_file = pd.ExcelFile(arquivo)
                planilhas = excel_file.sheet_names
                if len(planilhas) == 1:
                    print(f"{EMOJI_INFO} O arquivo Excel contém apenas uma planilha: '{planilhas[0]}'.")
                    return planilhas[0]
                else:
                    escolha = questionary.select(
                        f"{EMOJI_QUESTION} Selecione a planilha a ser utilizada:",
                        choices=planilhas
                    ).ask()
                    return escolha
            except Exception as e:
                print(f"{EMOJI_ERROR} Erro ao ler o arquivo Excel: {e}\n")
                logging.error(f"Erro ao ler planilhas do arquivo Excel '{arquivo}': {e}")
                sys.exit(1)
        else:
            return None  # Para arquivos CSV

    @staticmethod
    def selecionar_linha_inicio():
        """
        Pergunta ao usuário qual linha a tabela deve iniciar (zero-based).
        """
        while True:
            linha = questionary.text(
                f"{EMOJI_QUESTION} Digite o número da linha onde a tabela começa (zero-based, começando em 0):",
                validate=lambda text: text.isdigit() and int(text) >= 0 or "Número inválido."
            ).ask()
            if linha and linha.strip().isdigit() and int(linha.strip()) >= 0:
                return int(linha)
            else:
                print(f"{EMOJI_ERROR} Erro: Por favor, insira um número válido.\n")

    @staticmethod
    def carregar_dados(arquivo, planilha=None, header_row: int | Sequence[int] | None = 0):
        """
        Carrega os dados do arquivo CSV ou Excel a partir da linha especificada.
        """
        while True:
            try:
                if arquivo.suffix.lower() in [XLSX, XLS]:
                    df = pd.read_excel(arquivo, sheet_name=planilha, header=header_row)
                else:
                    df = pd.read_csv(arquivo, header=header_row)
                logging.info(f"Dados carregados com sucesso do arquivo '{arquivo}'.")
                return df
            except Exception as e:
                print(f"{EMOJI_ERROR} Erro ao carregar os dados do arquivo: {e}\n")
                logging.error(f"Erro ao carregar dados do arquivo '{arquivo}': {e}")
                tentar_novamente = questionary.confirm(
                    f"{EMOJI_QUESTION} Deseja tentar carregar o arquivo novamente?",
                    default=True
                ).ask()
                if not tentar_novamente:
                    print(f"{EMOJI_DONE} Saindo do programa.\n")
                    logging.info("Usuário optou por não tentar carregar o arquivo novamente. Encerrando o programa.")
                    sys.exit(1)

    @staticmethod
    def exibir_primeiras_linhas(df, num_linhas=5):
        """
        Exibe as primeiras linhas do DataFrame.
        """
        print("\n=== Primeiras Linhas do Arquivo ===\n")
        print(tabulate(df.head(num_linhas), headers='keys', tablefmt='grid'))
        print("\n")
        logging.info(f"Primeiras {num_linhas} linhas exibidas para o usuário.")


def main():
    calculator = IROBMCalculator()
    calculator.menu()


if __name__ == "__main__":
    main()
