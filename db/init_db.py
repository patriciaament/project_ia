import sqlite3
import pandas as pd
import os

def init_database_and_load_data(colunas_para_remover=None):
    if colunas_para_remover is None:
        colunas_para_remover = {}

    db_name = 'base.db'

    if os.path.exists(db_name):
        os.remove(db_name)
        print(f"Arquivo '{db_name}' existente removido.")

    conn = sqlite3.connect(db_name)
    print(f"Banco de dados '{db_name}' criado com sucesso.")

    excel_dir = '../assets'
    
    arquivos_para_carregar = {
        'item_master.xlsx': 'classificacao_items',
        'pos_week.xlsx': 'pos_week',
        'rel_week.xlsx': 'relatorio_week',
        'status_sku.xlsx': 'status_sku',
        'summary_country.xlsx': 'summary_country',
        'classificacao_clientes.xlsx': 'classificacao_clientes'
    }

    for arquivo_excel, nome_tabela in arquivos_para_carregar.items():
        caminho_completo = os.path.join(excel_dir, arquivo_excel)
        
        if os.path.exists(caminho_completo):
            try:
                df = pd.read_excel(caminho_completo)

                if nome_tabela in colunas_para_remover:
                    cols_to_drop = colunas_para_remover[nome_tabela]
                    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
                    print(f"Colunas {cols_to_drop} removidas da tabela '{nome_tabela}'.")
                
                df.to_sql(nome_tabela, conn, if_exists='replace', index=False)
                
                print(f"Dados do arquivo '{arquivo_excel}' carregados para a tabela '{nome_tabela}'.")
                
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({nome_tabela});")
                colunas_criadas = [info[1] for info in cursor.fetchall()]
                print(f"Colunas criadas na tabela '{nome_tabela}': {colunas_criadas}")

            except Exception as e:
                print(f"Erro ao carregar o arquivo '{arquivo_excel}': {e}")
        else:
            print(f"Aviso: Arquivo '{caminho_completo}' não encontrado. A tabela '{nome_tabela}' não será criada.")

    conn.close()
    print("Carga de dados concluída.")

def print_database_schema():
    """
    Conecta ao banco de dados e imprime o esquema de todas as tabelas,
    incluindo nomes de colunas e tipos de dados.
    """
    db_name = 'base.db'
    if not os.path.exists(db_name):
        print(f"Erro: O banco de dados '{db_name}' não foi encontrado.")
        return

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    print("\n--- Estrutura do Banco de Dados ---")

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tabelas = [row[0] for row in cursor.fetchall()]

        for tabela in tabelas:
            print(f"\nEsquema da tabela: '{tabela}'")
            cursor.execute(f"PRAGMA table_info({tabela});")
            colunas_info = cursor.fetchall()

            for info in colunas_info:
                print(f"  - Coluna: {info[1]}, Tipo: {info[2]}")
    finally:
        conn.close()
        print("\n--- Estrutura do Banco de Dados concluída. Conexão fechada. ---")

if __name__ == '__main__':
    colunas_para_remover = {
        'classificacao_items': ['ITEM DESCRIPTION', 'Level_0', 'GRS 6', 'PB3', 'PB4', 'LICENSE', 'PARTNER', 'PARENT ITEM'],
        'relatorio_week': ['ANO','CIA','ITEM','DESCR','PACK','LIST','APRICE','BRAND','SUBBRAND','GREENDOT','NCM','EAN13','EAN14',
                           'TLP3NTLP4','DMD','SOLDOUT','M3SKU','M3PACK','GRWTMC','STATIM'],
        'summary_country': ['Up to ...','Country Group LATAM','Country Adjusted','Bid Client','Client DC','Rec Inv CY','Rec Inv PY'],
        'classificacao_clientes': ['Unnamed']
    }

    init_database_and_load_data(colunas_para_remover)
    print_database_schema()
