import sqlite3
import pandas as pd

def run_sql_query(query):
    conn = sqlite3.connect('base.db')
    try:
        results_df = pd.read_sql_query(query, conn)
        return results_df
        
    except Exception as e:
        raise Exception(f"Erro ao executar a consulta SQL: {e}")
        
    finally:
        conn.close()