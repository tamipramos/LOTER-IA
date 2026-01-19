import sqlite3
from typing import Dict, List, Any, Optional, Tuple

class SQLiteController:
    def __init__(self, db_path: str = "database.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def create_table(self, table_name: str, columns: Dict[str, str]):
        columns_sql = ', '.join([f"{name} {type_}" for name, type_ in columns.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql})"
        self.cursor.execute(query)
        self.conn.commit()

    def insert(self, table_name: str, data: Dict[str, Any]):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        values = tuple(data.values())
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(query, values)
        self.conn.commit()

    def get_all(self, table_name: str) -> List[Dict[str, Any]]:
        self.cursor.execute(f"SELECT * FROM {table_name}")
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def get_by_id(self, table_name: str, record_id: int) -> Optional[Dict[str, Any]]:
        self.cursor.execute(f"SELECT * FROM {table_name} WHERE id = ?", (record_id,))
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def update_by_id(self, table_name: str, record_id: int, new_data: Dict[str, Any]):
        columns = ', '.join([f"{key}=?" for key in new_data])
        values = list(new_data.values())
        values.append(record_id)
        query = f"UPDATE {table_name} SET {columns} WHERE id = ?"
        self.cursor.execute(query, values)
        self.conn.commit()

    def delete_by_id(self, table_name: str, record_id: int):
        self.cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (record_id,))
        self.conn.commit()
        
    def get_by_id_custom(self, query: str, params: Tuple) -> Optional[Dict[str, Any]]:
        self.cursor.execute(query, params)
        row = self.cursor.fetchone()
        return dict(row) if row else None
      
    def close(self):
        self.conn.close()
        
    def execute_custom_query(self, query, params):
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    