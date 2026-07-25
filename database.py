"""
Database utilities for experiment tracking
"""
import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List
import os


class ExperimentDB:
    """
    SQLite database for storing ML experiments
    """
    
    def __init__(self, db_path: str = "experiments.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                dataset_name TEXT,
                n_samples INTEGER,
                n_features INTEGER,
                target_column TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preprocessing_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                step_type TEXT,
                step_config TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                model_name TEXT NOT NULL,
                metrics TEXT NOT NULL,
                feature_importance TEXT,
                training_time REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_experiment(self, 
                       experiment_name: str,
                       task_type: str,
                       dataset_info: Dict[str, Any],
                       preprocessing_steps: List[Dict[str, Any]],
                       model_results: Dict[str, Dict[str, Any]]) -> int:
        """
        Save complete experiment to database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert experiment
        cursor.execute('''
            INSERT INTO experiments 
            (experiment_name, task_type, dataset_name, n_samples, n_features, target_column)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            experiment_name,
            task_type,
            dataset_info.get('name', 'Unknown'),
            dataset_info.get('n_samples', 0),
            dataset_info.get('n_features', 0),
            dataset_info.get('target_column', '')
        ))
        
        experiment_id = cursor.lastrowid
        
        # Insert preprocessing steps
        for step in preprocessing_steps:
            cursor.execute('''
                INSERT INTO preprocessing_steps 
                (experiment_id, step_type, step_config)
                VALUES (?, ?, ?)
            ''', (
                experiment_id,
                step.get('type', ''),
                json.dumps(step.get('config', {}))
            ))
        
        # Insert model results
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            importance = results.get('feature_importance', None)
            
            cursor.execute('''
                INSERT INTO model_results 
                (experiment_id, model_name, metrics, feature_importance, training_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                model_name,
                json.dumps(metrics),
                json.dumps(importance),
                results.get('training_time', 0)
            ))
        
        conn.commit()
        conn.close()
        
        return experiment_id
    
    def load_experiments(self) -> pd.DataFrame:
        """
        Load all experiments
        """
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query('''
            SELECT 
                e.id,
                e.experiment_name,
                e.task_type,
                e.dataset_name,
                e.n_samples,
                e.n_features,
                e.target_column,
                e.created_at,
                COUNT(DISTINCT m.id) as n_models
            FROM experiments e
            LEFT JOIN model_results m ON e.id = m.experiment_id
            GROUP BY e.id
            ORDER BY e.created_at DESC
        ''', conn)
        
        conn.close()
        return df
    
    def load_experiment_details(self, experiment_id: int) -> Dict[str, Any]:
        """
        Load details of a specific experiment
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get experiment info
        cursor.execute('''
            SELECT * FROM experiments WHERE id = ?
        ''', (experiment_id,))
        
        experiment = cursor.fetchone()
        
        if not experiment:
            conn.close()
            return None
        
        # Get preprocessing steps
        cursor.execute('''
            SELECT step_type, step_config FROM preprocessing_steps 
            WHERE experiment_id = ?
        ''', (experiment_id,))
        
        preprocessing = [
            {
                'type': row[0],
                'config': json.loads(row[1])
            }
            for row in cursor.fetchall()
        ]
        
        # Get model results
        cursor.execute('''
            SELECT model_name, metrics, feature_importance, training_time 
            FROM model_results 
            WHERE experiment_id = ?
        ''', (experiment_id,))
        
        models = {}
        for row in cursor.fetchall():
            models[row[0]] = {
                'metrics': json.loads(row[1]),
                'feature_importance': json.loads(row[2]) if row[2] else None,
                'training_time': row[3]
            }
        
        conn.close()
        
        return {
            'experiment_id': experiment[0],
            'experiment_name': experiment[1],
            'task_type': experiment[2],
            'dataset_name': experiment[3],
            'n_samples': experiment[4],
            'n_features': experiment[5],
            'target_column': experiment[6],
            'created_at': experiment[7],
            'preprocessing': preprocessing,
            'models': models
        }
    
    def delete_experiment(self, exp_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM experiments WHERE id = ?",
            (int(exp_id),)
        )

        conn.commit()
        conn.close()
    
    def get_best_models(self, task_type: str, limit: int = 10) -> pd.DataFrame:
        """
        Get best performing models across all experiments
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                e.experiment_name,
                m.model_name,
                m.metrics,
                e.created_at
            FROM model_results m
            JOIN experiments e ON m.experiment_id = e.id
            WHERE e.task_type = ?
            ORDER BY e.created_at DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(task_type, limit))
        conn.close()
        
        # Parse metrics
        if len(df) > 0:
            df['metrics_parsed'] = df['metrics'].apply(json.loads)
        
        return df
