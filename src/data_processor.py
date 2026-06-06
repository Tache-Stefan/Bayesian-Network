import pandas as pd
import os

class DataProcessor:
    def __init__(self, file_path):
        """
        Initializes DataProcessor.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Loads data from the specified file path.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def rename_columns(self):
        """
        Renames abbreviated column names to more understandable names.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please use load_data() first.")
        
        column_mapping = {
            'age': 'Age',
            'sex': 'Gender',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Cholesterol Level',
            'fbs': 'Fasting Blood Sugar',
            'restecg': 'Resting Electrocardiogram',
            'thalach': 'Maximum Heart Rate',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression',
            'slope': 'ST Slope',
            'ca': 'Major Vessels Count',
            'thal': 'Thalassemia',
            'target': 'Heart Disease'
        }
        
        self.data = self.data.rename(columns=column_mapping)
        return self.data
    
    def discretize_data(self):
        """
        Discretizes continous variables into discrete variables.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please use load_data() first.")
        
        df_processed = self.data.copy()

        df_processed['Age'] = pd.cut(
            df_processed['Age'],
            bins=[0, 45, 55, 65, 100],
            labels=['0-45', '45-55', '55-65', '65+']
        )

        df_processed['Resting Blood Pressure'] = pd.cut(
            df_processed['Resting Blood Pressure'],
            bins=[0, 120, 140, 200],
            labels=['Normal', 'Prehypertension', 'Hypertension']
        )

        df_processed['Cholesterol Level'] = pd.cut(
            df_processed['Cholesterol Level'],
            bins=[0, 200, 240, 600],
            labels=['Optimal', 'Borderline', 'High']
        )

        df_processed['Maximum Heart Rate'] = pd.cut(
            df_processed['Maximum Heart Rate'],
            bins=[0, 110, 150, 220],
            labels=['Low', 'Normal', 'High']
        )

        df_processed['ST Depression'] = pd.cut(
            df_processed['ST Depression'], 
            bins=[-1, 0.5, 1.5, 10], 
            labels=['Low', 'Moderate', 'High']
        )
        
        for col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')
        
        self.data = df_processed
        return self.data
    
    def save_processed_data(self, output_path):
        """
        Saves the processed data to the specified output path.
        """
        if self.data is None:
            raise ValueError("No data available.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)


    def pipeline(self, output_path):
        """
        Runs the full data processing pipeline: load, rename, discretize, and save.
        """
        self.load_data()
        self.rename_columns()
        self.discretize_data()
        self.save_processed_data(output_path)
