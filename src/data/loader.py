import pandas as pd
import os

def load_car_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../../data/raw/car_data.csv')
    return pd.read_csv(data_path)