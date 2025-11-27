import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.features.target_engineering import create_premium_target

class CarPricePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.columns_to_drop = ['carheight', 'stroke', 'compressionratio',
                                'peakrpm', 'car_ID', 'CarName']
        self.new_features = ['power_to_weight', 'mpg_avg', 'size_ratio']

    def _extract_brand(self, df):
        brand_correction = {
            'maxda': 'mazda', 'porcshce': 'porsche',
            'toyouta': 'toyota', 'vokswagen': 'volkswagen',
            'vw': 'volkswagen'
        }
        df['CarName'] = df['CarName'].replace(brand_correction, regex=True)
        df['CarName'] = df['CarName'].str.lower()
        df['brand'] = df['CarName'].str.split().str[0]
        return df

    def _create_new_features(self, df):
        df['power_to_weight'] = df['horsepower'] / df['curbweight']
        df['mpg_avg'] = (df['citympg'] + df['highwaympg']) / 2
        df['size_ratio'] = df['carwidth'] / df['carlength']
        return df

    def _handle_rare_brands(self, df):
        other = df['brand'].value_counts() < 5
        rare_brands = other[other == True].index
        df['brand'] = df['brand'].replace(rare_brands, 'other')
        return df

    def _encode_categorical(self, X_train, X_test, df_processed):
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        distribution = df_processed[categorical_columns].nunique() < 4

        for_label = distribution[distribution == True].index.tolist()
        for_onehot = distribution[distribution == False].index.tolist()

        # Label Encoding
        for column in for_label:
            le = LabelEncoder()
            X_train[column] = le.fit_transform(X_train[column])
            X_test[column] = le.transform(X_test[column])
            self.label_encoders[column] = le

        # One-Hot Encoding
        for column in for_onehot:
            ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

            # Train
            onehot_encoded = ohe.fit_transform(X_train[[column]])
            feature_names = ohe.get_feature_names_out([column])
            onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=X_train.index)
            X_train = pd.concat([X_train, onehot_df], axis=1)
            X_train.drop(column, axis=1, inplace=True)

            # Test
            onehot_encoded = ohe.transform(X_test[[column]])
            onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=X_test.index)
            X_test = pd.concat([X_test, onehot_df], axis=1)
            X_test.drop(column, axis=1, inplace=True)

            self.onehot_encoders[column] = ohe


        return X_train, X_test

    def _feature_scaling(self, X_train, X_test, df_processed, target_column):
        scaler = StandardScaler()
        orig_num_col = df_processed.select_dtypes(include=[np.number]).columns.drop(target_column)
        X_train[orig_num_col] = scaler.fit_transform(X_train[orig_num_col])
        X_test[orig_num_col] = scaler.transform(X_test[orig_num_col])
        joblib.dump(scaler, '../models/scaler.pkl')
        return X_train, X_test

    def fit_transform(self, df, target_column):
        df_processed = df.copy()

        # 1. Извлечение бренда
        df_processed = self._extract_brand(df_processed)

        # 2. Создание новой целевой колонки для классификации и удаление price
        if target_column == 'is_premium':
            df_processed = create_premium_target(df_processed)
            df_processed = df_processed.drop('price', axis=1, errors='ignore')

        # 3. Создание новых признаков
        df_processed = self._create_new_features(df_processed)

        # 4. Удаление ненужных столбцов
        df_processed = df_processed.drop(self.columns_to_drop, axis=1, errors='ignore')

        # 5. Объединение брендов в other
        df_processed = self._handle_rare_brands(df_processed)

        # 6. Разделение на X, y
        X = df_processed.drop(target_column, axis=1)
        y = df_processed[target_column]

        # 7. Разделение на X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df_processed['brand']
        )
        # 8. Кодирование категориальных признаков
        X_train_encoded, X_test_encoded = self._encode_categorical(X_train, X_test, df_processed)

        # 9. Масштабирование числовых признаков
        X_train_final, X_test_final = self._feature_scaling(X_train_encoded, X_test_encoded, df_processed, target_column)

        joblib.dump(self.label_encoders, '../models/label_encoders.pkl')
        joblib.dump(self.onehot_encoders, '../models/onehot_encoders.pkl')

        return X_train_final, X_test_final, y_train, y_test


def preprocess_data(df, target_column):
    preprocessor = CarPricePreprocessor()
    return preprocessor.fit_transform(df, target_column)