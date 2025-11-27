import pandas as pd


def create_premium_target(df):
    """
    Создание интеллектуального бинарного таргета для классификации
    Определяет премиальные автомобили по комплексу характеристик
    """
    horsepower_threshold = df['horsepower'].quantile(0.7)
    enginesize_threshold = df['enginesize'].quantile(0.7)
    price_threshold = df['price'].quantile(0.7)

    premium_condition = (
            (df['horsepower'] >= horsepower_threshold) &
            (df['enginesize'] >= enginesize_threshold) &
            (df['price'] >= price_threshold)
    )

    premium_brands = ['bmw', 'jaguar', 'porsche', 'buick', 'audi', 'mercury']
    brand_condition = df['brand'].isin(premium_brands)

    df['is_premium'] = (premium_condition | brand_condition).astype(int)

    print(f"✅ Создан таргет 'is_premium'")
    print(f"📊 Распределение: {df['is_premium'].value_counts().to_dict()}")

    return df
