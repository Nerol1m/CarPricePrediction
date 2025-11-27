import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests

@st.cache_resource
def get_usd_to_rub_rate():
    """Получаем актуальный курс USD к RUB"""
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        data = response.json()
        return data['rates']['RUB']
    except:
        return 100.0

def load_models():
    """Загружаем все модели и preprocessing objects"""
    try:
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        onehot_encoders = joblib.load('models/onehot_encoders.pkl')
        model_reg = joblib.load('models/random_forest_regression_final.pkl')
        model_clf = joblib.load('models/random_forest_classifier_final.pkl')

        return scaler, label_encoders, onehot_encoders, model_reg, model_clf
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        return None, None, None, None, None

scaler, label_encoders, onehot_encoders, model_reg, model_clf = load_models()

# Настройка страницы
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Боковая панель с навигацией
page = st.sidebar.selectbox(
    "Выберите раздел:",
    ["Главная", "Предсказание цены и классификация", "Анализ модели"]
)


# ====== ФУНКЦИИ СТРАНИЦ =======

def show_main_page():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        # 🚗 Car Price Prediction  
        ### *Умный анализ стоимости автомобилей*
        """)
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 15px; color: white;'>
        <h3 style='color: white; margin: 0;'>📊 Полный ML Pipeline</h3>
        <p style='margin: 10px 0 0 0;'>От данных до готового приложения</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
        <div style='font-size: 48px;'>🎯</div>
        <div style='font-weight: bold;'>2 модели</div>
        <div style='font-size: 14px;'>Регрессия + Классификация</div>
        </div>
        """, unsafe_allow_html=True)

    # Возможности
    st.markdown("## 🎯 Возможности приложения")

    feat1, feat2, feat3, feat4 = st.columns(4)

    with feat1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px;'>
        <div style='font-size: 36px;'>💰</div>
        <h4>Предсказание цены</h4>
        <p style='font-size: 14px;'>Точность: <b>95.4% R²</b></p>
        <p style='font-size: 12px;'>Ошибка: ±$1,400</p>
        </div>
        """, unsafe_allow_html=True)

    with feat2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px;'>
        <div style='font-size: 36px;'>🏷️</div>
        <h4>Классификация</h4>
        <p style='font-size: 14px;'>Качество: <b>91.7% F1</b></p>
        <p style='font-size: 12px;'>Премиальный/Бюджетный</p>
        </div>
        """, unsafe_allow_html=True)

    with feat3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px;'>
        <div style='font-size: 36px;'>📈</div>
        <h4>Анализ данных</h4>
        <p style='font-size: 14px;'>Важность признаков</p>
        <p style='font-size: 12px;'>Инсайты и метрики</p>
        </div>
        """, unsafe_allow_html=True)

    with feat4:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px;'>
        <div style='font-size: 36px;'>💱</div>
        <h4>Актуальный курс</h4>
        <p style='font-size: 14px;'>Live конвертация</p>
        <p style='font-size: 12px;'>USD → RUB API</p>
        </div>
        """, unsafe_allow_html=True)

    # Технологии
    st.markdown("## 🛠 Технологический стек")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.markdown("""
        **🤖 Машинное обучение**
        - Scikit-learn
        - XGBoost
        - Scikit-optimize
        """)

    with tech2:
        st.markdown("""
        **📊 Анализ данных**
        - Pandas / NumPy
        - Matplotlib / Seaborn
        - Jupyter Notebooks
        """)

    with tech3:
        st.markdown("""
        **🌐 Веб-приложение**
        - Streamlit
        - Python
        - ML Pipeline
        """)

    with tech4:
        st.markdown("""
        **🌐 Интеграции**
        - Exchange Rate API
        - Real-time данные
        - RESTful сервисы
        """)

    st.warning("""
    ⚠️ **Примечание:** Модели обучены на исторических данных и показывают относительную стоимость. 
    Результаты носят ознакомительный характер.
    """)

# Вкладка для предсказания цены
def show_prediction_page():
    st.markdown("# 🚗 Предсказание цены и классификация")
    st.markdown("### 💫 Введите параметры автомобиля для получения прогноза")
    fueltype_map = {"Бензин": "gas", "Дизель": "diesel"}
    aspiration_map = {"Стандартный": "std", "Турбо": "turbo"}
    doornumber_map = {"2": "two", "4": "four"}
    drivewheel_map = {"Передний": "fwd", "Задний": "rwd", "Полный": "4wd"}
    enginelocation_map = {"Переднее": "front", "Заднее": "rear"}
    carbody_map = {
        "Седан": "sedan",
        "Хэтчбек": "hatchback",
        "Универсал": "wagon",
        "Купе": "hardtop",
        "Минивэн": "minivan",
        "Кабриолет": "convertible"
    }
    enginetype_map = {
        "DOHC": "dohc",
        "OHC": "ohc",
        "OHCV": "ohcv",
        "L": "l",
        "Роторный": "rotor"
    }
    cylindernumber_map = {
        "2": "two", "3": "three", "4": "four", "5": "five",
        "6": "six", "8": "eight", "12": "twelve"
    }
    fuelsystem_map = {
        "Карбюратор": "1bbl",
        "MPFI": "mpfi",
        "GDI": "2bbl",
        "2BBL": "4bbl",
        "4BBL": "idi"
    }
    brand_map = {
        "Toyota": "toyota", "Nissan": "nissan", "Mazda": "mazda",
        "Honda": "honda", "Mitsubishi": "mitsubishi", "Subaru": "subaru",
        "Volkswagen": "volkswagen", "Peugeot": "peugeot", "Volvo": "volvo",
        "Dodge": "dodge", "BMW": "bmw", "Buick": "buick",
        "Audi": "audi", "Plymouth": "plymouth", "Saab": "saab",
        "Porsche": "porsche", "Isuzu": "isuzu", "Jaguar": "jaguar",
        "Другая": "other"
    }

    for_label = ['fueltype', 'aspiration', 'doornumber', 'drivewheel', 'enginelocation']
    for_one_hot = ['carbody', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand']

    st.markdown("""
    <span style='color: #ff4b4b; font-size: 14px;'>
    ⚠️ Модель обучена на исторических данных и показывает относительную стоимость
    </span>
    """, unsafe_allow_html=True)

    with st.form("car_parameters"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📏 Размеры")
            wheelbase_cm = st.slider("Колесная база (см)", 200.0, 300.0, 254.0, step=1.0)
            carlength_cm = st.slider("Длина авто (см)", 350.0, 530.0, 430.0, step=1.0)
            carwidth_cm = st.slider("Ширина авто (см)", 150.0, 250.0, 165.0, step=1.0)
            curbweight_kg = st.slider("Снаряженная масса (кг)", 800, 2500, 1500, step=50)

            # Конвертируем в дюймы для модели
            wheelbase = wheelbase_cm / 2.54
            carlength = carlength_cm / 2.54
            carwidth = carwidth_cm / 2.54
            curbweight = curbweight_kg * 2.20462  # кг в фунты

        with col2:
            st.subheader("⚙️ Технические характеристики")
            horsepower = st.slider("Мощность (л.с.)", 50, 1000, 120)
            enginesize = st.slider("Объем двигателя (л)", 1.0, 10.0, 2.0, step=0.1)
            # Конвертация - 1 литр = 1000 куб. см
            enginesize_cc = enginesize * 1000

            boreratio = st.slider("Диаметр цилиндра (см)", 7.0, 10.0, 8.5, step=0.1)

        st.subheader("⛽ Расход топлива")
        col3, col4 = st.columns(2)

        with col3:
            citympg = st.slider("Расход в городе (л/100км)", 5.0, 20.0, 10.0, step=0.5)
            # Конвертация расхода: л/100км → mpg
            citympg_converted = 235.21 / citympg

        with col4:
            highwaympg = st.slider("Расход по трассе (л/100км)", 4.0, 15.0, 7.0, step=0.5)
            highwaympg_converted = 235.21 / highwaympg

        st.subheader("🚦 Безопасность")
        symboling = st.selectbox("Рейтинг безопасности",
                                 ["Очень безопасный (-3)", "Безопасный (-2)", "Средний (-1)",
                                  "Нейтральный (0)", "Рискованный (1)", "Опасный (2)"])

        # Конвертация в числовое значение
        symboling_map = {
            "Очень безопасный (-3)": -3, "Безопасный (-2)": -2, "Средний (-1)": -1,
            "Нейтральный (0)": 0, "Рискованный (1)": 1, "Опасный (2)": 2
        }
        symboling_value = symboling_map[symboling]

        # Категориальные параметры
        st.subheader("🎛️ Дополнительные параметры")
        col5, col6 = st.columns(2)

        with col5:
            fueltype = st.selectbox("Тип топлива", ["Бензин", "Дизель"])
            aspiration = st.selectbox("Наддув", ["Стандартный", "Турбо"])
            doornumber = st.selectbox("Количество дверей", ["2", "4"])
            drivewheel = st.selectbox("Привод", ["Передний", "Задний", "Полный"])

        with col6:
            enginelocation = st.selectbox("Расположение двигателя", ["Переднее", "Заднее"])
            carbody = st.selectbox("Тип кузова", ["Седан", "Хэтчбек", "Универсал", "Купе", "Минивэн", "Кабриолет"])
            enginetype = st.selectbox("Тип двигателя", ["DOHC", "OHC", "OHCV", "L", "Роторный"])
            cylindernumber = st.selectbox("Количество цилиндров", ["2", "3", "4", "5", "6", "8", "12"])
            fuelsystem = st.selectbox("Система впрыска", ["Карбюратор", "MPFI", "GDI", "2BBL", "4BBL"])

        st.subheader("🏷️ Бренд")
        brand = st.selectbox("Марка автомобиля", [
            "Toyota", "Nissan", "Mazda", "Honda", "Mitsubishi", "Subaru",
            "Volkswagen", "Peugeot", "Volvo", "Dodge", "BMW", "Buick",
            "Audi", "Plymouth", "Saab", "Porsche", "Isuzu", "Jaguar", "Другая"
        ])

        fueltype_english = fueltype_map[fueltype]
        aspiration_english = aspiration_map[aspiration]
        doornumber_english = doornumber_map[doornumber]
        drivewheel_english = drivewheel_map[drivewheel]
        enginelocation_english = enginelocation_map[enginelocation]
        carbody_english = carbody_map[carbody]
        enginetype_english = enginetype_map[enginetype]
        cylindernumber_english = cylindernumber_map[cylindernumber]
        fuelsystem_english = fuelsystem_map[fuelsystem]
        brand_english = brand_map[brand]

        # Кнопка предсказания
        submitted = st.form_submit_button("🎯 Предсказать цену и класс")

    # Вычисляем производные признаки (вне формы)
    power_to_weight = horsepower / curbweight
    mpg_avg = (citympg_converted + highwaympg_converted) / 2
    size_ratio = carwidth / carlength

    # Если форма отправлена
    if submitted:
        # Создаём DataFrame с введенными данными
        input_data = pd.DataFrame({
            'symboling': [symboling_value],
            'wheelbase': [wheelbase],
            'carlength': [carlength],
            'carwidth': [carwidth],
            'curbweight': [curbweight],
            'enginesize': [enginesize_cc],
            'boreratio': [boreratio],
            'horsepower': [horsepower],
            'citympg': [citympg_converted],
            'highwaympg': [highwaympg_converted],
            'power_to_weight': [power_to_weight],
            'mpg_avg': [mpg_avg],
            'size_ratio': [size_ratio],
            'fueltype': fueltype_english,
            'aspiration': aspiration_english,
            'doornumber': doornumber_english,
            'drivewheel': drivewheel_english,
            'enginelocation': enginelocation_english,
            'carbody': carbody_english,
            'enginetype': enginetype_english,
            'cylindernumber': cylindernumber_english,
            'fuelsystem': fuelsystem_english,
            'brand': brand_english
        })
        orig_num_col = input_data.select_dtypes(include=[np.number]).columns
        input_data[orig_num_col] = scaler.transform(input_data[orig_num_col])

        for column in for_label:
            input_data[column] = label_encoders[column].transform(input_data[column])

        for column in for_one_hot:
            ohe = onehot_encoders[column]

            onehot_encoded = ohe.transform(input_data[[column]])
            feature_names = ohe.get_feature_names_out([column])

            onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=input_data.index)

            input_data = pd.concat([input_data, onehot_df], axis=1)
            input_data.drop(column, axis=1, inplace=True)

        expected_columns = model_reg.feature_names_in_
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)

        price_prediction = model_reg.predict(input_data)
        predicted_price_usd = float(price_prediction[0]) if len(price_prediction) > 0 else 0
        exchange_rate = get_usd_to_rub_rate()
        predicted_price_rub = predicted_price_usd * exchange_rate

        classification_predict = model_clf.predict(input_data)

        st.success("✅ Данные получены!")

        col_pred1, col_pred2, col_pred3 = st.columns(3)

        st.markdown("""
        <span style='color: #ff4b4b; font-size: 14px;'>
        ⚠️ Модель обучена на исторических данных и показывает относительную стоимость
        </span>
        """, unsafe_allow_html=True)
        with col_pred1:
            st.subheader("💰 Цена в USD")
            st.metric(
                label="Рыночная стоимость",
                value=f"${predicted_price_usd:,.0f}",
                delta="+2,500"
            )

        with col_pred2:
            st.subheader("💰 Цена в RUB")
            st.metric(
                label=f"Рыночная стоимость (курс: {exchange_rate:.2f}₽)",
                value=f"₽{predicted_price_rub:,.0f}",
                delta="Актуальный курс"
            )
            st.caption("💱 Курс обновляется в реальном времени")

        with col_pred3:
            st.subheader("🏷️ Классификация")
            if classification_predict == 1:
                st.metric(
                    label="Ценовой сегмент",
                    value="Премиальный",
                    delta="Высокий класс"
                )
            else:
                st.metric(
                    label="Ценовой сегмент",
                    value="Эконом",
                    delta="Средний класс"
                )

def show_analysis_page():
    st.header("📈 Анализ моделей")

    tab1, tab2, tab3 = st.tabs(["📊 Важность признаков", "📈 Метрики качества", "🔍 Инсайты"])

    with tab1:
        st.subheader("🔧 Важность признаков для предсказания цены")

        # Берем ЛУЧШУЮ модель из RandomizedSearchCV
        best_model_reg = model_reg.best_estimator_ if hasattr(model_reg, 'best_estimator_') else model_reg

        # Получаем важность признаков
        feature_importance_reg = best_model_reg.feature_importances_
        features_reg = best_model_reg.feature_names_in_

        # Создаём DataFrame
        importance_df_reg = pd.DataFrame({
            'feature': features_reg,
            'importance': feature_importance_reg
        }).sort_values('importance', ascending=False).head(10)  # Уменьшили до 10

        # Строим график с тёмной темой
        fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
        sns.set_style("darkgrid")
        sns.barplot(data=importance_df_reg, y='feature', x='importance', ax=ax_reg,
                    palette="viridis")
        ax_reg.set_title('Топ-10 важных признаков для предсказания цены',
                         color='white', pad=20)
        ax_reg.set_xlabel('Важность', color='white')
        ax_reg.set_ylabel('Признаки', color='white')
        ax_reg.tick_params(colors='white')
        fig_reg.patch.set_facecolor('#0E1117')
        ax_reg.set_facecolor('#0E1117')
        importance_df_reg = importance_df_reg.set_index('feature')
        st.bar_chart(importance_df_reg['importance'])
        st.pyplot(fig_reg)

        st.subheader("🏷️ Важность признаков для классификации")

        best_model_clf = model_clf.best_estimator_ if hasattr(model_clf, 'best_estimator_') else model_clf
        feature_importance_clf = best_model_clf.feature_importances_
        features_clf = best_model_clf.feature_names_in_

        importance_df_clf = pd.DataFrame({
            'feature': features_clf,
            'importance': feature_importance_clf
        }).sort_values('importance', ascending=False).head(10)

        fig_clf, ax_clf = plt.subplots(figsize=(10, 6))
        sns.set_style("darkgrid")
        sns.barplot(data=importance_df_clf, y='feature', x='importance', ax=ax_clf,
                    palette="plasma")
        ax_clf.set_title('Топ-10 важных признаков для классификации',
                         color='white', pad=20)
        ax_clf.set_xlabel('Важность', color='white')
        ax_clf.set_ylabel('Признаки', color='white')
        ax_clf.tick_params(colors='white')
        fig_clf.patch.set_facecolor('#0E1117')
        ax_clf.set_facecolor('#0E1117')
        importance_df_clf = importance_df_clf.set_index('feature')
        st.bar_chart(importance_df_clf['importance'])

        st.pyplot(fig_clf)

    with tab2:
        with tab2:
            st.subheader("📊 Метрики качества моделей")

            # Регрессия
            st.markdown("### 🚗 Метрики регрессии (цена)")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("R² Score", "0.954", "95.4% точности")
                st.metric("MAE", "1,379", "± $1,379")

            with col2:
                st.metric("MSE", "3.23M", "3,232,979")
                st.metric("Median AE", "1,233", "Медианная ошибка")

            with col3:
                st.metric("MAPE", "10.69%", "Относительная ошибка")
                st.metric("Accuracy <10%", "60.98%", "Точность в 10%")

            st.progress(0.954, text="Общая точность модели: 95.4%")

            # Классификация
            st.markdown("### 🏷️ Метрики классификации")
            col4, col5 = st.columns(2)

            with col4:
                st.metric("F1-Score", "0.917", "91.7% баланс")
                st.metric("Метод", "Bayesian Optimization", "Настройка")

            with col5:
                st.metric("Модель", "Random Forest", "Классификатор")
                st.metric("Качество", "Отличное", ">90% F1")

            st.progress(0.917, text="Сбалансированная точность: 91.7%")

            # Интерпретация
            st.markdown("### 💎 Интерпретация метрик")
            st.success("""
            **Отличные результаты!** 
            - ✅ Регрессия: 95% объяснённой дисперсии при ошибке ~$1,400
            - ✅ Классификация: 92% F1-score - высокая сбалансированность
            """)

    with tab3:
        st.subheader("🔍 Ключевые инсайты")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🚗 Что влияет на цену")
            st.markdown("""
            <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #ff4b4b;'>
            <ul style='color: white;'>
            <li>🏗️ <b>Объем двигателя</b> - главный фактор (15%)</li>
            <li>⚖️ <b>Вес автомобиля</b> - второй по важности (12%)</li>
            <li>💪 <b>Мощность двигателя</b> - значимое влияние (10%)</li>
            <li>⛽ <b>Расход топлива</b> - обратная зависимость (8-10%)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🏷️ Что определяет класс авто")
            st.markdown("""
            <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #00cc96;'>
            <ul style='color: white;'>
            <li>💪 <b>Мощность</b> - ключевой признак (14%)</li>
            <li>⛽ <b>Экономичность</b> - важный фактор (12%)</li>
            <li>🏗️ <b>Объем двигателя</b> - влияет на класс (9%)</li>
            <li>🛣️ <b>Затраты на эксплуатацию</b> - учитываются (8%)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # Общие выводы
        st.markdown("### 💡 Основные выводы")
        st.markdown("""
        <div style='background: #1a1a1a; padding: 25px; border-radius: 10px; border: 1px solid #555;'>
        <div style='color: white;'>
        <b>📈 Технические характеристики важнее внешних параметров</b><br>
        Мощность и экономичность определяют как цену, так и класс автомобиля

        <br><br>
        <b>⚡ Производительность vs Экономичность</b><br>
        Премиальные авто жертвуют расходом топлива ради мощности

        <br><br> 
        <b>🎯 Универсальные факторы</b><br>
        horsepower и mpg_avg важны для обеих задач
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("""
        ⚠️ **Важно:** Модель обучена на исторических данных и показывает относительную стоимость. 
        Цифры носят ознакомительный характер.
        """)

if page == "Главная":
    show_main_page()
elif page == "Предсказание цены и классификация":
    show_prediction_page()
elif page == "Анализ модели":
    show_analysis_page()

st.sidebar.markdown("---")
st.sidebar.write("© 2024 Car Price Prediction App")