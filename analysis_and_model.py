import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve, precision_recall_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        try:
            data = pd.read_csv("data/ai4i2020.csv")
        except:
            dataset = fetch_ucirepo(id=601)
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    
    # Предобработка данных
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    
    # Масштабирование
    scaler = StandardScaler()
    num_cols = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    # Разделение данных
    X = data.drop('Machine failure', axis=1)
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Обучение моделей
    st.header("Обучение и сравнение моделей")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = []
    fig_roc = plt.figure(figsize=(10, 8))
    fig_pr = plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Расчет метрик
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc
        })
        
        # ROC-кривая
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(fig_roc.number)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        
        # Precision-Recall кривая
        plt.figure(fig_pr.number)
        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.3f})")
    
    # Визуализация ROC-кривых
    plt.figure(fig_roc.number)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые')
    plt.legend(loc='lower right')
    st.pyplot(fig_roc)
    
    # Визуализация Precision-Recall кривых
    plt.figure(fig_pr.number)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall кривые')
    plt.legend(loc='upper right')
    st.pyplot(fig_pr)
    
    # Таблица сравнения моделей
    results_df = pd.DataFrame(results)
    st.subheader("Сравнение моделей")
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.3f}",
        "ROC AUC": "{:.3f}",
        "PR AUC": "{:.3f}"
    }).highlight_max(color='lightgreen').highlight_min(color='#ffcccc'), use_container_width=True)
    
    # Выбор лучшей модели
    best_model_name = results_df.loc[results_df['ROC AUC'].idxmax(), 'Model']
    best_model = models[best_model_name]
    st.success(f"Лучшая модель: {best_model_name} (ROC AUC = {results_df.loc[results_df['ROC AUC'].idxmax(), 'ROC AUC']:.3f})")
    
    # Детализация для лучшей модели
    st.subheader(f"Детальный анализ: {best_model_name}")
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Предсказанные')
    ax.set_ylabel('Фактические')
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    
    st.text("Classification Report:\n" + classification_report(y_test, y_pred_best))
    
    # Интерфейс предсказаний
    with st.expander("Прогнозирование на новых данных"):
        with st.form("prediction_form"):
            st.write("Введите параметры оборудования:")
            col1, col2 = st.columns(2)
            
            with col1:
                type_val = st.selectbox("Тип продукта", ["L", "M", "H"])
                air_temp = st.number_input("Температура воздуха [K]", value=300.0)
                process_temp = st.number_input("Рабочая температура [K]", value=310.0)
                
            with col2:
                rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
                torque = st.number_input("Крутящий момент [Nm]", value=40.0)
                tool_wear = st.number_input("Износ инструмента [min]", value=100)
            
            if st.form_submit_button("Предсказать"):
                input_data = pd.DataFrame({
                    'Type': [{"L":0, "M":1, "H":2}[type_val]],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                })
                
                # Масштабирование
                input_data[num_cols] = scaler.transform(input_data[num_cols])
                
                # Предсказание
                prediction = best_model.predict(input_data)[0]
                probability = best_model.predict_proba(input_data)[0][1]
                
                st.metric("Прогноз", "Отказ оборудования ⚠️" if prediction == 1 else "Нормальная работа ✅")
                st.progress(probability)
                st.write(f"Вероятность отказа: {probability:.2%}")
                
                # Интерпретация
                if prediction == 1:
                    st.warning("Рекомендация: Требуется профилактическое обслуживание")
                    failure_reasons = []
                    if tool_wear > 200:
                        failure_reasons.append("Высокий износ инструмента")
                    if abs(air_temp - process_temp) < 8.6 and rotational_speed < 1380:
                        failure_reasons.append("Проблемы с теплоотводом")
                    if torque * rotational_speed < 3500 or torque * rotational_speed > 9000:
                        failure_reasons.append("Неправильная мощность")
                    
                    if failure_reasons:
                        st.write("Возможные причины:")
                        for reason in failure_reasons:
                            st.write(f"- {reason}")