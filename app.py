import pickle

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    model = load_model("gradient_boosting.sav")
    df = load_test_data("preprocessed_data.csv")

    y = df['price']
    X = df.drop(['price'], axis = 1)
    
    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Запрос к модели"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.write("Выберите страницу слева")

        st.header("Описание задачи")
        st.markdown("""
        Набор данных содержит информацию о домах, проданных в Мумбаи, Индия.\n
        Цель этой модели - предсказать цену дома на основе определенных параметров, доступных в наборе данных.
        """)

        st.header("Описание данных")
        st.markdown("""Предоставленные данные:\n
вещественные признаки:
* price - цена дома
* area - площадь дома
* latitude - географическая широта
* longitude - географическая долгота
* Bedrooms - количество спальных комнат
* Bathrooms - количество ванных комнат
* parking- количество парковочных мест
\n
бинарные признаки:
* Status_Ready to Move - показывает, достроен ли дом
* Status_Under Construction- показывает, находится ли дом в процессе стройки
* neworold_New Property - показывает, является ли мебелью дом новым
* neworold_Resale - показывает, что дом продает не застройщик
* Furnished_status_Furnished - показывает, что дом полностью обставлен мебелью
* Furnished_status_Semi-Furnished - показывает, что дом частично обставлен мебелью
* Furnished_status_Unfurnished - показывает, что дом не обставлен мебелью
* type_of_building_Flat - показывает, является ли жилое помещение квартирой
* type_of_building_Individual House - показывает, является ли жилое помещение домом""")

    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        st.write("Выберите страницу слева")
        request = st.selectbox(
            "Выберите запрос",
            ["Сделать прогноз", "Метрики", "Первые 20 предсказанных значений"]
        )

        if request == "Метрики":
            st.header("Метрики")
            y_pred = model.predict(X)
            cr=model.score(X, y)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            st.write("Score: ", cr)
            st.write("Mean Absolute Error: ", mae)
            st.write("Mean Squared Error: ", mse)
            
            #'Classification Report: ',cr
            #st.write(confusion_matrix(y, y_pred))
        elif request == "Первые 20 предсказанных значений":
            st.header("Первые 20 предсказанных значений")
            y_pred = model.predict(X.iloc[:20,:])
            for item in y_pred:
                st.write(f"{item:.2f}")
        elif request == "Сделать прогноз":
            st.header("Сделать прогноз")

            area = st.number_input("area", 0., 2000.)########################
            latitude = st.number_input("latitude", 0., 100.)
            longitude = st.number_input("longitude", 0., 100.)
            Bedrooms = st.number_input("Bedrooms", 0, 4)
            Bathrooms = st.number_input("Bathrooms", 0, 4)
            parking = st.number_input("parking", 0, 4)
            Status1 = st.number_input("Status_Ready to Move", 0, 1)
            Status0 = st.number_input("Status_Under Construction", 0, 1)
            neworold1 = st.number_input("neworold_New Property", 0, 1)
            neworold2 = st.number_input("neworold_Resale", 0, 1)
            fur1 = st.number_input("Furnished_status_Furnished", 0, 1)
            fur2 = st.number_input("Furnished_status_Semi-Furnished", 0, 1)
            fur3 = st.number_input("Furnished_status_Unfurnished", 0, 1)
            typ1 = st.number_input("type_of_building_Flat", 0, 1)
            typ2 = st.number_input("type_of_building_Individual House", 0, 1)
            
            

            if st.button('Предсказать'):
                data = [area,	latitude,	longitude,	Bedrooms, Bathrooms, parking, Status1, Status0, neworold1, neworold2, fur1, fur2, fur3, typ1, typ2]
                data = np.array(data).reshape((1, -1))
                pred = model.predict(data)

                st.write(pred[0])
            else:
                pass



@st.cache_data
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file, sep=";")
    return df


if __name__ == "__main__":
    main()
