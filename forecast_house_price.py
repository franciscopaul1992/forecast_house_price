# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 18:00:00 2025

@author: BRITT
"""

import streamlit as st
import joblib 
import pandas as pd
import numpy as np
import xgboost

# Cargar modelo, scaler y dataset
model = joblib.load('mejor_modelo_xgboost.pkl')
scaler = joblib.load('scaler.pkl')
data_futura = pd.read_csv("BostonHoustingFutura.csv")

# Asegurar que los nombres de las columnas estén en minúsculas (si es necesario)
data_futura.columns = data_futura.columns.str.lower()

# Menú de navegación
st.sidebar.title("Estas Son Tus Opciones:")
opcion = st.sidebar.selectbox(
    "Escoge Una:",
    ("Inicio", "Predicción", "Acerca de")
)

# Página de Inicio
if opcion == "Inicio":
    st.title("Bienvenido al Forecast de Precios de Casas")
    st.write("""Esta aplicación utiliza un modelo de ML para predecir los precios de casas basado en las características ingresadas.""")
    st.image("https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=800", caption="Casa residencial", use_container_width=True)

# Página de Predicción
elif opcion == "Predicción":
    st.title("Forecast de Precios de Casas")
    st.write("Ingresa las características de la casa para predecir su precio.")

    # Botón para seleccionar fila aleatoria
    if 'random_row' not in st.session_state:
        st.session_state.random_row = None
        st.session_state.random_index = None

    if st.button("Seleccionar registro aleatorio 🎲"):
        sampled_df = data_futura.sample(1)
        st.session_state.random_row = sampled_df.iloc[0]
        st.session_state.random_index = sampled_df.index[0]
        st.rerun()

    # Función para obtener valores (media o de la fila seleccionada)
    def get_value(column):
        if st.session_state.random_row is not None:
            return st.session_state.random_row[column]
        else:
            return data_futura[column].mean()

    # Campos de entrada con valores del dataset o fila aleatoria
    crim = st.number_input(
        "Tasa de criminalidad (crim):", 
        min_value=float(data_futura['crim'].min()), 
        max_value=float(data_futura['crim'].max()), 
        value=float(get_value('crim'))
    )
    zn = st.number_input(
        "Proporción de terreno residencial (zn):", 
        min_value=float(data_futura['zn'].min()), 
        max_value=float(data_futura['zn'].max()), 
        value=float(get_value('zn'))
    )
    indus = st.number_input(
        "Negocios mayoristas (indus):", 
        min_value=float(data_futura['indus'].min()), 
        max_value=float(data_futura['indus'].max()), 
        value=float(get_value('indus'))
    )
    chas = st.number_input(
        "Proximidad al río (chas):", 
        min_value=int(data_futura['chas'].min()), 
        max_value=int(data_futura['chas'].max()), 
        value=int(get_value('chas')),
        step=1
    )
    nox = st.number_input(
        "Contaminación del aire (nox):", 
        min_value=float(data_futura['nox'].min()), 
        max_value=float(data_futura['nox'].max()), 
        value=float(get_value('nox'))
    )
    rm = st.number_input(
        "Número promedio de habitaciones (rm):", 
        min_value=float(data_futura['rm'].min()), 
        max_value=float(data_futura['rm'].max()), 
        value=float(get_value('rm'))
    )
    age = st.number_input(
        "Antigüedad de la vivienda (age):", 
        min_value=float(data_futura['age'].min()), 
        max_value=float(data_futura['age'].max()), 
        value=float(get_value('age'))
    )
    dis = st.number_input(
        "Accesibilidad zona de empleo (dis):", 
        min_value=float(data_futura['dis'].min()), 
        max_value=float(data_futura['dis'].max()), 
        value=float(get_value('dis'))
    )
    rad = st.number_input(
        "Acceso a autopistas (rad):", 
        min_value=float(data_futura['rad'].min()), 
        max_value=float(data_futura['rad'].max()), 
        value=float(get_value('rad'))
    )
    tax = st.number_input(
        "Impuestos (tax):", 
        min_value=float(data_futura['tax'].min()), 
        max_value=float(data_futura['tax'].max()), 
        value=float(get_value('tax'))
    )
    ptratio = st.number_input(
        "Relación alumno/profesor (ptratio):", 
        min_value=float(data_futura['ptratio'].min()), 
        max_value=float(data_futura['ptratio'].max()), 
        value=float(get_value('ptratio'))
    )
    b = st.number_input(
        "Medida demográfica (b):", 
        min_value=float(data_futura['b'].min()), 
        max_value=float(data_futura['b'].max()), 
        value=float(get_value('b'))
    )
    lstat = st.number_input(
        "% de personas con bajo nivel socioeconómico (lstat):", 
        min_value=float(data_futura['lstat'].min()), 
        max_value=float(data_futura['lstat'].max()), 
        value=float(get_value('lstat'))
    )
    if st.button("Predecir"):
        inputs = np.array([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]).reshape(1, -1)
        new_data_scaled = scaler.transform(inputs)
        prediccion = model.predict(new_data_scaled)
        
        if st.session_state.random_row is not None:
            valor_real = st.session_state.random_row['medv']
            prediccion_valor = prediccion[0]
            
            # Cálculo de métricas
            mape = (abs(valor_real - prediccion_valor) / valor_real) * 100
            diferencia = prediccion_valor - valor_real
            estimacion = "Sobreestimado" if diferencia > 0 else "Subestimado"
            
            # Crear DataFrame de resultados
            resultados_df = pd.DataFrame({
                "Registro": [st.session_state.random_index],
                "Precio estimado casa (medv)": [f"${prediccion_valor:,.2f}"],
                "Precio real casa (medv)": [f"${valor_real:,.2f}"],
                "MAPE (%)": [f"{mape:.2f}%"],
                "Diferencia": [f"${diferencia:+,.2f}"],
                "Estimación": [estimacion]
            })
            
            # Mostrar tabla con resultados
            st.dataframe(
                resultados_df,
                column_config={
                    "Registro": "Registro",
                    "Precio estimado casa (medv)": st.column_config.NumberColumn(format="$%.2f"),
                    "Precio real casa (medv)": st.column_config.NumberColumn(format="$%.2f"),
                    "MAPE (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Diferencia": st.column_config.NumberColumn(format="$%+.2f")
                },
                hide_index=True
            )
            
        else:
            st.success(f"**Precio estimado:** ${prediccion[0]:,.2f}")

# Página Acerca de (sin cambios)
elif opcion == "Acerca de":
    st.title("Acerca de")
    st.write("Esta aplicación fue creada para predecir el precio de casas utilizando un modelo de Machine Learning.")
    st.write("**Tecnologías utilizadas:**\n- Streamlit\n- Joblib \n- XGBoost \n- Machine Learning")
    st.write("**Desarrollado por:**\n- Carolina Gamarra\n- Britany ....\n- Paul Morales")
    st.markdown("[Visita mi perfil de GitHub](https://github.com/franciscopaul1992/forecast_house_price)")