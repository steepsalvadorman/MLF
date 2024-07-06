import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo y el vectorizador
model = joblib.load('models/logistic_regression_model.joblib')
vectorizer = joblib.load('models/count_vectorizer.joblib')

# Título de la aplicación
st.title("Predicción del Nivel CEFR de Textos en Inglés")

# Instrucciones
st.write("""
    Ingrese un texto en inglés para predecir su nivel CEFR (A1, A2, B1, B2 y C1).
""")

# Entrada de texto
input_text = st.text_area("Escriba el texto aquí:")

# Botón para realizar la predicción
if st.button("Predecir Nivel"):
    if input_text:
        # Transformar el texto de entrada
        input_text_vec = vectorizer.transform([input_text])
        
        # Realizar la predicción
        prediction = model.predict(input_text_vec)[0]
        probabilities = model.predict_proba(input_text_vec)[0]
        
        # Crear un DataFrame para mostrar los resultados
        prob_df = pd.DataFrame(probabilities, index=model.classes_, columns=["Probabilidad"])
        prob_df = prob_df.sort_values(by="Probabilidad", ascending=False)
        
        # Mostrar el resultado
        st.write(f"El nivel CEFR predicho es: **{prediction}**")
        st.write("Probabilidades por nivel CEFR:")
        st.dataframe(prob_df.style.format("{:.2%}"))
    else:
        st.write("Por favor, ingrese un texto.")
