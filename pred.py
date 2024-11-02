import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Charger les données à partir du fichier CSV
@st.cache_data
def load_data():
    data = pd.read_csv('air_q.csv')
    data['date'] = pd.to_datetime(data['date'])  # Convertir la colonne date en format datetime
    return data

data = load_data()

st.write('''
# Application de Prédiction de la Qualité de l'Air
''')

st.sidebar.header("Les paramètres d'entrée")

def user_input():
    ville_selectionnee = st.sidebar.selectbox('Sélectionnez la ville', data['Ville'].unique())
    date_selectionnee = st.sidebar.date_input('Sélectionnez la date', value=pd.to_datetime(data['date'].max()) + pd.DateOffset(days=1))

    donnees_filtrees = data[(data['Ville'] == ville_selectionnee) & (data['date'] == pd.to_datetime(date_selectionnee))]

    if not donnees_filtrees.empty:
        pm25 = donnees_filtrees['pm25'].values[0]
        pm10 = donnees_filtrees['pm10'].values[0]
        o3 = donnees_filtrees['o3'].values[0]
        no2 = donnees_filtrees['no2'].values[0]
        so2 = donnees_filtrees['so2'].values[0]
        co = donnees_filtrees['co'].values[0]
        aqi = donnees_filtrees['AQI'].values[0]
    else:
        pm25 = pm10 = o3 = no2 = so2 = co = aqi = 0  # Valeurs par défaut si aucune donnée n'est trouvée

    data_dict = {
        'pm25': [pm25],
        'pm10': [pm10],
        'o3': [o3],
        'no2': [no2],
        'so2': [so2],
        'co': [co]
    }

    parametres_air = pd.DataFrame(data_dict).round(2)
    return parametres_air, ville_selectionnee, date_selectionnee, aqi

def predictAirQuality(aqi):
    if aqi <= 50:
        return 'Bon'
    elif 51 < aqi <= 150:
        return 'Modéré'
    elif 151 < aqi <= 200:
        return 'Nocif'
    else:
        return 'Très Nocif'

# Prédiction ARIMA
def arima_forecast(data, ville, steps=1):
    ville_data = data[data['Ville'] == ville]
    ville_data = ville_data.set_index('date')
    
    # Modèle ARIMA sur AQI
    model = ARIMA(ville_data['AQI'], order=(5, 1, 0))
    model_fit = model.fit()
    
    # Prédire les valeurs futures
    forecast = model_fit.forecast(steps=steps)
    return forecast

df, ville, date, aqi = user_input()

st.subheader(f'Qualité de l\'air à {ville} pour le {date}')
st.write(df.style.format(precision=2).set_table_styles(
    [{'selector': 'th', 'props': [('border', '1px solid black'), ('padding', '8px')]},
     {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '8px')]}]
))

# Prédiction de la qualité de l'air
if not df.empty:
    imax = df.max(axis=1).values[0]
else:
    imax = 0

# Vérifier si la date sélectionnée est supérieure à la date maximale dans le dataframe
if pd.to_datetime(date) > data['date'].max():
    # Faire la prédiction de l'AQI et de sa catégorie
    days_ahead = (pd.to_datetime(date) - data['date'].max()).days
    forecast = arima_forecast(data, ville, steps=days_ahead)
    predicted_aqi = forecast.iloc[-1]
    prediction = predictAirQuality(predicted_aqi)
    st.subheader(f"Prédiction de l'AQI à la date sélectionnée : {predicted_aqi:.2f} - Catégorie d'air prédite : {prediction}")
else:
    # Afficher l'AQI et sa catégorie dans le dataframe pour la date sélectionnée
    prediction = predictAirQuality(aqi)
    st.subheader(f"AQI à la date sélectionnée : {aqi} - Catégorie d'air actuelle : {prediction}")

