import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de la Demande Énergétique en Afrique",
    layout="wide"
)

# Titre et description
st.title("📊 Prédiction de la Demande Énergétique en Afrique")
st.markdown("""
Cette application permet d'analyser et de prédire la demande énergétique projetée dans différentes régions d'Afrique.
""")

# Fonction pour charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/dataafriquehub/energy_data/refs/heads/main/train.csv')
    return df

# Fonction pour le prétraitement des données
def preprocess_dataframe(df):
    # Remplir les valeurs manquantes
    df['taux_adoption_energies_renouvelables'] = df.groupby('country')[
        'taux_adoption_energies_renouvelables'].transform(lambda x: x.fillna(x.mean() * 0))

    # Supprimer les colonnes inutiles
    columns_to_drop = ['types_sols', 'habit_de_mariage', 'nombre_animaux_domestiques', 'lat', 'lon', 'country']
    if 'demande_energetique_projectee' in df.columns:
        columns_to_drop.append('demande_energetique_projectee')
    df_new = df.drop(columns=columns_to_drop, errors='ignore')

    return df_new

# Fonction pour garantir la cohérence des colonnes pour la prédiction
def ensure_feature_consistency(df_input, columns_train):
    for col in columns_train:
        if col not in df_input.columns:
            df_input[col] = 0  # Ajouter une colonne manquante avec une valeur par défaut
    return df_input[columns_train]

# Charger les données
try:
    df = load_data()

    # Barre latérale
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choisissez une page",
        ["Aperçu des données", "Analyse exploratoire", "Prédiction"]
    )

    if page == "Aperçu des données":
        st.header("📋 Aperçu des données")
        st.dataframe(df.head())
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe())

    elif page == "Analyse exploratoire":
        st.header("📈 Analyse exploratoire")

        # Distribution de la variable cible
        st.subheader("Distribution de la demande énergétique projetée")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='demande_energetique_projectee', kde=True)
        st.pyplot(fig)

        # Matrice de corrélation
        st.subheader("Matrice de corrélation")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        st.pyplot(fig)

        # Variables corrélées avec la cible
        st.subheader("Top 5 des variables les plus corrélées avec la demande énergétique projetée")
        target_correlations = correlation_matrix['demande_energetique_projectee'].sort_values(ascending=False)
        st.dataframe(target_correlations.head())

        # Boxplot avec un filtre pour taux_adoption_energies_renouvelables
        st.subheader("Boxplot de la demande énergétique en fonction du taux d'adoption des énergies renouvelables")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='taux_adoption_energies_renouvelables', y='demande_energetique_projectee')
        st.pyplot(fig)

    elif page == "Prédiction":
        st.header("🎯 Prédiction de la demande énergétique")

        # Formulaire d'entrée pour la prédiction
        st.subheader("Entrez les caractéristiques pour la prédiction")
        col1, col2, col3 = st.columns(3)

        with col1:
            population = st.number_input("Population", min_value=0, value=10000)
            taux_ensoleillement = st.number_input("Taux d'ensoleillement", min_value=0.0, value=5.0)
            demande_energetique_actuelle = st.number_input("Demande énergétique actuelle", min_value=0.0, value=1000.0)

        with col2:
            capacite_installee_actuelle = st.number_input("Capacité installée actuelle", min_value=0.0, value=500.0)
            duree_ensoleillement_annuel = st.number_input("Durée d'ensoleillement annuel", min_value=0, value=2000)
            cout_installation_solaire = st.number_input("Coût installation solaire", min_value=0.0, value=1000.0)

        with col3:
            taux_adoption_energies_renouvelables = st.number_input("Taux adoption énergies renouvelables",
                                                                   min_value=0.0, max_value=100.0, value=50.0)
            stabilite_politique = st.number_input("Stabilité politique", min_value=0.0, max_value=10.0, value=5.0)
            taux_acces_energie = st.number_input("Taux accès énergie", min_value=0.0, max_value=100.0, value=70.0)

        # Prédiction
        if st.button("Prédire la demande"):
            # Préparer les données d'entrée
            input_data = pd.DataFrame({
                'population': [population],
                'taux_ensoleillement': [taux_ensoleillement],
                'demande_energetique_actuelle': [demande_energetique_actuelle],
                'capacite_installee_actuelle': [capacite_installee_actuelle],
                'duree_ensoleillement_annuel': [duree_ensoleillement_annuel],
                'cout_installation_solaire': [cout_installation_solaire],
                'taux_adoption_energies_renouvelables': [taux_adoption_energies_renouvelables],
                'stabilite_politique': [stabilite_politique],
                'taux_acces_energie': [taux_acces_energie]
            })

            # Prétraitement des données
            X_train = preprocess_dataframe(df)
            columns_train = X_train.columns.tolist()
            y_train = df['demande_energetique_projectee']

            # Créer et entraîner le modèle
            model = make_pipeline(StandardScaler(), MinMaxScaler(), Lasso(alpha=0.001))
            model.fit(X_train, y_train)

            # Évaluer le modèle avec une validation croisée
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            mean_cv_score = np.mean(np.abs(cv_scores))

            # Préparer les données pour la prédiction
            input_data_prepared = ensure_feature_consistency(input_data, columns_train)

            # Prédire
            prediction = model.predict(input_data_prepared)

            # Afficher la prédiction
            st.success(f"Demande énergétique projetée : {prediction[0]:.2f}")

            # Afficher la précision du modèle
            st.write(f"Précision du modèle (Erreur quadratique moyenne, validation croisée) : {mean_cv_score:.2f}")

            # Visualisation de la prédiction
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='demande_energetique_projectee', kde=True)
            plt.axvline(prediction[0], color='red', linestyle='--', label='Prédiction')
            plt.legend()
            st.pyplot(fig)

except Exception as e:
    st.error(f"Une erreur s'est produite : {str(e)}")
