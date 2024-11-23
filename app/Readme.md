```markdown
# 📊 Prédiction de la Demande Énergétique en Afrique

Cette application Streamlit permet d'analyser et de prédire la demande énergétique projetée dans différentes régions d'Afrique à partir de données démographiques, économiques et climatiques.

---

## 🚀 Fonctionnalités

### 1. **Aperçu des Données**
- Affichage des dimensions du dataset.
- Identification des colonnes avec des valeurs manquantes.
- Visualisation des premières lignes et statistiques descriptives.

### 2. **Analyse Exploratoire**
- Histogrammes pour les variables numériques.
- Matrice de corrélation avec des options de palettes de couleurs.
- Boxplots pour visualiser la distribution des variables.

### 3. **Prédiction de la Demande Énergétique**
- Formulaire interactif pour saisir les caractéristiques de la région.
- Modèle Lasso pour la prédiction.
- Affichage de la demande énergétique projetée en Mega Watt.
- Validation croisée pour évaluer les performances du modèle.

---

## 📂 Structure du Projet

```plaintext
📦 project-directory
├── 📜 README.md           # Ce fichier
├── 📜 app.py              # Code principal de l'application Streamlit
├── 📜 requirements.txt    # Dépendances Python nécessaires
├── 📂 data
│   └── train.csv          # Dataset utilisé pour l'analyse et l'entraînement
```

---

## 🛠️ Prérequis

- Python 3.7 ou plus récent
- Un gestionnaire de packages comme `pip` ou `conda`

---

## ⚙️ Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/doloadama/Predictions_demande_energie.git
   cd nom-du-repo
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Lancez l'application :
   ```bash
   streamlit run app.py
   ```

---

## 📈 Modèle de Prédiction

- **Algorithme utilisé** : Lasso Regression
- **Pipeline de prétraitement** :
  - Standardisation (`StandardScaler`)
  - Normalisation (`MinMaxScaler`)

---

## 📊 Visualisation

L'application génère plusieurs visualisations utiles pour l'analyse exploratoire, comme :
- Histogrammes
- Matrice de corrélation
- Boxplots

---

## 📋 Données

Les données utilisées proviennent de [ce dataset](https://github.com/dataafriquehub/energy_data) et contiennent :
- Caractéristiques démographiques (population, taux d'accès à l'énergie, etc.)
- Variables climatiques (ensoleillement)
- Indicateurs politiques et économiques

---

## 🧑‍💻 Contributeurs

- [Votre Nom](https://github.com/votre-utilisateur)

---

## 📜 Licence

Ce projet est entierement libre d'utilisation

---

## 🌟 Remerciements

Merci à [Data Afrique Hub](https://github.com/dataafriquehub) pour les données fournies.

---

## 📝 Notes

- Si vous rencontrez des problèmes, n'hésitez pas à ouvrir une [issue](https://github.com/votre-utilisateur/nom-du-repo/issues).
- Cette application est en développement actif et peut être améliorée.
```
