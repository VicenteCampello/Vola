# Prédiction des Pics de Volatilité sur les Marchés Financiers

## Objectif

Ce projet a pour ambition de détecter les pics de volatilité sur les marchés financiers (notamment ETH et BTC) à l'aide de techniques avancées de machine learning, et plus précisément à travers une architecture séquentielle hybride (Conv1D + BiLSTM).

L'idée est de transformer la compréhension de la volatilité en un outil prédictif utilisable en condition réelle via un pipeline automatisé.

## Contenu du projet

- Prétraitement avancé des données : log-return, GARCH, Hurst, rolling volatility
- Définition dynamique des spikes
- Entraînement d'un modèle LSTM pondéré
- Pipeline de prédiction en temps réel (données live via l’API Binance)
- Visualisations : matrices de confusion, courbes ROC, performances comparées
- Structuration complète avec sauvegarde du modèle et du scaler

## Arborescence du dépôt

.
├── data/
│   |── ETH-USD.csv
    └── BTC-USD.csv
├── figures/
│   ├── roc_first.png
│   ├── roc_last.png
│   ├── matrice_first.png
│   └── matrice_last.png
├── notebooks/
│   ├── analyse_btc.ipynb
│   ├── volatilite_avancee.ipynb
│   └── Live_predict.ipynb
├── model/
│   ├── lstm_volatility_model.keras
│   └── scaler.save
├── requirements.txt
└── README.md

## Lancer le projet

### Installation des dépendances

Crée un environnement virtuel puis :

```
pip install -r requirements.txt
```

### Entraînement du modèle

Ouvrir le notebook `notebooks/volatilite_avancee.ipynb`.

### Prédiction en live

Lancer le notebook `notebooks/Live_predict.ipynb`. Le modèle utilise les données récentes de l'API Binance et signale si un pic est probable.

## Résultats

- AUC : 0.85
- Recall spikes : 72%
- Modèle robuste aux déséquilibres de classe
- Détection en avance de phases de volatilité grâce au pattern learning

## Technologies utilisées

- Python, NumPy, Pandas
- TensorFlow / Keras
- Scikit-learn, imbalanced-learn (SMOTE)
- Arch (modèle GARCH)
- Binance API

## Références techniques

- Utilisation ponctuelle de ChatGPT pour la clarification d’architectures séquentielles, la réécriture de certaines fonctions, et la structuration finale du projet.
- Stack Overflow pour la gestion des erreurs liées à SMOTE, joblib, reshape de tenseurs, et débogage des modèles Keras.

## Auteur

Vicente Campello — Avril 2025  
Projet réalisé dans le cadre du module de data science, semestre 2.

Lien vers la vidéo de soutenance : Que pour les intimes.

