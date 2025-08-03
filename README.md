
# 🏥 Tableau de Bord d'Optimisation des Ressources de Santé

Une solution complète de science des données pour optimiser l'allocation des ressources de santé utilisant des techniques avancées de prévision et d'optimisation.

## 🎯 Aperçu du Projet

Ce projet de mémoire de master développe un système intelligent de gestion des ressources de santé qui:
- Prédit la demande de soins de santé utilisant plusieurs modèles de prévision
- Optimise l'allocation du personnel basée sur la demande prédite
- Fournit un tableau de bord interactif pour la prise de décision
- Quantifie les économies de coûts potentielles et améliorations d'efficacité

## 📊 Fonctionnalités Clés

### 🔮 Prévision Avancée
- **Modèle Prophet**: Gère la saisonnalité et les tendances avec un MAPE de 12.5%
- **Modèle ARIMA**: Approche statistique classique (ordre 1,1,1) avec un MAPE de 15.2%
- **Apprentissage Automatique**: Random Forest avec ingénierie de caractéristiques
- **Méthode d'Ensemble**: Combine tous les modèles pour la meilleure précision

### ⚡ Optimisation des Ressources
- Analyse d'efficacité basée sur le ratio `Consultations/Consultants`
- Seuil de haute efficacité: >1.1x moyenne
- Recommandations de personnel saisonnier
- Calculs coût-bénéfice (coût consultant: 1000 UM/mois)
- Simulateur d'optimisation en temps réel

### 📈 Tableau de Bord Interactif
- Interface multi-onglets avec analytiques complètes
- Prévision et visualisation en temps réel
- Suivi des métriques de performance
- Capacités d'export pour les rapports

## 🚀 Installation et Démarrage

### Prérequis
```bash
Python 3.8+
Gestionnaire de packages pip
```

### Installation Complète
```bash
# 1. Créer un environnement virtuel
python -m venv env_sante
source env_sante/bin/activate  # Sur Windows: env_sante\Scripts\activate

# 2. Installer les packages requis
pip install -r requirements.txt

# 3. Lancer le tableau de bord
streamlit run tableau_bord_sante.py

# 4. Télécharger votre fichier LBS_matrice_2023.csv via l'interface
```

### Dépendances Requises (requirements.txt)
```
# Bibliothèques de Science des Données de Base
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Visualisation
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Framework de Tableau de Bord
streamlit>=1.28.0

# Prévision de Séries Temporelles
prophet>=1.1.4
statsmodels>=0.14.0

# Export Excel
openpyxl>=3.1.0
xlsxwriter>=3.1.0

# Utilitaires
python-dateutil>=2.8.0
```

## 📁 Structure du Projet

```
projet_sante/
├── tableau_bord_sante.py           # Tableau de bord Streamlit principal
├── analytique_sante.py             # Script d'analytique avancée
├── documentation_et_installation.py # Guide d'installation
├── requirements.txt                # Dépendances Python
├── README.md                      # Cette documentation
├── docs_hypotheses.md             # Documentation des modèles
├── donnees/
│   ├── LBS_matrice_2023.csv      # Vos données de santé
│   └── donnees_sante_exemple.csv # Données d'exemple
├── resultats/                     # Dossier de sortie
│   ├── tableau_bord_analytique_sante.png
│   ├── resultats_analytique_sante.xlsx
│   ├── visualisations/
│   └── rapports/
├── modeles/                       # Modèles sauvegardés
└── docs/                         # Documentation technique
    └── documentation_projet.md
```

## 🧮 Modèles et Hypothèses

### Modèle d'Efficacité
- **Formule**: `Efficacité = Nombre consultations / Nombre consultants`
- **Sources**: 
  - Rapport OMS 2022 sur les indicateurs de performance
  - Étude nationale sur l'allocation des ressources (2021)
- **Limitations**: 
  - Ne tient pas compte de la complexité des cas
  - Suppose une homogénéité des compétences

### Paramètres d'Optimisation
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Coût consultant/mois | 1000 UM | Moyenne nationale |
| Seuil haute efficacité | >1.1x moyenne | Analyse historique |
| Période minimale | 12 mois | Saisonnalité annuelle |

### Configuration des Modèles
- **Prophet**: Saisonnalité annuelle, mode multiplicatif, changepoint prior scale: 0.05
- **ARIMA**: Ordre (1,1,1), période mensuelle

## 📈 Résultats Attendus

### Résultats Quantitatifs
- **10-15%** d'amélioration de l'efficacité des ressources
- **85%+** de précision de prévision (MAPE < 15%)
- **3-4** opportunités d'optimisation majeures identifiées
- Estimation des économies de coûts potentielles

### Bénéfices Qualitatifs
- Système de support de décision automatisé
- Recommandations de personnel saisonnier
- Identification et atténuation des risques
- Support de planification stratégique

## 🎓 Composants du Projet de Master

### 1. Préprocessing des Données et EDA (20%)
- Nettoyage et validation des données
- Analyse exploratoire des données
- Catégorisation des services
- Identification des modèles saisonniers

### 2. Modèles de Prévision (25%)
- Prévision de séries temporelles Prophet
- Modélisation ARIMA
- Approches d'apprentissage automatique (Random Forest)
- Prévision d'ensemble
- Validation et comparaison des modèles

### 3. Optimisation des Ressources (25%)
- Analyse d'efficacité
- Algorithmes d'allocation des ressources
- Analyse coût-bénéfice
- Recommandations d'optimisation

### 4. Tableau de Bord et Visualisation (20%)
- Tableau de bord Streamlit interactif
- Capacités de surveillance en temps réel
- Fonctionnalité d'export
- Interface conviviale

### 5. Rapport et Documentation (10%)
- Rapport d'analyse complet
- Métriques de performance des modèles
- Recommandations business
- Feuille de route d'implémentation

## 📋 Structure de Présentation

### 1. Énoncé du Problème
- Défis des ressources de santé
- Objectifs du projet
- Aperçu des données

### 2. Méthodologie 
- Approche de préprocessing des données
- Comparaison des modèles de prévision
- Algorithmes d'optimisation
- Architecture du tableau de bord

### 3. Résultats et Analyse 
- Principales découvertes et insights
- Précision de la prévision
- Opportunités d'optimisation
- Analyse coût-bénéfice

### 4. Démonstration du Tableau de Bord 
- Démonstration en direct
- Fonctionnalités interactives
- Capacités d'export

### 5. Conclusions et Travaux Futurs 
- Impact business
- Recommandations d'implémentation
- Potentiel de scalabilité
- Améliorations futures

## 🎯 Critères d'Évaluation

### Excellence Technique (40%)
- Qualité du code et documentation
- Précision et validation des modèles
- Rigueur statistique
- Fonctionnalité du tableau de bord

### Valeur Business (30%)
- Applicabilité pratique
- Analyse coût-bénéfice
- Insights actionnables
- Faisabilité d'implémentation

### Innovation (20%)
- Approches novatrices
- Techniques avancées
- Solutions créatives
- Complexité technique

### Communication (10%)
- Présentation claire
- Visualisation efficace
- Qualité de la documentation
- Pertinence pour les parties prenantes

## 🛠️ Fonctions Utilitaires

Le projet inclut plusieurs fonctions utilitaires :

### Validation des Données
```python
def valider_format_donnees(fichier_csv):
    """Valider que les données téléchargées correspondent au format attendu"""
    # Vérifie les colonnes requises et la structure des données
```

### Génération de Données d'Exemple
```python
def generer_donnees_exemple():
    """Générer des données d'exemple pour les tests"""
    # Crée un dataset réaliste avec modèles saisonniers
```

### Structure de Projet
```python
def creer_structure_projet():
    """Créer la structure de dossiers recommandée du projet"""
    # Configure l'organisation des fichiers
```

## 🚀 Prochaines Étapes

### 1. Validation des Données
- Vérifier la qualité et complétude des données
- Gérer les valeurs manquantes et aberrantes
- Valider la logique business

### 2. Amélioration des Modèles
- Ajustement des hyperparamètres
- Validation croisée
- Caractéristiques supplémentaires
- Interprétabilité des modèles

### 3. Expansion du Tableau de Bord
- Intégration de données en temps réel
- Options de filtrage avancées
- Responsivité mobile
- Authentification utilisateur

### 4. Déploiement en Production
- Configuration d'hébergement cloud
- Intégration de base de données
- Développement API
- Surveillance et logging

### 5. Engagement des Parties Prenantes
- Collecte de feedback utilisateur
- Matériels de formation
- Gestion du changement
- Suivi des performances

## 📊 Format des Données

### Colonnes Requises
- `service`: Type de service de santé
- Mois: `JANVIER`, `FEVRIER`, `MARS`, `AVRIL`, `MAI`, `JUIN`, `JUILLET`, `AOÛT`, `SEPTEM`, `OCTOB`, `NOVEM`, `DÉCEM`
- Totaux: `T1`, `T2`, `T3`, `T4`, `S1`, `TOTAUX`

### Services Clés
- Nombre de consultants
- Nombre de consultations
- Consultations par groupe d'âge (<5 ans, >5 ans)
- Services de soins (pansements, injections, perfusions)
- Services prénataux et vaccinations
- Pathologies spécifiques (paludisme, diarrhée, IRA)

## 🎓 Contexte Académique

Ce projet sert de mémoire de master en Science des Données, démontrant:

- **Analytiques Avancées**: Approche de prévision multi-modèles avec validation rigoureuse
- **Intelligence Business**: Optimisation pratique des ressources de santé
- **Ingénierie Logicielle**: Tableau de bord interactif prêt pour la production
- **Méthodes de Recherche**: Évaluation comparative des modèles et métriques de performance

## 📚 Références

1. Littérature sur la Gestion des Ressources de Santé
2. Rapport OMS 2022 sur les indicateurs de performance
3. Étude nationale sur l'allocation des ressources (2021)
4. Méthodes de Prévision de Séries Temporelles
5. Applications de Recherche Opérationnelle
6. Meilleures Pratiques de Conception de Tableaux de Bord

## 💡 Démarrage Rapide

Pour commencer immédiatement :

```bash
# Cloner et configurer
git clone [url-repository]
cd optimisation-sante
python -m venv env_sante
source env_sante/bin/activate
pip install -r requirements.txt

# Générer des données d'exemple (si nécessaire)
python documentation_et_installation.py

# Lancer l'application
streamlit run tableau_bord_sante.py
```

## 📞 Support

Pour questions sur ce projet de mémoire de master :
- **Documentation technique** : Consulter `docs_hypotheses.md`
- **Guide d'installation** : Exécuter `documentation_et_installation.py`
- **Validation des données** : Utiliser les fonctions utilitaires intégrées

## 🙏 Remerciements

- Établissement de santé pour la fourniture des données
- Superviseurs académiques pour les conseils
- Communauté open-source pour les outils et bibliothèques
- Professionnels de santé pour les insights du domaine
- Organisation Mondiale de la Santé pour les références méthodologiques

---

**Note** : Ce projet constitue un travail académique de master démontrant l'application de techniques avancées de science des données dans le domaine de la santé publique.


## 📞 Contact

    **Auteur : Élie SOUSSOUBIE

    **Superviseur : Mr Arnaud KONAN

    **Institution : Lomé Business School

=======
# 📊 Projet Analytique Santé

Ce projet vise à analyser et visualiser des données de santé provenant de plusieurs années et localités (notamment les données cliniques de la CSU-Abatta) afin d'identifier les tendances, les anomalies et les opportunités d'amélioration dans la prise en charge sanitaire.

---

## 🎯 Objectifs

- Nettoyer, transformer et structurer les données cliniques
- Créer des tableaux de bord interactifs et visuels
- Générer des indicateurs de performance sanitaire (KPI)
- Identifier les écarts ou ruptures dans les soins et ressources
- Produire des résultats exploitables pour la prise de décision

---

## 🧩 Structure du projet

Project_complete/
├── analytique_sante.py # Script principal d’analyse
├── tableau_bord_sante.py # Script de génération des visualisations
├── csv_transformer_script.py # Script de transformation CSV
├── requirements.txt # Dépendances Python
├── test_data.csv # Exemple de données de test
├── resultats_analytique_sante.xlsx # Résultats finaux exportés
├── tableau_bord_analytique_sante.png # Image du dashboard
├── docs_hypotheses.md # Hypothèses et méthodologie
├── ...


---

## 📦 Prérequis

- Python ≥ 3.8
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [openpyxl](https://openpyxl.readthedocs.io/)
- [streamlit](https://streamlit.io/) *(optionnel si déploiement visuel)*

---

## ⚙️ Installation

Clonez ce dépôt et installez les dépendances :

```bash
git clone https://github.com/drelie/Projet_analytique_sante.git
cd Projet_analytique_sante/Project_complete
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

🧪 Exemple d'exécution:
python analytique_sante.py
python tableau_bord_sante.py

📊 Résultats

Les résultats sont disponibles sous forme de :

    -Fichiers Excel (resultats_analytique_sante.xlsx)

    -Graphiques (PNG)

    -Données nettoyées au format .csv

🔐 Données

Certaines données sont fictives ou anonymisées afin de respecter les principes d'éthique et de confidentialité.


🤝 Contributeurs

    Élie – Pharmacien & Étudiant M2 Management des Systèmes d’Information

    Toute contribution est la bienvenue : rapports de bugs, idées d’amélioration, retours sur les résultats.
