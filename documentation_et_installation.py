# Tableau de Bord d'Optimisation des Ressources de Santé
# Guide d'Installation et Exigences

import pandas as pd

"""
GUIDE D'INSTALLATION
====================

1. Créer un environnement virtuel:
   python -m venv env_sante
   source env_sante/bin/activate  # Sur Windows: env_sante\Scripts\activate

2. Installer les packages requis:
   pip install -r requirements.txt

3. Lancer le tableau de bord:
   streamlit run tableau_bord_sante.py

4. Téléchargez votre fichier LBS_matrice_2023.csv via l'interface

REQUIREMENTS.txt
================
"""

exigences = """
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
"""

# Sauvegarder les exigences dans un fichier
with open('requirements.txt', 'w') as f:
    f.write(exigences.strip())

print("✅ Fichier des exigences créé!")

"""
STRUCTURE DU PROJET
===================

projet_sante/
├── tableau_bord_sante.py           # Tableau de bord Streamlit principal
├── analytique_sante.py             # Script d'analytique avancée
├── requirements.txt                # Dépendances Python
├── LBS_matrice_2023.csv           # Vos données de santé
├── resultats/                     # Dossier de sortie
│   ├── tableau_bord_analytique_sante.png
│   └── resultats_analytique_sante.xlsx
└── README.md                      # Documentation du projet

COMPOSANTS DU PROJET DE MASTER
===============================

1. Préprocessing des Données et EDA (20%)
   - Nettoyage et validation des données
   - Analyse exploratoire des données
   - Catégorisation des services
   - Identification des modèles saisonniers

2. Modèles de Prévision (25%)
   - Prévision de séries temporelles Prophet
   - Modélisation ARIMA
   - Approches d'apprentissage automatique (Random Forest)
   - Prévision d'ensemble
   - Validation et comparaison des modèles

3. Optimisation des Ressources (25%)
   - Analyse d'efficacité
   - Algorithmes d'allocation des ressources
   - Analyse coût-bénéfice
   - Recommandations d'optimisation

4. Tableau de Bord et Visualisation (20%)
   - Tableau de bord Streamlit interactif
   - Capacités de surveillance en temps réel
   - Fonctionnalité d'export
   - Interface conviviale

5. Rapport et Documentation (10%)
   - Rapport d'analyse complet
   - Métriques de performance des modèles
   - Recommandations business
   - Feuille de route d'implémentation

RÉSULTATS ATTENDUS
==================

Résultats Quantitatifs:
- 10-15% d'amélioration de l'efficacité des ressources
- 85%+ de précision de prévision (MAPE < 15%)
- Identification de 3-4 opportunités d'optimisation
- Estimation des économies de coûts potentielles

Résultats Qualitatifs:
- Système de support de décision automatisé
- Recommandations de personnel saisonnier
- Identification et atténuation des risques
- Support de planification stratégique

STRUCTURE DE PRÉSENTATION
=========================

1. Énoncé du Problème (5 minutes)
   - Défis des ressources de santé
   - Objectifs du projet
   - Aperçu des données

2. Méthodologie (10 minutes)
   - Approche de préprocessing des données
   - Comparaison des modèles de prévision
   - Algorithmes d'optimisation
   - Architecture du tableau de bord

3. Résultats et Analyse (15 minutes)
   - Principales découvertes et insights
   - Précision de la prévision
   - Opportunités d'optimisation
   - Analyse coût-bénéfice

4. Démonstration du Tableau de Bord (10 minutes)
   - Démonstration en direct
   - Fonctionnalités interactives
   - Capacités d'export

5. Conclusions et Travaux Futurs (5 minutes)
   - Impact business
   - Recommandations d'implémentation
   - Potentiel de scalabilité
   - Améliorations futures

CRITÈRES D'ÉVALUATION
=====================

Excellence Technique (40%):
- Qualité du code et documentation
- Précision et validation des modèles
- Rigueur statistique
- Fonctionnalité du tableau de bord

Valeur Business (30%):
- Applicabilité pratique
- Analyse coût-bénéfice
- Insights actionnables
- Faisabilité d'implémentation

Innovation (20%):
- Approches novatrices
- Techniques avancées
- Solutions créatives
- Complexité technique

Communication (10%):
- Présentation claire
- Visualisation efficace
- Qualité de la documentation
- Pertinence pour les parties prenantes

PROCHAINES ÉTAPES
=================

1. Validation des Données:
   - Vérifier la qualité et complétude des données
   - Gérer les valeurs manquantes et aberrantes
   - Valider la logique business

2. Amélioration des Modèles:
   - Ajustement des hyperparamètres
   - Validation croisée
   - Caractéristiques supplémentaires
   - Interprétabilité des modèles

3. Expansion du Tableau de Bord:
   - Intégration de données en temps réel
   - Options de filtrage avancées
   - Responsivité mobile
   - Authentification utilisateur

4. Déploiement en Production:
   - Configuration d'hébergement cloud
   - Intégration de base de données
   - Développement API
   - Surveillance et logging

5. Engagement des Parties Prenantes:
   - Collecte de feedback utilisateur
   - Matériels de formation
   - Gestion du changement
   - Suivi des performances
"""

# Fonctions utilitaires supplémentaires pour le projet

def creer_structure_projet():
    """Créer la structure de dossiers recommandée du projet"""
    import os
    
    dossiers = [
        'projet_sante',
        'projet_sante/donnees',
        'projet_sante/resultats',
        'projet_sante/modeles',
        'projet_sante/docs'
    ]
    
    for dossier in dossiers:
        os.makedirs(dossier, exist_ok=True)
    
    print("✅ Structure du projet créée!")

def generer_donnees_exemple():
    """Générer des données d'exemple pour les tests si les données originales ne sont pas disponibles"""
    import pandas as pd
    import numpy as np
    
    # Créer la structure des données de santé d'exemple
    mois = ['JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN', 
            'JUILLET', 'AOÛT', 'SEPTEM', 'OCTOB', 'NOVEM', 'DÉCEM']
    
    services = [
        'Nombre de consultants',
        'Nombre de consultations',
        'Nombre de consultants <5 ans',
        'Nombre de consultants >5 ans',
        'Nombre de consultations <5 ans',
        'Nombre de consultations >5 ans',
        'Cas M O',
        'Cas référés vers une autre structure sanitaire',
        'Pansements',
        'Injections',
        'Perfusions',
        'Autres soins',
        'Petite chirurgie, circoncision',
        'Petite chirurgie, suture de plaie',
        '1ère consultation prénatale 1er trimestre',
        'Total CPN1',
        'Grossesses à risque dépistées en CPN1',
        'Vaccin BCG',
        'Vaccin Polio 0',
        'Vaccin Penta 1',
        'Paludisme grave <5 ans',
        'Paludisme simple <5 ans',
        'Diarrhée simple <5 ans',
        'IRA/Pneumonie <5 ans'
    ]
    
    # Générer des données d'exemple réalistes
    np.random.seed(42)
    donnees_exemple = []
    
    for i, service in enumerate(services):
        # Valeurs de base avec modèles saisonniers
        valeur_base = np.random.randint(50, 1000)
        modele_saisonnier = np.sin(np.arange(12) * 2 * np.pi / 12) * 0.3 + 1
        
        ligne = {'PAGE': i + 1, 'service': service}
        valeurs_mensuelles = []
        
        for j, mois_nom in enumerate(mois):
            # Ajouter saisonnalité et variation aléatoire
            valeur = int(valeur_base * modele_saisonnier[j] * np.random.uniform(0.8, 1.2))
            ligne[mois_nom] = valeur
            valeurs_mensuelles.append(valeur)
        
        # Calculer les totaux
        ligne['T1'] = sum(valeurs_mensuelles[:3])
        ligne['T2'] = sum(valeurs_mensuelles[3:6])
        ligne['T3'] = sum(valeurs_mensuelles[6:9])
        ligne['T4'] = sum(valeurs_mensuelles[9:12])
        ligne['S1'] = ligne['T1'] + ligne['T2']
        ligne['TOTAUX'] = sum(valeurs_mensuelles)
        
        donnees_exemple.append(ligne)
    
    # Créer DataFrame et sauvegarder
    df = pd.DataFrame(donnees_exemple)
    df.to_csv('donnees_sante_exemple.csv', index=False)
    
    print("✅ Données d'exemple générées comme 'donnees_sante_exemple.csv'")
    return df

def valider_format_donnees(fichier_csv):
    """Valider que les données téléchargées correspondent au format attendu"""
    try:
        df = pd.read_csv(fichier_csv)
        
        colonnes_requises = ['service', 'JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 
                           'MAI', 'JUIN', 'JUILLET', 'AOÛT', 'SEPTEM', 
                           'OCTOB', 'NOVEM', 'DÉCEM', 'TOTAUX']
        
        colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
        
        if colonnes_manquantes:
            print(f"❌ Colonnes manquantes: {colonnes_manquantes}")
            return False
        
        # Vérifier les services clés
        services_cles = ['Nombre de consultants', 'Nombre de consultations']
        services_disponibles = df['service'].dropna().tolist()
        
        services_manquants = [srv for srv in services_cles if srv not in services_disponibles]
        if services_manquants:
            print(f"⚠️ Services clés manquants: {services_manquants}")
        
        print("✅ Validation du format des données réussie!")
        print(f"   - Lignes: {len(df)}")
        print(f"   - Services: {df['service'].nunique()}")
        print(f"   - Colonnes: {len(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Échec de la validation des données: {str(e)}")
        return False

# Template de documentation du projet de Master

documentation_projet = """
# Tableau de Bord d'Optimisation des Ressources de Santé
## Mémoire de Master en Science des Données

### Résumé
Ce projet développe un système avancé d'optimisation des ressources de santé utilisant l'apprentissage automatique et la prévision de séries temporelles pour améliorer l'efficacité d'allocation des ressources dans les établissements de santé. Le système analyse les données historiques de services de santé pour prédire les modèles de demande et optimiser l'allocation du personnel, résultant en une amélioration de l'efficacité opérationnelle et des économies de coûts.

### 1. Introduction

#### 1.1 Énoncé du Problème
Les établissements de santé font face à des défis significatifs dans l'allocation des ressources dus à:
- Variations saisonnières de la demande
- Flux de patients imprévisibles
- Flexibilité limitée du personnel
- Contraintes budgétaires
- Exigences de qualité de service

#### 1.2 Objectifs
- Développer des modèles de prévision de demande précis
- Créer des algorithmes d'optimisation des ressources
- Construire un tableau de bord interactif pour le support de décision
- Quantifier les économies de coûts potentielles et améliorations d'efficacité

#### 1.3 Description des Données
Le jeu de données contient des statistiques mensuelles de services de santé pour 2023, incluant:
- Nombres de consultants et consultations
- Démographie par groupe d'âge (<5 ans, >5 ans)
- Catégories de services (soins prénataux, vaccinations, traitements)
- Cas de pathologies (paludisme, diarrhée, infections respiratoires)

### 2. Méthodologie

#### 2.1 Préprocessing des Données
- Nettoyage et validation des données
- Imputation des valeurs manquantes
- Catégorisation des services
- Préparation des séries temporelles

#### 2.2 Analyse Exploratoire des Données
- Identification des modèles saisonniers
- Analyse de corrélation des services
- Évaluation de la variabilité de la demande
- Calcul des métriques d'efficacité

### Références
1. Littérature sur la Gestion des Ressources de Santé
2. Méthodes de Prévision de Séries Temporelles
3. Applications de Recherche Opérationnelle
4. Meilleures Pratiques de Conception de Tableaux de Bord
5. Études de Cas d'Analytique de Santé
"""

def creer_readme():
    """Créer un fichier README complet"""
    contenu_readme = """
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
- **Modèle Prophet**: Gère la saisonnalité et les tendances
- **Modèle ARIMA**: Approche statistique classique  
- **Apprentissage Automatique**: Random Forest avec ingénierie de caractéristiques
- **Méthode d'Ensemble**: Combine tous les modèles pour la meilleure précision

### ⚡ Optimisation des Ressources
- Analyse d'efficacité et benchmarking
- Recommandations de personnel saisonnier
- Calculs coût-bénéfice
- Simulateur d'optimisation en temps réel

### 📈 Tableau de Bord Interactif
- Interface multi-onglets avec analytiques complètes
- Prévision et visualisation en temps réel
- Suivi des métriques de performance
- Capacités d'export pour les rapports

## 🚀 Démarrage Rapide

### Prérequis
```bash
Python 3.8+
gestionnaire de packages pip
```

### Installation
```bash
# Cloner le repository
git clone [url-repository]
cd optimisation-sante

# Créer environnement virtuel
python -m venv env_sante
source env_sante/bin/activate  # Sur Windows: env_sante\\Scripts\\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer le Tableau de Bord
```bash
streamlit run tableau_bord_sante.py
```

## 📁 Structure du Projet

```
projet_sante/
├── tableau_bord_sante.py          # Tableau de bord Streamlit principal
├── analytique_sante.py            # Moteur d'analytique avancée
├── requirements.txt               # Dépendances Python
├── README.md                     # Ce fichier
├── donnees/
│   └── LBS_matrice_2023.csv     # Données de santé
├── resultats/
│   ├── visualisations/
│   └── rapports/
└── docs/
    └── documentation_projet.md
```

## 📈 Résultats Attendus

### Résultats Quantitatifs
- **10-15%** d'amélioration de l'efficacité des ressources
- **85%+** de précision de prévision (MAPE < 15%)
- **3-4** opportunités d'optimisation majeures identifiées
- Potentiel d'économies de coûts quantifié

### Bénéfices Qualitatifs
- Support de décision automatisé
- Capacités de planification saisonnière
- Identification et atténuation des risques
- Planification stratégique des ressources

## 🎓 Contexte Académique

Ce projet sert de mémoire de master en Science des Données, démontrant:
- **Analytiques Avancées**: Approche de prévision multi-modèles
- **Intelligence Business**: Optimisation pratique de la santé
- **Ingénierie Logicielle**: Tableau de bord prêt pour la production
- **Méthodes de Recherche**: Évaluation et validation rigoureuses

## 📞 Contact

Pour questions sur ce projet de mémoire de master:
- Superviseur académique: [Nom du superviseur]
- Étudiant: [Votre nom]
- Institution: [Nom de l'université]

## 🙏 Remerciements

- Établissement de santé pour la fourniture des données
- Superviseurs académiques pour les conseils
- Communauté open-source pour les outils
- Professionnels de santé pour les insights du domaine
"""
    
    with open('README.md', 'w') as f:
        f.write(contenu_readme)
    
    print("✅ README.md créé!")

# Exécution principale
if __name__ == "__main__":
    print("🏥 CONFIGURATION DU PROJET D'OPTIMISATION DE SANTÉ")
    print("=" * 50)
    
    print("\n1. Création du fichier des exigences...")
    # Exigences déjà créées ci-dessus
    
    print("\n2. Création de la documentation du projet...")
    with open('documentation_projet.md', 'w') as f:
        f.write(documentation_projet)
    
    print("\n3. Création du fichier README...")
    creer_readme()
    
    print("\n4. Génération de données d'exemple pour les tests...")
    generer_donnees_exemple()
    
    print("\n5. Création de la structure du projet...")
    creer_structure_projet()
    
    print("\n✅ CONFIGURATION DU PROJET TERMINÉE!")
    print("\nÉtapes suivantes:")
    print("1. Installer les exigences: pip install -r requirements.txt")
    print("2. Lancer le tableau de bord: streamlit run tableau_bord_sante.py")  
    print("3. Télécharger votre fichier LBS_matrice_2023.csv")
    print("4. Explorer les fonctionnalités d'analytique et d'optimisation")
    
    print(f"\n📚 Composants du Projet de Master:")
    print("✓ Tableau de Bord Streamlit Interactif")
    print("✓ Moteur d'Analytique Avancée") 
    print("✓ Prévision Multi-Modèles")
    print("✓ Optimisation des Ressources")
    print("✓ Documentation Complète")
    print("✓ Génération de Données d'Exemple")
    print("✓ Fonctionnalités d'Export et Rapport")
    print("✓ Structure de Projet Recommandée")
    print("✓ Validation de Format de Données")
    print("✓ Instructions d'Installation et de Démarrage")
