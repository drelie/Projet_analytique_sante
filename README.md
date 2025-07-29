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
