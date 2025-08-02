import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from analytique_sante import AnalytiqueAvanceeSante
import os

# Créez un répertoire pour les fichiers de test si non existant
@pytest.fixture(scope='session', autouse=True)
def create_test_data_dir(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("test_data")
    return test_dir

@pytest.fixture
def sample_data_path(create_test_data_dir):
    # Correction : suppression de l'espace final dans le nom du service VIH
    data = {
        'Service': [
            'Nombre de consultants',            # Clé
            'Accouchement dans l’établissement', # Clé
            'TOTAL PALUDISME',                 # Clé
            'Service Non Pertinent A',         # Non clé
            'Nombre de consultations',         # Clé
            'Service Non Pertinent B',         # Non clé
            'Naissances vivantes',             # Clé
            'TOTAL IRA',                       # Clé
            'Femme Enceinte ou allaitante dépistée VIH+' # Clé (corrigé)
        ],
        'JANVIER': [100, 50, 120, 10, 60, 5, 30, 40, 25],
        'FÉVRIER': [110, 55, 130, 12, 65, 6, 32, 42, 28],
        'MARS': [120, 60, 140, 15, 70, 7, 35, 45, 30],
        'AVRIL': [130, 65, 150, 18, 75, 8, 38, 48, 32],
        'MAI': [140, 70, 160, 20, 80, 9, 40, 50, 35],
        'JUIN': [150, 75, 170, 22, 85, 10, 42, 52, 38],
        'JUILLET': [160, 80, 180, 25, 90, 11, 45, 55, 40],
        'AOÛT': [170, 85, 190, 28, 95, 12, 48, 58, 42],
        'SEPTEMBRE': [180, 90, 200, 30, 100, 13, 50, 60, 45],
        'OCTOBRE': [190, 95, 210, 32, 105, 14, 52, 62, 48],
        'NOVEMBRE': [200, 100, 220, 35, 110, 15, 55, 65, 50],
        'DÉCEMBRE': [210, 105, 230, 38, 115, 16, 58, 68, 52],
    }
    df = pd.DataFrame(data)
    filepath = create_test_data_dir / "sample_data_2023.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

@pytest.fixture
def sample_data_with_missing_months_path(create_test_data_dir):
    data = {
        'Service': ['Nombre de consultants', 'TOTAL IRA'],
        'JANVIER': [100, 50],
        'FÉVRIER': [110, 55],
    }
    df = pd.DataFrame(data)
    filepath = create_test_data_dir / "sample_data_missing_months_2024.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

@pytest.fixture
def non_existent_data_path():
    return "non_existent_file.csv"

@pytest.fixture
def empty_file_path(create_test_data_dir):
    filepath = create_test_data_dir / "empty_file.csv"
    with open(filepath, 'w') as f:
        f.write("")
    return str(filepath)

@pytest.fixture
def invalid_data_path(create_test_data_dir):
    filepath = create_test_data_dir / "invalid_data.csv"
    with open(filepath, 'w') as f:
        f.write("col1,col2\nval1,val2\n")
    return str(filepath)

@pytest.fixture
def data_without_full_months(create_test_data_dir):
    data = {
        'Service': ['Nombre de consultations', 'Naissances vivantes'],
        'JANVIER': [100, 50],
        'FÉVRIER': [110, 55],
    }
    df = pd.DataFrame(data)
    filepath = create_test_data_dir / "data_without_full_months_2025.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

class TestAnalytiqueAvanceeSante:

    def test_initialization_success(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        assert analytique.df is not None
        assert not analytique.df.empty
        assert analytique.annee == 2023
        
        # Vérification des services standardisés
        expected_standard_services = sorted([
            'Consultants', 'Accouchements', 'PaludismeTotal',
            'Consultations', 'NaissancesVivantes', 'IRATotal', 'FemmesVIH+'
        ])
        assert sorted(analytique.get_services_uniques()) == expected_standard_services

    def test_initialization_file_not_found(self, non_existent_data_path):
        analytique = AnalytiqueAvanceeSante(non_existent_data_path)
        assert analytique.df is None
        assert "Erreur: Le fichier n'existe pas" in analytique.erreurs

    def test_initialization_empty_file(self, empty_file_path):
        analytique = AnalytiqueAvanceeSante(empty_file_path)
        assert analytique.df is None
        assert "Erreur: Le fichier est vide ou illisible." in analytique.erreurs

    def test_initialization_invalid_data_format(self, invalid_data_path):
        analytique = AnalytiqueAvanceeSante(invalid_data_path)
        assert analytique.df is None
        assert "Erreur: Le fichier ne contient pas les colonnes essentielles" in analytique.erreurs

    def test_transformer_en_mensuel(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        assert analytique.df_mensuel is not None
        assert 'ds' in analytique.df_mensuel.columns
        assert 'y' in analytique.df_mensuel.columns
        assert 'service' in analytique.df_mensuel.columns
        
        # Vérification du type de données des dates
        assert pd.api.types.is_datetime64_dtype(analytique.df_mensuel['ds'])
        
        # Vérification des valeurs pour Janvier
        jan_data = analytique.df_mensuel[analytique.df_mensuel['ds'].dt.month == 1]
        assert jan_data['y'].sum() == 100 + 50 + 120 + 60 + 30 + 40 + 25  # 425

    def test_transformer_en_mensuel_partial_months(self, data_without_full_months):
        analytique = AnalytiqueAvanceeSante(data_without_full_months)
        assert analytique.df_mensuel is not None
        assert 'ds' in analytique.df_mensuel.columns
        assert 'y' in analytique.df_mensuel.columns
        assert 'service' in analytique.df_mensuel.columns
        assert len(analytique.df_mensuel) == 4  # 2 services × 2 mois

    def test_get_annees_disponibles(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        annees = analytique.get_annees_disponibles()
        assert annees == [2023]

    def test_get_services_uniques(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        services = analytique.get_services_uniques()
        assert services == [
            'Consultants', 'Accouchements', 'PaludismeTotal',
            'Consultations', 'NaissancesVivantes', 'IRATotal', 'FemmesVIH+'
        ]

    def test_categorisation_automatique(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        assert 'categorie_service' in analytique.df_mensuel.columns
        
        # Vérification des catégories
        categories = analytique.df_mensuel['categorie_service'].unique()
        expected_categories = [
            'Consultations & Personnel',
            'Maternité & Naissance',
            'Maladies Majeures',
            'Dépistage & Diagnostic'
        ]
        for cat in expected_categories:
            assert cat in categories

    def test_get_data_summary(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        summary = analytique.get_data_summary()
        
        assert summary['nombre_services_uniques'] == 7
        assert summary['total_consultations_ou_interventions'] == 425 * 12  # Total annuel

    def test_preparer_donnees_prediction(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'Consultants'
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        
        assert len(df_prepared) == 12
        assert df_prepared['y'].iloc[0] == 100  # Janvier

    def test_preparation_service_inexistant(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        df = analytique.preparer_donnees_prediction("Service Inconnu")
        assert df is None
        assert "Service non trouvé" in analytique.erreurs

    def test_faire_predictions_prophet(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'Consultants'
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        
        predictions = analytique.faire_predictions(df_prepared, modele='Prophet', periodes_futur=3)
        assert len(predictions) == 15  # 12 mois + 3 prévisions
        assert predictions['yhat'].min() > 0  # Valeurs cohérentes

    def test_faire_predictions_random_forest(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'Accouchements'
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        
        predictions = analytique.faire_predictions(df_prepared, modele='RandomForest', periodes_futur=3)
        assert len(predictions) == 15
        assert 'yhat' in predictions.columns

    def test_faire_predictions_invalid_model(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'PaludismeTotal'
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        
        predictions = analytique.faire_predictions(df_prepared, modele='InvalidModel', periodes_futur=3)
        assert predictions is None
        assert "Modèle de prévision non supporté" in analytique.erreurs

    def test_optimisation_des_ressources(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'Consultants'
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        predictions = analytique.faire_predictions(df_prepared, modele='Prophet', periodes_futur=3)
        
        # Ajout du service aux prédictions
        predictions['service'] = service_test
        
        analytique.optimiser_ressources(
            predictions_df=predictions,
            type_ressource='consultants',
            cout_moyen_ressource=1000
        )
        
        resultats = analytique.resultats_optimisation[service_test]
        assert 'ressources_necessaires' in resultats['predictions_optimisees'].columns
        assert resultats['total_cout_estime'] > 0

    def test_integration_data_flow(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'Consultants'
        
        # Chargement et transformation
        assert analytique.df is not None
        assert analytique.df_mensuel is not None
        
        # Préparation et prédiction
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        predictions = analytique.faire_predictions(df_prepared, modele='Prophet', periodes_futur=3)
        
        # Optimisation
        predictions['service'] = service_test
        analytique.optimiser_ressources(predictions, type_ressource='consultants')
        
        # Vérification finale
        resultats = analytique.resultats_optimisation[service_test]
        assert not resultats['predictions_optimisees'].empty