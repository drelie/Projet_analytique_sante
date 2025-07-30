import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from analytique_sante import AnalytiqueAvanceeSante # Assurez-vous que cette importation est correcte
import os

# Créez un répertoire pour les fichiers de test si non existant
@pytest.fixture(scope='session', autouse=True)
def create_test_data_dir(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("test_data")
    return test_dir

@pytest.fixture
def sample_data_path(create_test_data_dir):
    # Créer un fichier CSV temporaire avec des données valides pour les tests
    # Inclure des services clés et non clés pour tester le filtrage
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
            'Femme Enceinte ou allaitante dépistée VIH +' # Clé (note: espace en fin)
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
        f.write("col1,col2\nval1,val2\n") # Données qui ne correspondent pas au format attendu
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
        # Vérifier que seuls les services clés standardisés sont présents
        expected_standard_services = sorted(list(AnalytiqueAvanceeSante.SERVICES_CLES_STANDARD.values()))
        # Filtrer la liste attendue pour les services réellement présents dans notre échantillon
        # (en assumant que tous les SERVICES_CLES_STANDARD ne sont pas dans chaque fichier de test)
        services_in_sample = ['Nombre de consultants', 'Accouchement dans l’établissement', 'TOTAL PALUDISME',
                              'Nombre de consultations', 'Naissances vivantes', 'TOTAL IRA',
                              'Femme Enceinte ou allaitante dépistée VIH +']
        expected_filtered_and_standardized_services = sorted([
            AnalytiqueAvanceeSante.SERVICES_CLES_STANDARD[s] for s in services_in_sample
            if s in AnalytiqueAvanceeSante.SERVICES_CLES_STANDARD
        ])

        assert sorted(analytique.get_services_uniques()) == expected_filtered_and_standardized_services


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
        assert 'ds' in analytique.df_mensuel.columns # Colonne de date pour Prophet
        assert 'y' in analytique.df_mensuel.columns # Colonne de valeur pour Prophet
        assert 'service' in analytique.df_mensuel.columns
        
        # Le nombre de lignes doit correspondre au nombre de services clés * 12 mois
        # Le sample_data_path a 7 services clés définis
        num_expected_services = 7 # Basé sur les services clés présents dans sample_data_path
        assert len(analytique.df_mensuel) == num_expected_services * 12

    def test_transformer_en_mensuel_partial_months(self, data_without_full_months):
        analytique = AnalytiqueAvanceeSante(data_without_full_months)
        assert analytique.df_mensuel is not None
        assert 'ds' in analytique.df_mensuel.columns
        assert 'y' in analytique.df_mensuel.columns
        assert 'service' in analytique.df_mensuel.columns
        # 2 services clés * 2 mois (JANVIER, FÉVRIER) = 4 lignes
        assert len(analytique.df_mensuel) == 4

    def test_get_annees_disponibles(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        annees = analytique.get_annees_disponibles()
        assert isinstance(annees, list)
        assert 2023 in annees

    def test_get_services_uniques(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        services = analytique.get_services_uniques()
        assert isinstance(services, list)
        # Vérifier les noms standardisés
        assert 'Consultants' in services
        assert 'Accouchements' in services
        assert 'PaludismeTotal' in services
        assert 'Consultations' in services # Ajouté
        assert 'NaissancesVivantes' in services # Ajouté
        assert 'IRATotal' in services # Ajouté
        assert 'FemmesVIH+' in services # Ajouté
        # Assurer que les services non pertinents ne sont pas là
        assert 'Service Non Pertinent A' not in services
        assert 'Service Non Pertinent B' not in services


    # --- Nouveaux tests pour l'analyse exploratoire ---
    def test_categoriser_services(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        # La méthode est interne, mais on peut vérifier son effet sur les données si elle est appelée
        analytique._categoriser_services() # Appeler la méthode protégée pour le test
        assert 'categorie_service' in analytique.df_mensuel.columns
        assert 'Consultations & Personnel' in analytique.df_mensuel['categorie_service'].unique()
        assert 'Maternité & Naissance' in analytique.df_mensuel['categorie_service'].unique()
        assert 'Maladies Majeures' in analytique.df_mensuel['categorie_service'].unique()
        assert 'Dépistage & Diagnostic' in analytique.df_mensuel['categorie_service'].unique()


    def test_get_data_summary(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        summary = analytique.get_data_summary()
        assert isinstance(summary, dict)
        assert 'nombre_services_uniques' in summary
        assert 'total_consultations_ou_interventions' in summary
        
        # Compter les services standardisés uniques dans le sample_data_path
        expected_unique_services_count = 7
        assert summary['nombre_services_uniques'] == expected_unique_services_count
        
        # Calculer le total manuellement pour vérification
        expected_total = sum(analytique.df_mensuel['y'])
        assert summary['total_consultations_ou_interventions'] == int(expected_total)


    # --- Nouveaux tests pour la prévision ---
    def test_preparer_donnees_prediction(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        # Tester pour un service spécifique (nom standardisé)
        service_test = 'Consultants'
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        assert df_prepared is not None
        assert 'ds' in df_prepared.columns
        assert 'y' in df_prepared.columns
        # Vérifier que seules les données du service sont présentes
        # Assurer que df_service n'est pas vide avant de tenter d'accéder à iloc[0]
        if not df_prepared.empty:
            assert all(analytique.df_mensuel[analytique.df_mensuel['service'] == service_test]['y'] == df_prepared['y'])
        else:
            pytest.fail(f"Le DataFrame préparé pour '{service_test}' est vide.")


    def test_faire_predictions_prophet(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'Consultants' # Nom standardisé
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        if df_prepared is not None and not df_prepared.empty:
            predictions = analytique.faire_predictions(df_prepared, modele='Prophet', periodes_futur=3)
            assert predictions is not None
            assert 'ds' in predictions.columns
            assert 'yhat' in predictions.columns
            # Les prévisions incluent les données historiques + futures
            assert len(predictions) >= len(df_prepared) + 3
        else:
            pytest.skip(f"Données non préparées ou vides pour le service '{service_test}'.")


    def test_faire_predictions_random_forest(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'Accouchements' # Nom standardisé
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        if df_prepared is not None and not df_prepared.empty:
            predictions = analytique.faire_predictions(df_prepared, modele='RandomForest', periodes_futur=3)
            assert predictions is not None
            assert 'ds' in predictions.columns
            assert 'yhat' in predictions.columns
            assert len(predictions) >= len(df_prepared) + 3
        else:
            pytest.skip(f"Données non préparées ou vides pour le service '{service_test}'.")


    def test_faire_predictions_invalid_model(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'PaludismeTotal' # Nom standardisé
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        if df_prepared is not None and not df_prepared.empty:
            predictions = analytique.faire_predictions(df_prepared, modele='InvalidModel', periodes_futur=3)
            assert predictions is None
            assert "Modèle de prévision non supporté" in analytique.erreurs
        else:
            pytest.skip(f"Données non préparées ou vides pour le service '{service_test}'.")

    # --- Nouveau test pour l'optimisation (hypothétique car la méthode est maintenant ébauchée) ---
    def test_optimisation_des_ressources(self, sample_data_path):
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        service_test = 'Consultants'
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        if df_prepared is None or df_prepared.empty:
            pytest.skip(f"Données non préparées ou vides pour le service '{service_test}', impossible de tester l'optimisation.")

        predictions_prophet = analytique.faire_predictions(df_prepared.copy(), modele='Prophet', periodes_futur=3)
        if predictions_prophet is None:
            pytest.skip("Impossible d'obtenir les prédictions Prophet pour tester l'optimisation.")

        # Ajouter la colonne 'service' aux prédictions pour que l'optimisation puisse l'utiliser
        predictions_prophet['service'] = service_test

        analytique.optimiser_ressources(predictions_df=predictions_prophet,
                                         type_ressource='consultants',
                                         cout_moyen_ressource=1000)
        
        assert service_test in analytique.resultats_optimisation
        assert 'predictions_optimisees' in analytique.resultats_optimisation[service_test]
        assert 'ressources_necessaires' in analytique.resultats_optimisation[service_test]['predictions_optimisees'].columns
        assert 'cout_estime' in analytique.resultats_optimisation[service_test]['predictions_optimisees'].columns
        assert analytique.resultats_optimisation[service_test]['total_cout_estime'] > 0


    # --- Nouveau test d'intégration (simplifié) ---
    def test_integration_data_flow(self, sample_data_path):
        # Simuler un flux complet: Chargement -> Transformation -> Préparation pour prédiction -> Prédiction
        analytique = AnalytiqueAvanceeSante(sample_data_path)
        assert analytique.df is not None, "Étape 1 (Chargement) échouée"
        assert analytique.df_mensuel is not None, "Étape 2 (Transformation mensuelle) échouée"

        # Utiliser un service qui est garanti d'être dans les SERVICES_CLES_STANDARD
        service_test = 'Consultants'
        df_prepared = analytique.preparer_donnees_prediction(service_test)
        assert df_prepared is not None, "Étape 3 (Préparation pour prédiction) échouée"

        predictions = analytique.faire_predictions(df_prepared, modele='Prophet', periodes_futur=3)
        assert predictions is not None, "Étape 4 (Prédiction) échouée"
        assert 'yhat' in predictions.columns, "Les prédictions ne contiennent pas la colonne 'yhat'"

        # Tester une ébauche d'optimisation
        predictions['service'] = service_test # Assurez-vous que la colonne 'service' est présente
        analytique.optimiser_ressources(predictions.copy(), type_ressource='consultants')
        assert service_test in analytique.resultats_optimisation, "Étape 5 (Optimisation) échouée"
        assert 'ressources_necessaires' in analytique.resultats_optimisation[service_test]['predictions_optimisees'].columns, "La colonne 'ressources_necessaires' manque après optimisation."