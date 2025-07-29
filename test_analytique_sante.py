import pandas as pd
import numpy as np
import pytest
import os
import sys

# Ajouter le répertoire parent au PYTHONPATH pour que les modules puissent être trouvés
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analytique_sante import AnalytiqueAvanceeSante # Assurez-vous que le chemin est correct

# Fixture pour créer un fichier CSV temporaire propre pour les tests
@pytest.fixture
def sample_data_path(tmp_path):
    """Crée un fichier CSV temporaire avec des données valides pour les tests."""
    data = {
        'service': ['Consultations', 'Vaccinations', 'Pathologies'],
        'JANVIER': [100, 50, 20],
        'FEVRIER': [120, 60, 25],
        'MARS': [110, 55, 22],
        'T1': [330, 165, 67], # Totaux trimestriels
        'AVRIL': [130, 70, 30],
        'Année': [2023, 2023, 2023] # Ajout d'une colonne année pour le test
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data_valid.csv"
    df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def invalid_data_path(tmp_path):
    """Crée un fichier CSV temporaire avec des données invalides (non numériques) pour les tests."""
    data = {
        'service': ['Consultations', 'Vaccinations'],
        'JANVIER': [100, 'abc'], # Valeur invalide
        'FEVRIER': [120, 60],
        'MARS': ['xyz', 55], # Valeur invalide
        'TOTAUX': [220, 115]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data_invalid.csv"
    df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def missing_column_data_path(tmp_path):
    """Crée un fichier CSV temporaire avec une colonne essentielle manquante."""
    data = {
        'service': ['Consultations', 'Vaccinations'],
        'FEVRIER': [120, 60],
        'MARS': [110, 55],
        'TOTAUX': [230, 115]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data_missing_col.csv"
    df.to_csv(file_path, index=False)
    return file_path

# --- Tests de Chargement et Pré-traitement des Données ---

def test_initial_data_loading_success(sample_data_path):
    """Teste le chargement réussi d'un fichier CSV valide."""
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    assert analytique.donnees is not None
    assert not analytique.donnees.empty
    assert 'service' in analytique.donnees.columns
    assert 'JANVIER' in analytique.donnees.columns
    assert analytique.annee_donnees == 2023

def test_data_loading_file_not_found():
    """Teste la gestion d'un fichier non trouvé."""
    with pytest.raises(FileNotFoundError):
        AnalytiqueAvanceeSante("non_existent_file.csv")

def test_data_loading_empty_file(tmp_path):
    """Teste la gestion d'un fichier CSV vide."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch() # Crée un fichier vide
    with pytest.raises(ValueError, match="est vide"):
        AnalytiqueAvanceeSante(empty_file)

def test_robust_preprocessing_numerical_conversion(invalid_data_path):
    """Teste la conversion robuste des colonnes numériques avec des données invalides."""
    analytique = AnalytiqueAvanceeSante(invalid_data_path)
    # Vérifie que les valeurs non numériques sont converties en 0 (par fillna(0))
    assert analytique.donnees['JANVIER'].iloc[1] == 0
    assert analytique.donnees['MARS'].iloc[0] == 0
    # Vérifie que les types sont maintenant numériques
    assert pd.api.types.is_numeric_dtype(analytique.donnees['JANVIER'])
    assert pd.api.types.is_numeric_dtype(analytique.donnees['MARS'])

def test_robust_preprocessing_missing_essential_column(missing_column_data_path):
    """Teste la gestion d'une colonne essentielle manquante."""
    with pytest.raises(ValueError, match="essentielle 'JANVIER' est manquante"):
        # Le test s'attend à ce que 'JANVIER' soit une colonne requise
        AnalytiqueAvanceeSante(missing_column_data_path)

def test_mensual_data_transformation(sample_data_path):
    """Teste la transformation des données en format mensuel."""
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    assert analytique.donnees_mensuelles is not None
    assert not analytique.donnees_mensuelles.empty
    assert 'Date' in analytique.donnees_mensuelles.columns
    assert 'Mois' in analytique.donnees_mensuelles.columns
    assert 'Valeur' in analytique.donnees_mensuelles.columns
    # Vérifier le nombre total de lignes (services * mois existants)
    expected_rows = 3 * 4 # 3 services * 4 mois (Jan-Avril)
    assert len(analytique.donnees_mensuelles) == expected_rows
    
    # Vérifier une valeur spécifique après transformation
    assert analytique.donnees_mensuelles[(analytique.donnees_mensuelles['service'] == 'Consultations') & 
                                       (analytique.donnees_mensuelles['Mois'] == 'JANVIER')]['Valeur'].iloc[0] == 100

# --- Tests de Modélisation (Exemples, à compléter) ---

def test_forecast_output_format(sample_data_path):
    """Teste que la prévision génère un DataFrame dans le bon format."""
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    # Assurez-vous d'appeler la méthode de prévision réelle dans votre classe
    # Ici, nous allons simuler un appel ou tester une petite partie
    
    # Pour un test réel, vous appelleriez analytique.prevision_demande()
    # et vérifieriez le format de son output ou de self.modeles_prevision
    
    # Simulation pour l'exemple (remplacez par un vrai appel)
    # Supposez que prevision_demande met à jour self.modeles_prevision
    # Ou retourne un df de prévisions
    mock_forecast_df = pd.DataFrame({
        'ds': pd.to_datetime(['2023-01-01', '2023-02-01']),
        'yhat': [105, 125],
        'yhat_lower': [100, 120],
        'yhat_upper': [110, 130]
    })
    
    # Assurez-vous que votre méthode prevision_demande() met les résultats
    # de prévision dans un format accessible, par ex. un dictionnaire
    # self.resultats_prevision['service_name'] = forecast_df
    
    # Pour ce test simple, nous allons juste vérifier si la méthode s'exécute sans erreur
    try:
        analytique.prevision_demande()
        # Si la méthode ne lève pas d'exception, considérez le test comme réussi pour l'exécution
        # Des tests plus approfondis vérifieraient la forme des données et la plage des valeurs
        assert isinstance(analytique.modeles_prevision, dict)
        assert len(analytique.modeles_prevision) > 0 # Vérifiez que des modèles ont été entraînés
    except Exception as e:
        pytest.fail(f"La méthode prevision_demande a échoué : {e}")

# Vous ajouteriez plus de tests ici pour l'optimisation, l'EDA, etc.