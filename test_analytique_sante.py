import pytest
import pandas as pd
from pathlib import Path
from analytique_sante import AnalytiqueAvanceeSante # Assurez-vous que le chemin est correct

# --- Fixtures pour les données de test ---

@pytest.fixture
def sample_data_path(tmp_path):
    """Crée un fichier CSV valide pour les tests de chargement réussi et de transformation."""
    content = """service,JANVIER,FEVRIER,MARS,AVRIL,MAI,JUIN,JUILLET,AOUT,SEPTEMBRE,OCTOBRE,NOVEMBRE,DECEMBRE,TOTAUX
Consultations,100,120,110,130,140,150,160,170,180,190,200,210,1960
Pathologies,20,25,22,30,32,35,38,40,42,45,48,50,427
Vaccinations,50,60,55,70,75,80,85,90,95,100,105,110,975
"""
    # Inclure l'année dans le nom du fichier pour le test de détection de l'année
    file_path = tmp_path / "test_data_valid_2023.csv"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def non_existent_data_path(tmp_path):
    """Retourne un chemin vers un fichier qui n'existe pas."""
    return tmp_path / "non_existent.csv"

@pytest.fixture
def empty_file_path(tmp_path):
    """Crée un fichier CSV vide."""
    file_path = tmp_path / "empty.csv"
    file_path.touch() # Crée un fichier vide
    return file_path

@pytest.fixture
def invalid_data_path(tmp_path):
    """Crée un fichier CSV avec des données non numériques dans les colonnes de mois."""
    content = """service,JANVIER,FEVRIER,MARS
Consultations,100,abc,110
Pathologies,20,25,def
Vaccinations,50,60,55
"""
    file_path = tmp_path / "test_data_invalid.csv"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def missing_column_data_path(tmp_path):
    """Crée un fichier CSV où une colonne essentielle (JANVIER) est manquante."""
    content = """service,FEVRIER,MARS
Consultations,120,110
Pathologies,25,22
Vaccinations,60,55
"""
    file_path = tmp_path / "test_data_missing_col.csv"
    file_path.write_text(content)
    return file_path

# --- Tests de la classe AnalytiqueAvanceeSante ---

def test_initial_data_loading_success(sample_data_path):
    """Teste le chargement réussi d'un fichier CSV valide."""
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    assert analytique.donnees is not None
    assert not analytique.donnees.empty
    assert 'service' in analytique.donnees.columns
    assert 'JANVIER' in analytique.donnees.columns
    assert analytique.annee_donnees == 2023 # L'année est maintenant 2023 grâce au nom du fichier

def test_data_loading_file_not_found(non_existent_data_path):
    """Teste la gestion des fichiers CSV introuvables."""
    with pytest.raises(FileNotFoundError, match="n'a pas été trouvé"):
        AnalytiqueAvanceeSante(non_existent_data_path)

def test_data_loading_empty_file(empty_file_path):
    """Teste la gestion d'un fichier CSV vide."""
    with pytest.raises(ValueError, match="est vide"):
        AnalytiqueAvanceeSante(empty_file_path)

def test_robust_preprocessing_numerical_conversion(invalid_data_path):
    """Teste la conversion robuste des colonnes numériques avec des données invalides."""
    analytique = AnalytiqueAvanceeSante(invalid_data_path)
    # Après la conversion, 'abc' et 'def' devraient être devenus 0
    assert analytique.donnees['FEVRIER'].iloc[0] == 0 # 'abc' converti en 0
    assert analytique.donnees['MARS'].iloc[1] == 0 # 'def' converti en 0

def test_robust_preprocessing_missing_essential_column(missing_column_data_path):
    """Teste la gestion d'une colonne essentielle manquante."""
    with pytest.raises(ValueError, match="essentielle 'JANVIER' est manquante"): # Le test s'attend à ce que 'JANVIER' soit une colonne requise
        AnalytiqueAvanceeSante(missing_column_data_path)

def test_mensual_data_transformation(sample_data_path):
    """Teste la transformation des données en format mensuel."""
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    assert analytique.donnees_mensuelles is not None
    assert not analytique.donnees_mensuelles.empty
    assert analytique.donnees_mensuelles.index.name == 'date' # Correction de l'assertion
    assert 'Consultations' in analytique.donnees_mensuelles.columns
    assert analytique.donnees_mensuelles.loc['2023-01-01', 'Consultations'] == 100

def test_forecast_output_format(sample_data_path):
    """Teste que la prévision génère un DataFrame dans le bon format."""
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    previsions = analytique.prevision_demande(periodes_prevision=1) # Prévoit un seul mois
    
    assert 'Consultations' in previsions
    forecast_df = previsions['Consultations']['forecast']
    
    assert isinstance(forecast_df, pd.DataFrame)
    assert 'ds' in forecast_df.columns # Colonne de date de Prophet
    assert 'yhat' in forecast_df.columns # Prévision de Prophet
    assert not forecast_df.empty
    assert len(forecast_df) > analytique.donnees_mensuelles.shape[0] # Doit contenir des prévisions futures