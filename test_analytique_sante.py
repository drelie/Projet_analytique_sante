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
    # Utilisation des noms standardisés
    data = {
        'Service': [
            'Nb Consultants',            # Clé
            'Accouchements',             # Clé
            'Paludisme',                 # Clé
            'Nb Consultations',          # Clé
            'Naissances vivantes',       # Clé
            'Infections Respiratoires',  # Clé
            'Femmes VIH+'                # Clé
        ],
        'JANVIER': [100, 50, 120, 60, 30, 40, 25],
        'FÉVRIER': [110, 55, 130, 65, 32, 42, 28],
        'MARS': [120, 60, 140, 70, 35, 45, 30],
        'AVRIL': [130, 65, 150, 75, 38, 48, 32],
        'MAI': [140, 70, 160, 80, 40, 50, 35],
        'JUIN': [150, 75, 170, 85, 42, 52, 38],
        'JUILLET': [160, 80, 180, 90, 45, 55, 40],
        'AOÛT': [170, 85, 190, 95, 48, 58, 42],
        'SEPTEMBRE': [180, 90, 200, 100, 50, 60, 45],
        'OCTOBRE': [190, 95, 210, 105, 52, 62, 48],
        'NOVEMBRE': [200, 100, 220, 110, 55, 65, 50],
        'DÉCEMBRE': [210, 105, 230, 115, 58, 68, 52],
    }
    df = pd.DataFrame(data)
    filepath = create_test_data_dir / "sample_data_2023.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

@pytest.fixture
def sample_data_with_missing_months_path(create_test_data_dir):
    data = {
        'Service': ['Nb Consultants', 'Infections Respiratoires'],
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
        'Service': ['Nb Consultations', 'Naissances vivantes'],
        'JANVIER': [100, 50],
        'FÉVRIER': [110, 55],
    }
    df = pd.DataFrame(data)
    filepath = create_test_data_dir / "data_without_full_months_2025.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

def test_initialization_success(sample_data_path):
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    assert analytique.donnees is not None
    assert not analytique.donnees.empty
    assert analytique.annee_donnees == 2023
    
    # Vérification des services standardisés
    expected_standard_services = sorted([
        'Nb Consultants', 'Accouchements', 'Paludisme',
        'Nb Consultations', 'Naissances vivantes', 
        'Infections Respiratoires', 'Femmes VIH+'
    ])
    # Les services dans les données chargées doivent être ceux attendus
    services_dans_donnees = sorted(analytique.donnees['service'].unique())
    assert services_dans_donnees == expected_standard_services

def test_initialization_file_not_found(non_existent_data_path):
    with pytest.raises(FileNotFoundError):
        analytique = AnalytiqueAvanceeSante(non_existent_data_path)

def test_initialization_empty_file(empty_file_path):
    with pytest.raises(ValueError):
        analytique = AnalytiqueAvanceeSante(empty_file_path)

def test_initialization_invalid_data_format(invalid_data_path):
    with pytest.raises(ValueError):
        analytique = AnalytiqueAvanceeSante(invalid_data_path)

def test_transformer_en_mensuel(sample_data_path):
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    assert analytique.donnees_mensuelles is not None
    assert 'service' in analytique.donnees_mensuelles.columns
    assert 'date' in analytique.donnees_mensuelles.columns
    assert 'valeur' in analytique.donnees_mensuelles.columns
    
    # Vérification du type de données des dates
    assert pd.api.types.is_datetime64_dtype(analytique.donnees_mensuelles['date'])
    
    # Vérification des valeurs pour Janvier
    jan_data = analytique.donnees_mensuelles[analytique.donnees_mensuelles['date'].dt.month == 1]
    # Somme des valeurs de janvier pour tous les services
    total_janvier = jan_data['valeur'].sum()
    # Attendu: 100+50+120+60+30+40+25 = 425
    assert total_janvier == 425

def test_transformer_en_mensuel_partial_months(data_without_full_months):
    analytique = AnalytiqueAvanceeSante(data_without_full_months)
    assert analytique.donnees_mensuelles is not None
    assert 'service' in analytique.donnees_mensuelles.columns
    assert 'date' in analytique.donnees_mensuelles.columns
    assert 'valeur' in analytique.donnees_mensuelles.columns
    assert len(analytique.donnees_mensuelles) == 4  # 2 services × 2 mois

def test_analyse_exploratoire_donnees(sample_data_path):
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    stats = analytique.analyse_exploratoire_donnees()
    assert stats['nombre_services'] == 7
    assert '2023-01-01 to 2023-12-01' in stats['periode_couverte']
    assert 'statistiques_services' in stats
    assert 'Nb Consultants' in stats['statistiques_services']

def test_prevision_demande(sample_data_path):
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    previsions = analytique.prevision_demande(periodes_prevision=3)
    assert len(previsions) > 0
    for service_key in previsions:
        assert 'forecast' in previsions[service_key]
        assert 'metrics' in previsions[service_key]

def test_optimisation_ressources(sample_data_path):
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    # Appel direct à optimisation_ressources, qui doit appeler prevision_demande si nécessaire
    recommandations = analytique.optimisation_ressources()
    assert not recommandations.empty
    assert 'Service' in recommandations.columns
    assert 'DemandePrévue' in recommandations.columns
    assert 'RessourcesNécessaires' in recommandations.columns

def test_generer_rapport(sample_data_path, tmp_path):
    analytique = AnalytiqueAvanceeSante(sample_data_path)
    # S'assurer que les prévisions et optimisations sont faites
    analytique.prevision_demande()
    analytique.optimisation_ressources()
    rapport_path = analytique.generer_rapport()
    assert os.path.exists(rapport_path)
    assert "rapport_optimisation_2023.xlsx" in rapport_path
    # Nettoyer
    os.remove(rapport_path)