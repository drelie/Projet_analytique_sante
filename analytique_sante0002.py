"""
Analytique d'Optimisation des Ressources de Santé
Version corrigée et optimisée
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning et Statistiques
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr

# Séries temporelles
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Visualisation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Utilitaires
import re
import os
import hashlib
import pickle
from functools import lru_cache
from typing import Optional, Union, Dict, List

class AnalytiqueAvanceeSante:
    """Analyse avancée des données de santé pour l'optimisation des ressources."""
    
    # Dictionnaire de standardisation des noms de services
    SERVICE_STANDARD = {
        r"Nombre de consultants(?!.*<)": "Nb Consultants",
        r"Nombre de consultants.*< *5": "Nb Consultants <5 ans",
        r"Nombre de consultations(?!.*<)": "Nb Consultations",
        r"Nombre de consultations.*< *5": "Nb Consultations <5 ans",
        r"Accouchement.*établissement": "Accouchements",
        r"Naissances vivantes": "Naissances vivantes",
        r"TOTAL PALUDISME": "Paludisme",
        r"TOTAL IRA": "Infections Respiratoires",
        r"TOTAL DIARRHEES": "Diarrhées",
        r"clients dépistés TOTAL": "Dépistage Total",
        r"Femme.*dépistée VIH": "Femmes VIH+",
        r"TOTAL CONSULTATION PF": "Consultations PF",
        r"TDR positifs": "TDR Paludisme Positifs",
        r"TOTAUX MORBIDITE": "Morbidité Totale",
        r"DECES": "Décès",
        r"Cas référés": "Référés",
        r"Femmes.*vaccinées.*VAT": "Femmes Vaccinées VAT"
    }
    
    SERVICES_CLES = list(SERVICE_STANDARD.values())

    def __init__(self, chemin_fichier_csv: Union[str, os.PathLike], use_dask: bool = False):
        """Initialiser avec les données de santé."""
        self.chemin_fichier_csv = str(chemin_fichier_csv)
        self.use_dask = use_dask
        self._dask_client = None
        self.donnees = None
        self.donnees_mensuelles = None
        self.modeles_prevision = {}
        self.resultats_optimisation = {}
        self.annee_donnees = None

        self._charger_donnees()
        self._preparer_donnees()

    def _standardiser_nom_service(self, nom: str) -> str:
        """Standardise le nom du service selon le mapping défini."""
        nom = str(nom).strip()
        for pattern, standard in self.SERVICE_STANDARD.items():
            if re.search(pattern, nom, re.IGNORECASE):
                return standard
        return nom

    def _charger_donnees(self):
        """Charge les données en mémoire avec optimisation automatique."""
        if not os.path.exists(self.chemin_fichier_csv):
            raise FileNotFoundError(f"Fichier '{self.chemin_fichier_csv}' introuvable.")

        file_size = os.path.getsize(self.chemin_fichier_csv) / (1024 * 1024)  # Taille en MB
        
        if file_size > 100 or self.use_dask:
            self.use_dask = True
            self._init_dask_cluster()

        try:
            if self.use_dask:
                print("Chargement avec Dask (traitement parallèle)...")
                with ProgressBar():
                    self.donnees = dd.read_csv(
                        self.chemin_fichier_csv,
                        dtype={'service': 'object'},
                        assume_missing=True
                    ).compute()
            else:
                self.donnees = pd.read_csv(
                    self.chemin_fichier_csv,
                    dtype={'service': 'object'},
                    engine='c',
                    low_memory=False
                )
                
            if self.donnees.empty:
                raise ValueError("Le fichier ne contient aucune donnée valide.")
                
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement: {str(e)}")

    def _init_dask_cluster(self):
        """Initialise un cluster Dask local pour le traitement parallèle."""
        from dask.distributed import Client
        self._dask_client = Client(
            n_workers=min(4, os.cpu_count()),
            threads_per_worker=2,
            memory_limit='4GB'
        )
        print(f"Cluster Dask initialisé: {self._dask_client}")

    def _preparer_donnees(self):
        """Prépare et optimise les données chargées."""
        match = re.search(r'(\d{4})', os.path.basename(self.chemin_fichier_csv))
        self.annee_donnees = int(match.group(1)) if match else datetime.now().year

        # Standardiser les noms de services
        self.donnees['service'] = self.donnees['service'].apply(self._standardiser_nom_service)
        
        self._optimiser_memoire()
        self.donnees_mensuelles = self._transformer_en_mensuel()

    def _optimiser_memoire(self):
        """Optimise l'utilisation mémoire des données."""
        # Filtrer uniquement les services clés
        self.donnees = self.donnees[self.donnees['service'].isin(self.SERVICES_CLES)]
        
        for col in self.donnees.select_dtypes(include=['float64']).columns:
            self.donnees[col] = pd.to_numeric(self.donnees[col], downcast='float')
        
        for col in self.donnees.select_dtypes(include=['int64']).columns:
            self.donnees[col] = pd.to_numeric(self.donnees[col], downcast='integer')
            
        for col in self.donnees.select_dtypes(include=['object']).columns:
            if self.donnees[col].nunique() / len(self.donnees[col]) < 0.5:
                self.donnees[col] = self.donnees[col].astype('category')

    def _transformer_en_mensuel(self) -> pd.DataFrame:
        """Transforme les données brutes en format mensuel optimisé avec gestion robuste des noms de colonnes."""
        if self.donnees is None or self.donnees.empty:
            return pd.DataFrame()

        # Dictionnaire de variantes pour chaque mois
        mois_variantes = {
            'JANVIER': ['JAN', 'JANVIER', 'JANUARY', 'JANV'],
            'FEVRIER': ['FEV', 'FEVRIER', 'FEBRUARY', 'FÉVRIER', 'FEB'],
            'MARS': ['MARS', 'MARCH', 'MAR'],
            'AVRIL': ['AVR', 'AVRIL', 'APRIL', 'APR'],
            'MAI': ['MAI', 'MAY'],
            'JUIN': ['JUIN', 'JUNE', 'JUN'],
            'JUILLET': ['JUL', 'JUILLET', 'JULY'],
            'AOUT': ['AOUT', 'AOÛT', 'AUGUST', 'AUG'],
            'SEPTEMBRE': ['SEPT', 'SEPTEMBRE', 'SEPTEMBER', 'SEP'],
            'OCTOBRE': ['OCT', 'OCTOBRE', 'OCTOBER'],
            'NOVEMBRE': ['NOV', 'NOVEMBRE', 'NOVEMBER'],
            'DECEMBRE': ['DEC', 'DECEMBRE', 'DECEMBER', 'DÉCEMBRE']
        }

        # Trouver les colonnes correspondantes dans le dataset
        colonnes_trouvees = {}
        for mois_std, variantes in mois_variantes.items():
            for col in self.donnees.columns:
                col_norm = str(col).strip().upper()
                if col_norm in variantes:
                    colonnes_trouvees[mois_std] = col
                    break

        if not colonnes_trouvees:
            print("Aucune colonne mensuelle trouvée - utilisation des données brutes.")
            return self.donnees

        # Préparer le format long
        donnees_long = pd.melt(
            self.donnees,
            id_vars=['service'],
            value_vars=list(colonnes_trouvees.values()),
            var_name='mois_brut',
            value_name='valeur'
        )
        
        # Supprimer uniquement les lignes où valeur OU date est manquante
        donnees_long = donnees_long.dropna(subset=['valeur'])
        
        # Créer un mapping inverse pour la normalisation
        mapping_normalisation = {v: k for k, v in colonnes_trouvees.items()}
        donnees_long['mois'] = donnees_long['mois_brut'].map(mapping_normalisation)

        # Mapping mois -> numéro
        mois_to_num = {m: i+1 for i, m in enumerate(mois_variantes.keys())}
        donnees_long['mois_num'] = donnees_long['mois'].map(mois_to_num)

        # Créer les dates
        donnees_long['date'] = pd.to_datetime(
            donnees_long['mois_num'].apply(
                lambda m: f"{self.annee_donnees}-{m}-01"
            ), errors='coerce'
        )
        
        # Supprimer les lignes avec date invalide
        donnees_long = donnees_long.dropna(subset=['date'])

        return donnees_long[['service', 'date', 'valeur', 'mois']]

    def analyse_exploratoire_donnees(self) -> Dict:
        """Réalise une analyse exploratoire des données."""
        if self.donnees_mensuelles is None or self.donnees_mensuelles.empty:
            return {}

        stats = {
            'nombre_services': len(self.SERVICES_CLES),
            'periode_couverte': f"{self.donnees_mensuelles['date'].min().date()} to {self.donnees_mensuelles['date'].max().date()}",
            'valeurs_manquantes': self.donnees.isna().sum().to_dict()
        }

        # Statistiques descriptives par service
        stats_descriptives = {}
        for service in self.SERVICES_CLES:
            service_data = self.donnees_mensuelles[
                (self.donnees_mensuelles['service'] == service) & 
                (self.donnees_mensuelles['valeur'].notna())
            ]
            if not service_data.empty:
                stats_descriptives[service] = {
                    'moyenne': service_data['valeur'].mean(),
                    'mediane': service_data['valeur'].median(),
                    'ecart_type': service_data['valeur'].std(),
                    'total': service_data['valeur'].sum()
                }

        stats['statistiques_services'] = stats_descriptives
        return stats

    def prevision_demande(self, periodes_prevision: int = 12, service_cible: Optional[str] = None) -> Dict:
        """Prévoit la demande pour les services clés."""
        if self.donnees_mensuelles is None or self.donnees_mensuelles.empty:
            return {}

        # Filtrer uniquement les services clés
        services = [service_cible] if service_cible else self.SERVICES_CLES
        
        print(f"⏳ Prévision pour {len(services)} services clés")
        
        for service in services:
            service_data = self.donnees_mensuelles[
                (self.donnees_mensuelles['service'] == service) & 
                (self.donnees_mensuelles['valeur'].notna())
            ]
            
            if len(service_data) < 6:  # Pas assez de données
                print(f"⚠️ Pas assez de données pour {service} ({len(service_data)} points)")
                continue
                
            # Vérifier si les données sont constantes
            if service_data['valeur'].nunique() == 1:
                print(f"⚠️ Données constantes pour {service} - utilisation de la dernière valeur")
                last_value = service_data['valeur'].iloc[-1]
                forecast_dates = pd.date_range(
                    start=service_data['date'].iloc[-1] + pd.DateOffset(months=1),
                    periods=periodes_prevision,
                    freq='MS'
                )
                forecast = pd.DataFrame({
                    'ds': forecast_dates,
                    'yhat': [last_value] * periodes_prevision,
                    'yhat_lower': [last_value] * periodes_prevision,
                    'yhat_upper': [last_value] * periodes_prevision
                })
                
                # Métriques pour données constantes
                mae = 0.0
                r2 = 1.0
            else:
                try:
                    # Préparation des données pour Prophet
                    df = service_data[['date', 'valeur']].rename(columns={'date': 'ds', 'valeur': 'y'})
                    
                    model = Prophet(
                        yearly_seasonality=True,
                        daily_seasonality=False,
                        weekly_seasonality=False,
                        seasonality_mode='multiplicative'
                    )
                    model.fit(df)
                    
                    # Génération des prévisions
                    future = model.make_future_dataframe(periods=periodes_prevision, freq='MS')
                    forecast = model.predict(future)
                    
                    # Post-processing: éliminer les valeurs négatives
                    forecast['yhat'] = forecast['yhat'].clip(lower=0)
                    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                    
                    # Calcul des métriques
                    evaluation = forecast.set_index('ds')[['yhat']].join(df.set_index('ds'))
                    evaluation = evaluation.dropna()
                    
                    if len(evaluation) > 2:
                        mae = mean_absolute_error(evaluation['y'], evaluation['yhat'])
                        r2 = r2_score(evaluation['y'], evaluation['yhat'])
                    else:
                        mae = r2 = np.nan
                except Exception as e:
                    print(f"❌ Erreur lors de la prévision pour {service}: {str(e)}")
                    continue
            
            cache_key = f"model_{service}_{periodes_prevision}"
            self.modeles_prevision[cache_key] = {
                'model': model if 'model' in locals() else None,
                'forecast': forecast,
                'metrics': {
                    'mae': mae,
                    'r2': r2
                }
            }
            print(f"✅ Prévision terminée pour {service}")

        return self.modeles_prevision

    def optimisation_ressources(self) -> pd.DataFrame:
        """Génère des recommandations d'optimisation des ressources."""
        if not self.modeles_prevision:
            self.prevision_demande()

        recommandations = []
        for service, data in self.modeles_prevision.items():
            service_name = service.replace('model_', '').split('_')[0]
            forecast = data['forecast']
            last_actual = data['metrics']['mae'] if not np.isnan(data['metrics']['mae']) else 0
            
            # Calcul de la dernière valeur réelle
            last_value = None
            if self.donnees_mensuelles is not None:
                service_data = self.donnees_mensuelles[
                    (self.donnees_mensuelles['service'] == service_name) & 
                    (self.donnees_mensuelles['valeur'].notna())
                ]
                if not service_data.empty:
                    last_value = service_data['valeur'].iloc[-1]
            
            # Calcul des ressources nécessaires
            derniere_prevision = forecast['yhat'].iloc[-1] if not forecast.empty else 0
            ressources_necessaires = max(0, derniere_prevision - (last_value or 0))
            
            # Période correcte
            periode = forecast['ds'].iloc[-1].strftime('%Y-%m') if not forecast.empty else "N/A"
            
            recommandations.append({
                'Service': service_name,
                'DemandePrévue': derniere_prevision,
                'RessourcesNécessaires': ressources_necessaires,
                'Confiance': data['metrics']['r2'],
                'Période': periode
            })

        self.resultats_optimisation = pd.DataFrame(recommandations)
        return self.resultats_optimisation

    def generer_rapport(self, format: str = 'excel') -> str:
        """Génère un rapport consolidé des analyses."""
        if format == 'excel':
            output_path = f"rapport_optimisation_{self.annee_donnees}.xlsx"
            with pd.ExcelWriter(output_path) as writer:
                # Résumé des données
                summary = pd.DataFrame({
                    'Métrique': ['Services uniques', 'Période couverte', 'Données manquantes'],
                    'Valeur': [
                        len(self.SERVICES_CLES),
                        f"{self.donnees_mensuelles['date'].min().date()} to {self.donnees_mensuelles['date'].max().date()}",
                        self.donnees.isna().sum().sum()
                    ]
                })
                summary.to_excel(writer, sheet_name='Résumé', index=False)
                
                # Prévisions
                if self.modeles_prevision:
                    forecasts = pd.concat([
                        pd.DataFrame({
                            'Service': k.replace('model_', '').split('_')[0],
                            'Date': v['forecast']['ds'],
                            'Prévision': v['forecast']['yhat'],
                            'Intervalle inférieur': v['forecast']['yhat_lower'],
                            'Intervalle supérieur': v['forecast']['yhat_upper']
                        }) for k, v in self.modeles_prevision.items()
                    ])
                    forecasts.to_excel(writer, sheet_name='Prévisions', index=False)
                
                # Optimisation
                if not self.resultats_optimisation.empty:
                    self.resultats_optimisation.to_excel(writer, sheet_name='Optimisation', index=False)
            
            return output_path
        else:
            raise ValueError("Format non supporté")

    def __del__(self):
        """Nettoyage des ressources."""
        if self._dask_client:
            self._dask_client.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            analyse = AnalytiqueAvanceeSante(sys.argv[1])
            
            # Analyse exploratoire
            print("\n=== Analyse exploratoire ===")
            exploratoire = analyse.analyse_exploratoire_donnees()
            print(f"Nombre de services: {exploratoire.get('nombre_services')}")
            print(f"Période couverte: {exploratoire.get('periode_couverte')}")
            
            # Prévisions
            print("\n=== Prévisions ===")
            previsions = analyse.prevision_demande()
            for service, data in previsions.items():
                print(f"{service}: MAE={data['metrics']['mae']:.2f}, R²={data['metrics']['r2']:.2f}")
            
            # Optimisation
            print("\n=== Optimisation ===")
            optimisation = analyse.optimisation_ressources()
            print(optimisation.to_string(index=False))
            
            # Rapport
            print("\n=== Génération du rapport ===")
            rapport = analyse.generer_rapport()
            print(f"Rapport généré: {rapport}")
            
        except Exception as e:
            print(f"Erreur: {str(e)}")
    else:
        print("Usage: python analytique_sante.py <chemin_fichier.csv>")