"""
Analytique d'Optimisation des Ressources de Santé
Modélisation avancée et analyse pour la planification des ressources de santé
"""

import pandas as pd
import numpy as np
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

# Nouvelle importation pour la détection de l'année
import re
import os

class AnalytiqueAvanceeSante:
    """Analyse avancée des données de santé pour l'optimisation des ressources."""

    def __init__(self, chemin_fichier_csv):
        """Initialiser avec les données de santé.
        Args:
            chemin_fichier_csv (str ou pathlib.Path): Chemin vers le fichier CSV.
        """
        self.chemin_fichier_csv = str(chemin_fichier_csv)

        if not os.path.exists(self.chemin_fichier_csv):
            raise FileNotFoundError(f"Le fichier '{self.chemin_fichier_csv}' n'a pas été trouvé.")

        if os.path.getsize(self.chemin_fichier_csv) == 0:
            raise ValueError(f"Le fichier '{self.chemin_fichier_csv}' est vide.")

        try:
            self.donnees = pd.read_csv(self.chemin_fichier_csv)
            if self.donnees.empty:
                 raise ValueError(f"Le fichier '{self.chemin_fichier_csv}' est vide ou ne contient aucune donnée exploitable après lecture.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Le fichier '{self.chemin_fichier_csv}' est vide ou ne contient aucune donnée exploitable après lecture.")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier CSV '{self.chemin_fichier_csv}': {e}")

        # Tenter d'extraire l'année du nom de fichier
        match = re.search(r'(\d{4})', os.path.basename(self.chemin_fichier_csv))
        # Si aucune année n'est trouvée, définir une valeur par défaut (par exemple, 2023)
        self.annee_donnees = int(match.group(1)) if match else 2023 # <--- MODIFICATION ICI

        self.donnees_mensuelles = self._transformer_en_mensuel()
        self.modeles_prevision = {}
        self.resultats_optimisation = {}

    def _transformer_en_mensuel(self):
        """Transforme les données brutes en un format mensuel si possible."""
        if self.donnees is None or self.donnees.empty:
            return pd.DataFrame()

        mois_colonnes = ['JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN',
                         'JUILLET', 'AOUT', 'SEPTEMBRE', 'OCTOBRE', 'NOVEMBRE', 'DECEMBRE']
        
        # --- NOUVELLE VÉRIFICATION DES COLONNES ESSENTIELLES ---
        # Si 'service' est une colonne essentielle pour le pivotement et 'JANVIER' pour la complétude
        colonnes_essentielles_requises = ['service', 'JANVIER'] # Ajoutez toutes les colonnes requises
        for col in colonnes_essentielles_requises:
            if col not in self.donnees.columns:
                raise ValueError(f"La colonne essentielle '{col}' est manquante dans le fichier.")
        # --- FIN NOUVELLE VÉRIFICATION ---


        colonnes_presentes = [col for col in mois_colonnes if col in self.donnees.columns]

        if not colonnes_presentes:
            print("Aucune colonne mensuelle trouvée pour la transformation. Utilisation des données telles quelles.")
            if 'Annee' in self.donnees.columns and 'TOTAUX' in self.donnees.columns:
                df_temp = self.donnees.copy()
                df_temp['date'] = pd.to_datetime(df_temp['Annee'], format='%Y')
                df_temp.rename(columns={'service': 'ds', 'TOTAUX': 'y'}, inplace=True)
                return df_temp[['ds', 'y']].set_index('ds')
            return self.donnees

        for col in colonnes_presentes:
            self.donnees[col] = pd.to_numeric(self.donnees[col], errors='coerce').fillna(0)

        donnees_mensuelles = []
        for index, row in self.donnees.iterrows():
            service_name = row['service']
            annee = self.annee_donnees
            for i, month_col in enumerate(mois_colonnes):
                if month_col in row.index:
                    month_num = i + 1
                    date_str = f"{annee}-{month_num:02d}-01"
                    donnees_mensuelles.append({
                        'service': service_name,
                        'date': pd.to_datetime(date_str),
                        'valeur': row[month_col]
                    })
        df_mensuel = pd.DataFrame(donnees_mensuelles)

        df_mensuel = df_mensuel.pivot_table(index='date', columns='service', values='valeur', aggfunc='sum')
        df_mensuel.fillna(0, inplace=True)

        return df_mensuel


    def analyse_exploratoire_donnees(self):
        """Réalise une analyse exploratoire des données et génère des visualisations."""
        if self.donnees_mensuelles.empty:
            print("Pas de données mensuelles pour l'analyse exploratoire.")
            return {}

        print("📊 Réalisation de l'analyse exploratoire des données...")
        
        stats_descriptives = self.donnees_mensuelles.describe().to_dict()
        print("\nStatistiques Descriptives des Données Mensuelles:")
        for service, stats in stats_descriptives.items():
            print(f"  Service '{service}':")
            for stat_name, value in stats.items():
                print(f"    {stat_name}: {value:.2f}")

        fig = px.line(self.donnees_mensuelles.reset_index().melt(id_vars='date', var_name='Service', value_name='Valeur'),
                      x='date', y='Valeur', color='Service',
                      title='Tendances Mensuelles des Services au Fil du Temps')
        return stats_descriptives

    def prevision_demande(self, periodes_prevision=12, service_cible=None):
        """Réalise la prévision de la demande pour les services."""
        if self.donnees_mensuelles.empty:
            print("Pas de données mensuelles pour la prévision.")
            return

        print(f"🔮 Réalisation de la prévision de la demande pour {periodes_prevision} mois...")

        services_a_prevoir = [service_cible] if service_cible else self.donnees_mensuelles.columns

        for service in services_a_prevoir:
            if service not in self.donnees_mensuelles.columns:
                print(f"Service '{service}' non trouvé dans les données. Skipping.")
                continue

            print(f"  Prévision pour le service: {service}")
            
            df_prophet = self.donnees_mensuelles[[service]].reset_index()
            df_prophet.rename(columns={'date': 'ds', service: 'y'}, inplace=True)

            model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=periodes_prevision, freq='MS')
            forecast = model.predict(future)
            
            self.modeles_prevision[service] = {
                'model': model,
                'forecast': forecast,
                'data': df_prophet
            }

            df_merged = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds', how='inner')
            if not df_merged.empty:
                r2 = r2_score(df_merged['y'], df_merged['yhat'])
                print(f"    R2 Score pour {service} (sur données d'entraînement): {r2:.3f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Réel', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prévision', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Intervalle inférieur', line=dict(color='red', width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Intervalle supérieur', fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(color='red', width=0), showlegend=False))
            fig.update_layout(title_text=f'Prévisions de Demande pour {service}', xaxis_title='Date', yaxis_title='Valeur')

        return self.modeles_prevision

    def optimisation_ressources(self):
        """Détermine les recommandations d'optimisation des ressources."""
        if not self.modeles_prevision:
            print("Aucun modèle de prévision disponible. Exécutez prevision_demande d'abord.")
            return {}

        print("⚙️ Détermination des recommandations d'optimisation des ressources...")
        
        recommandations = []
        for service, data in self.modeles_prevision.items():
            forecast_df = data['forecast']
            donnees_actuelles = data['data']
            
            derniere_valeur_reelle = donnees_actuelles['y'].iloc[-1] if not donnees_actuelles.empty else 0
            demande_future_estimee = forecast_df['yhat'].iloc[-1] 
            
            seuil_efficacite_cible = 0.90
            efficacite_actuelle_simulee = np.random.uniform(0.60, 0.95) 

            ressources_supplementaires_necessaires = 0
            message_reco = "Capacité actuelle adéquate ou aucune règle d'optimisation définie."
            cout_estime = 0
            
            if efficacite_actuelle_simulee < seuil_efficacite_cible:
                if demande_future_estimee > derniere_valeur_reelle * 1.10:
                    ressources_supplementaires_necessaires = int(np.ceil((demande_future_estimee - derniere_valeur_reelle) / 100))
                    if ressources_supplementaires_necessaires > 0:
                        message_reco = f"Augmenter les ressources de {ressources_supplementaires_necessaires} unités pour répondre à la demande future croissante et améliorer l'efficacité."
                        cout_estime = ressources_supplementaires_necessaires * 500
                elif efficacite_actuelle_simulee < 0.75:
                    ressources_supplementaires_necessaires = int(np.ceil(derniere_valeur_reelle * (seuil_efficacite_cible - efficacite_actuelle_simulee) / 100))
                    if ressources_supplementaires_necessaires > 0:
                        message_reco = f"Augmenter les ressources de {ressources_supplementaires_necessaires} unités pour améliorer l'efficacité actuelle et atteindre la cible de {int(seuil_efficacite_cible*100)}%."
                        cout_estime = ressources_supplementaires_necessaires * 500

            recommandations.append({
                'Service': service,
                'Dernière Valeur Réelle': derniere_valeur_reelle,
                'Demande Future Estimée (M+1)': demande_future_estimee,
                'Efficacité Actuelle Estimée (%)': round(efficacite_actuelle_simulee * 100, 2),
                'Efficacité Cible (%)': int(seuil_efficacite_cible * 100),
                'Ressources Supplémentaires Recommandées': ressources_supplementaires_necessaires,
                'Coût Estimé (€)': cout_estime,
                'Recommandation': message_reco
            })
        
        self.resultats_optimisation = pd.DataFrame(recommandations)
        print("\nRecommandations d'Optimisation Générées:")
        print(self.resultats_optimisation.to_markdown(index=False))

        return self.resultats_optimisation
    
    def generer_rapport_insights(self, nom_fichier="rapport_analyse_sante.xlsx"):
        """Génère un rapport consolidé des analyses et prévisions."""
        print(f"📄 Génération du rapport d'analyse vers '{nom_fichier}'...")

        with pd.ExcelWriter(nom_fichier, engine='xlsxwriter') as writer:
            if not self.donnees_mensuelles.empty:
                self.donnees_mensuelles.to_excel(writer, sheet_name='Donnees_Mensuelles')

            for service, data in self.modeles_prevision.items():
                if not data['forecast'].empty:
                    data['forecast'].to_excel(writer, sheet_name=f'Prevision_{service.replace(" ", "_")}', index=False)
            
            if not self.resultats_optimisation.empty:
                donnees_optimisation_export = self.resultats_optimisation.copy()
                donnees_optimisation_export.to_excel(writer, sheet_name='Optimisation', index=False)
            
            metriques_df = pd.DataFrame.from_dict(self.analyse_exploratoire_donnees(), orient='index')
            if not metriques_df.empty:
                metriques_df.to_excel(writer, sheet_name='Resume_Metriques_Cles')
        
        print(f"✅ Résultats exportés vers '{nom_fichier}'")

def main():
    """Fonction d'exécution principale"""
    print("🏥 ANALYTIQUE D'OPTIMISATION DES RESSOURCES DE SANTÉ")
    print("=" * 60)
    
    import sys
    if len(sys.argv) > 1:
        chemin_fichier_csv = sys.argv[1]
    else:
        print("Veuillez spécifier le chemin du fichier CSV. Ex: python analytique_sante.py mon_fichier.csv")
        return

    try:
        analytique = AnalytiqueAvanceeSante(chemin_fichier_csv)
        
        analytique.analyse_exploratoire_donnees()
        analytique.prevision_demande()
        analytique.optimisation_ressources()
        analytique.generer_rapport_insights()
    except (FileNotFoundError, ValueError) as e:
        print(f"Erreur d'initialisation : {e}")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")

if __name__ == "__main__":
    main()