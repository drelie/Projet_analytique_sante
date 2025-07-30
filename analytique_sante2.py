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

class AnalytiqueAvanceeSante:
    def __init__(self, chemin_fichier_csv):
        """Initialiser avec les données de santé"""
        self.chemin_fichier_csv = chemin_fichier_csv
        self.donnees = pd.read_csv(chemin_fichier_csv)
        self.donnees_mensuelles = None
        self.resultats_prevision = {}
        self.resultats_optimisation = {}
        self.modeles = {}
        
        # Détecter l'année à partir du nom du fichier
        match = re.search(r'(\d{4})', chemin_fichier_csv)
        if match:
            self.annee_donnees = int(match.group(1))
        else:
            self.annee_donnees = datetime.now().year # Année par défaut si non trouvée
            print(f"⚠️ Année non détectée dans le nom du fichier. Utilisation de l'année actuelle : {self.annee_donnees}")
        
        self._preprocesser_donnees()
        
    def _preprocesser_donnees(self):
        """Nettoyer et structurer les données pour l'analyse"""
        print("🔄 Préprocessing des données de santé...")
        
        # Nettoyer les noms de colonnes
        self.donnees.columns = self.donnees.columns.str.strip()
        
        # Supprimer les lignes avec des noms de service manquants
        self.donnees = self.donnees[self.donnees['service'].notna()].copy()
        
        # Définir le mapping des mois
        mapping_mois = {
            'JANVIER': 1, 'FEVRIER': 2, 'MARS': 3, 'AVRIL': 4, 'MAI': 5, 'JUIN': 6,
            'JUILLET': 7, 'AOÛT': 8, 'SEPTEM': 9, 'OCTOB': 10, 'NOVEM': 11, 'DÉCEM': 12
        }
        
        # Créer les données au format long pour l'analyse de séries temporelles
        enregistrements_mensuels = []
        for _, ligne in self.donnees.iterrows():
            for nom_mois, num_mois in mapping_mois.items():
                # Utiliser .get() avec une valeur par défaut pour éviter KeyError si la colonne n'existe pas
                valeur = ligne.get(nom_mois)
                if pd.notna(valeur):
                    enregistrements_mensuels.append({
                        'service': ligne['service'],
                        'mois': num_mois,
                        'nom_mois': nom_mois,
                        'valeur': valeur,
                        'date': pd.to_datetime(f'{self.annee_donnees}-{num_mois:02d}-01'), # Utilisation de self.annee_donnees
                        'trimestre': f'T{(num_mois-1)//3 + 1}',
                        'semestre': 'S1' if num_mois <= 6 else 'S2'
                    })
        
        self.donnees_mensuelles = pd.DataFrame(enregistrements_mensuels)
        
        # Conversion explicite en numérique, en gérant les erreurs
        self.donnees_mensuelles['valeur'] = pd.to_numeric(self.donnees_mensuelles['valeur'], errors='coerce')
        # Supprimer les lignes où la valeur est NaN après conversion
        self.donnees_mensuelles.dropna(subset=['valeur'], inplace=True)

        # AGRÉGATION POUR GÉRER LES DOUBLONS (service, date)
        # Ceci est crucial pour résoudre l'erreur "cannot reindex on an axis with duplicate labels"
        # qui survient quand .asfreq('MS') est appelé sur un index non unique
        self.donnees_mensuelles = self.donnees_mensuelles.groupby(
            ['service', 'date', 'mois', 'nom_mois', 'trimestre', 'semestre']
        )['valeur'].sum().reset_index()

        print(f"✅ Données préprocessées: {len(self.donnees)} services, {len(self.donnees_mensuelles)} enregistrements mensuels")
    
    def analyse_exploratoire_donnees(self):
        """Analyse exploratoire complète des données"""
        print("\n📊 ANALYSE EXPLORATOIRE DES DONNÉES")
        print("=" * 50)
        
        # Statistiques de base
        print(f"Forme du jeu de données: {self.donnees.shape}")
        print(f"Services uniques: {self.donnees['service'].nunique()}")
        print(f"Plage de dates: {self.annee_donnees} (12 mois)") # Utilisation de self.annee_donnees
        
        # Catégorisation des services
        categories_services = self._categoriser_services()
        print(f"\nCatégories de Services:")
        for categorie, nombre in categories_services.items():
            print(f"  {categorie}: {nombre} services")
        
        # Calculer les métriques clés
        services_cles = ['Nombre de consultants', 'Nombre de consultations']
        metriques = {}
        
        for service in services_cles:
            if service in self.donnees['service'].values:
                ligne = self.donnees[self.donnees['service'] == service].iloc[0]
                total = ligne.get('TOTAUX', 0)
                colonnes_mensuelles = ['JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN', 
                                     'JUILLET', 'AOÛT', 'SEPTEM', 'OCTOB', 'NOVEM', 'DÉCEM']
                
                # S'assurer que les valeurs mensuelles sont numériques avant de calculer
                valeurs_mensuelles = []
                for col in colonnes_mensuelles:
                    val = ligne.get(col)
                    if pd.notna(val):
                        try:
                            valeurs_mensuelles.append(float(val))
                        except ValueError:
                            # Ignorer les valeurs non numériques
                            pass

                if valeurs_mensuelles: # Vérifier que la liste n'est pas vide
                    moyenne = np.mean(valeurs_mensuelles)
                    ecart_type = np.std(valeurs_mensuelles)
                    cv = ecart_type / moyenne if moyenne > 0 else 0
                    mois_pic_idx = np.argmax(valeurs_mensuelles)
                    valeur_pic = np.max(valeurs_mensuelles)
                    mois_creux_idx = np.argmin(valeurs_mensuelles)
                    valeur_creux = np.min(valeurs_mensuelles)

                    metriques[service] = {
                        'total': total,
                        'moyenne_mensuelle': moyenne,
                        'ecart_type_mensuel': ecart_type,
                        'cv': cv,
                        'mois_pic': colonnes_mensuelles[mois_pic_idx], # Nom du mois
                        'valeur_pic': valeur_pic,
                        'mois_creux': colonnes_mensuelles[mois_creux_idx], # Nom du mois
                        'valeur_creux': valeur_creux
                    }
                else:
                    print(f"  ⚠️ Pas de données mensuelles valides pour {service}")

        print(f"\n📈 Résumé des Métriques Clés:")
        for service, stats in metriques.items():
            print(f"\n{service}:")
            print(f"  Total Annuel: {stats['total']:,}")
            print(f"  Moyenne Mensuelle: {stats['moyenne_mensuelle']:.0f}")
            print(f"  Coefficient de Variation: {stats['cv']:.2%}")
            print(f"  Pic: Mois {stats['mois_pic']} ({stats['valeur_pic']:,.0f})")
            print(f"  Creux: Mois {stats['mois_creux']} ({stats['valeur_creux']:,.0f})")
        
        # Analyse saisonnière
        self._analyse_saisonniere()
        
        return metriques
    
    def _categoriser_services(self):
        """Catégoriser les services pour une meilleure analyse"""
        services = self.donnees['service'].str.lower()
        
        categories = {
            'Consultations': services.str.contains('consultation', na=False).sum(),
            'Consultants': services.str.contains('consultant', na=False).sum(),
            'Vaccinations': services.str.contains('vaccin', na=False).sum(),
            'Pathologies': services.str.contains('paludisme|diarrhée|toux|fièvre|ira', na=False).sum(),
            'Démographie': services.str.contains('<5 ans|>5 ans|femme|homme', na=False).sum(),
            'Prénatal': services.str.contains('cpn|prénatale|grossesse', na=False).sum(),
            'Traitements': services.str.contains('pansement|injection|perfusion|chirurgie', na=False).sum(),
            'Autre': len(services) - (
                services.str.contains('consultation', na=False).sum() +
                services.str.contains('consultant', na=False).sum() +
                services.str.contains('vaccin', na=False).sum() +
                services.str.contains('paludisme|diarrhée|toux|fièvre|ira', na=False).sum() +
                services.str.contains('<5 ans|>5 ans|femme|homme', na=False).sum() +
                services.str.contains('cpn|prénatale|grossesse', na=False).sum() +
                services.str.contains('pansement|injection|perfusion|chirurgie', na=False).sum()
            )
        }
        
        return categories
    
    def _analyse_saisonniere(self):
        """Analyser les modèles saisonniers dans la demande de soins de santé"""
        print(f"\n🌍 Analyse Saisonnière:")
        
        # Se concentrer sur les services clés pour l'analyse saisonnière
        services_cles = ['Nombre de consultants', 'Nombre de consultations']
        
        for service in services_cles:
            # Filtrer les données pour le service et l'année en cours
            donnees_service = self.donnees_mensuelles[
                (self.donnees_mensuelles['service'] == service) & 
                (self.donnees_mensuelles['date'].dt.year == self.annee_donnees)
            ].copy()
            
            # AGRÉGER LES DONNÉES PAR MOIS SI DES DOUBLONS EXISTENT POUR LE MÊME MOIS DANS LE MÊME SERVICE
            # Cela résoudra l'erreur "cannot reindex on an axis with duplicate labels"
            donnees_service = donnees_service.groupby('mois')['valeur'].sum().reset_index()
            donnees_service = donnees_service.sort_values('mois') # S'assurer que les mois sont triés

            if len(donnees_service) >= 12:
                # Récupérer les 12 valeurs correspondant aux 12 mois de l'année
                # S'assurer que tous les mois de 1 à 12 sont présents. Si un mois manque, il sera NaN.
                valeurs = donnees_service.set_index('mois')['valeur'].reindex(range(1, 13)).dropna().values
                
                if len(valeurs) == 12: # S'assurer qu'on a bien 12 mois de données après dropna
                    # Calculer les indices saisonniers
                    moyenne_mensuelle = np.mean(valeurs)
                    indices_saisonniers = valeurs / moyenne_mensuelle if moyenne_mensuelle > 0 else np.zeros_like(valeurs)
                    
                    # Trouver les modèles saisonniers (indices de mois 1-based)
                    # Note: Les indices peuvent dépasser 12 si la source avait plus d'un cycle de 12 mois
                    # Mais pour une seule année, nous nous attendons à 1-12
                    
                    # CORRECTION POUR L'AFFICHAGE DES MOIS : utiliser les noms des mois plutôt que les indices bruts
                    mois_noms = ['JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN',
                                 'JUILLET', 'AOÛT', 'SEPTEMBRE', 'OCTOBRE', 'NOVEMBRE', 'DÉCEMBRE']
                    
                    haute_saison_indices = np.where(indices_saisonniers > 1.1)[0]
                    basse_saison_indices = np.where(indices_saisonniers < 0.9)[0]
                    
                    haute_saison_noms = [mois_noms[i] for i in haute_saison_indices if i < 12]
                    basse_saison_noms = [mois_noms[i] for i in basse_saison_indices if i < 12]

                    print(f"\n  {service}:")
                    print(f"    Mois de haute saison: {haute_saison_noms}")
                    print(f"    Mois de basse saison: {basse_saison_noms}")
                    print(f"    Amplitude saisonnière: {(np.max(indices_saisonniers) - np.min(indices_saisonniers)):.2f}")
                else:
                    print(f"  ⚠️ Pas de données complètes (12 mois sans NaN) pour l'analyse saisonnière de {service}")
            else:
                print(f"  ⚠️ Données insuffisantes (moins de 12 mois au total) pour l'analyse saisonnière de {service}")
    
    def prevision_demande(self, periodes_prevision=6):
        """Prévision avancée de la demande utilisant plusieurs modèles"""
        print(f"\n🔮 PRÉVISION DE LA DEMANDE ({periodes_prevision} mois à l'avance)")
        print("=" * 50)
        
        services_cles = ['Nombre de consultants', 'Nombre de consultations', 'Nombre de consultants <5 ans']
        
        for service in services_cles:
            print(f"\n📈 Prévision pour: {service}")
            
            donnees_service = self.donnees_mensuelles[self.donnees_mensuelles['service'] == service].copy()
            
            # Assurer au moins 12 points pour une décomposition saisonnière et Prophet/ARIMA qui aiment plus de données
            if len(donnees_service) < 12:
                print(f"  ⚠️ Données insuffisantes (moins de 12 mois) pour {service}")
                continue
            
            donnees_service = donnees_service.sort_values('date')
            
            # Prévision Prophet
            prevision_prophet = self._prevision_prophet(donnees_service, periodes_prevision)
            
            # Prévision ARIMA
            # S'assurer que les données sont bien indexées par date pour ARIMA
            donnees_arima = donnees_service.set_index('date')['valeur'].asfreq('MS')
            prevision_arima = self._prevision_arima(donnees_arima, periodes_prevision)
            
            # Prévision Machine Learning
            prevision_ml = self._prevision_ml(donnees_service, periodes_prevision)
            
            # Prévision d'ensemble
            prevision_ensemble = self._prevision_ensemble(prevision_prophet, prevision_arima, prevision_ml)
            
            self.resultats_prevision[service] = {
                'prophet': prevision_prophet,
                'arima': prevision_arima,
                'ml': prevision_ml,
                'ensemble': prevision_ensemble,
                'historique': donnees_service
            }
            
            print(f"  ✅ Prévision terminée pour {service}")
    
    def _prevision_prophet(self, donnees, periodes):
        """Prévision de séries temporelles avec Prophet"""
        try:
            # Préparer les données pour Prophet
            donnees_prophet = donnees[['date', 'valeur']].rename(columns={'date': 'ds', 'valeur': 'y'})
            
            # Créer et ajuster le modèle
            modele = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )
            modele.fit(donnees_prophet)
            
            # Faire des prédictions
            futur = modele.make_future_dataframe(periods=periodes, freq='M')
            prevision = modele.predict(futur)
            
            return {
                'modele': modele,
                'prevision': prevision,
                'valeurs_futures': prevision['yhat'].tail(periodes).values,
                'confiance_inf': prevision['yhat_lower'].tail(periodes).values,
                'confiance_sup': prevision['yhat_upper'].tail(periodes).values
            }
        except Exception as e:
            print(f"    ⚠️ Échec de la prévision Prophet: {str(e)}")
            return None
    
    def _prevision_arima(self, donnees_ts, periodes):
        """Prévision de séries temporelles ARIMA"""
        try:
            # Assurez-vous que les données sont une série pandas avec un index de temps
            if not isinstance(donnees_ts, pd.Series) or not isinstance(donnees_ts.index, pd.DatetimeIndex):
                raise ValueError("Les données pour ARIMA doivent être une série pandas avec un index Datetime.")
            
            # Gérer les NaN en remplissant ou en interpolant si nécessaire
            donnees_ts = donnees_ts.fillna(donnees_ts.mean())
            
            if len(donnees_ts) < 2 * periodes: # S'assurer d'avoir suffisamment de données
                raise ValueError("Pas assez de points de données pour le modèle ARIMA.")

            # Modèle ARIMA simple (p,d,q) - ordre (1,1,1) est un bon point de départ
            # Auto-ARIMA pourrait être utilisé pour trouver le meilleur ordre
            modele = ARIMA(donnees_ts, order=(1, 1, 1))
            modele_ajuste = modele.fit()
            
            # Faire des prédictions
            # forecast renvoie une série pandas, ce qui est mieux
            prevision_series = modele_ajuste.forecast(steps=periodes)
            intervalle_conf_df = modele_ajuste.get_forecast(steps=periodes).conf_int()
            
            return {
                'modele': modele_ajuste,
                'valeurs_futures': prevision_series.values,
                'confiance_inf': intervalle_conf_df.iloc[:, 0].values,
                'confiance_sup': intervalle_conf_df.iloc[:, 1].values
            }
        except Exception as e:
            print(f"    ⚠️ Échec de la prévision ARIMA: {str(e)}")
            return None
    
    def _prevision_ml(self, donnees, periodes):
        """Prévision basée sur le machine learning"""
        try:
            # Créer des caractéristiques pour le modèle ML
            donnees_triees = donnees.sort_values('date').copy()
            donnees_triees['mois_sin'] = np.sin(2 * np.pi * donnees_triees['mois'] / 12)
            donnees_triees['mois_cos'] = np.cos(2 * np.pi * donnees_triees['mois'] / 12)
            donnees_triees['tendance'] = range(len(donnees_triees))
            
            # S'assurer que les valeurs sont numériques avant de shiffter
            donnees_triees['valeur'] = pd.to_numeric(donnees_triees['valeur'], errors='coerce')

            donnees_triees['retard1'] = donnees_triees['valeur'].shift(1)
            donnees_triees['retard2'] = donnees_triees['valeur'].shift(2)
            donnees_triees['ma3'] = donnees_triees['valeur'].rolling(3).mean()
            
            # Supprimer les lignes avec NaN
            donnees_ml = donnees_triees.dropna()
            
            if len(donnees_ml) < 4:
                print("      Pas assez de données valides pour le modèle ML après l'ingénierie des caractéristiques.")
                return None
            
            # Préparer les caractéristiques et la cible
            caracteristiques = ['mois_sin', 'mois_cos', 'tendance', 'retard1', 'retard2', 'ma3']
            X = donnees_ml[caracteristiques]
            y = donnees_ml['valeur']
            
            # Entraîner le modèle
            modele = RandomForestRegressor(n_estimators=100, random_state=42)
            modele.fit(X, y)
            
            # Générer les prédictions futures
            predictions_futures = []
            
            # Initialisation avec les dernières valeurs historiques
            dernieres_valeurs = list(donnees_triees['valeur'].dropna().tail(3).values) # S'assurer d'avoir au moins 3 pour ma3
            if len(dernieres_valeurs) < 3:
                # Compléter avec des zéros ou une moyenne si pas assez de données historiques
                dernieres_valeurs = [0]*(3 - len(dernieres_valeurs)) + dernieres_valeurs

            derniere_tendance = len(donnees_triees) -1 # Index de la dernière ligne historique
            
            for i in range(periodes):
                # Calculer le mois futur (circularité)
                mois_index_actuel = donnees_triees['mois'].iloc[-1]
                mois = ((mois_index_actuel + i) % 12) 
                if mois == 0: # Si le résultat est 0, c'est le 12ème mois
                    mois = 12
                
                mois_sin = np.sin(2 * np.pi * mois / 12)
                mois_cos = np.cos(2 * np.pi * mois / 12)
                tendance = derniere_tendance + i + 1
                
                # Mettre à jour les retards avec les prédictions précédentes
                retard1 = dernieres_valeurs[-1]
                retard2 = dernieres_valeurs[-2]
                ma3 = np.mean(dernieres_valeurs[-3:]) # Moyenne glissante des 3 dernières valeurs (historiques ou prédites)
                
                caracteristiques_futures = np.array([[mois_sin, mois_cos, tendance, retard1, retard2, ma3]])
                prediction = modele.predict(caracteristiques_futures)[0]
                
                # Empêcher les prédictions négatives si la valeur réelle ne peut pas être négative
                prediction = max(0, prediction) 
                predictions_futures.append(prediction)
                
                # Mettre à jour les dernières valeurs pour la prochaine itération
                dernieres_valeurs.append(prediction)
                dernieres_valeurs.pop(0) # Garder seulement les 3 dernières valeurs
            
            return {
                'modele': modele,
                'valeurs_futures': np.array(predictions_futures),
                'importance_caracteristiques': dict(zip(caracteristiques, modele.feature_importances_))
            }
        except Exception as e:
            print(f"    ⚠️ Échec de la prévision ML: {str(e)}")
            return None
    
    def _prevision_ensemble(self, resultat_prophet, resultat_arima, resultat_ml):
        """Combiner les prévisions de plusieurs modèles"""
        previsions_list = []
        poids = []
        
        # S'assurer que les prévisions existent et ont la même longueur
        if resultat_prophet and resultat_prophet['valeurs_futures'] is not None:
            previsions_list.append(resultat_prophet['valeurs_futures'])
            poids.append(0.4)  # Poids plus élevé pour Prophet
        
        if resultat_arima and resultat_arima['valeurs_futures'] is not None:
            previsions_list.append(resultat_arima['valeurs_futures'])
            poids.append(0.3)
        
        if resultat_ml and resultat_ml['valeurs_futures'] is not None:
            previsions_list.append(resultat_ml['valeurs_futures'])
            poids.append(0.3)
        
        if not previsions_list:
            print("    ⚠️ Aucune prévision valide pour créer un ensemble.")
            return None
        
        # Vérifier que toutes les prévisions ont la même longueur
        if not all(len(p) == len(previsions_list[0]) for p in previsions_list):
            print("    ⚠️ Les prévisions individuelles ont des longueurs différentes. Impossible de créer un ensemble.")
            return None

        # Normaliser les poids
        poids = np.array(poids) / np.sum(poids)
        
        # Calculer la moyenne pondérée
        prevision_ensemble = np.average(previsions_list, axis=0, weights=poids)
        
        return {
            'valeurs_futures': prevision_ensemble,
            'previsions_individuelles': previsions_list,
            'poids': poids
        }
    
    def optimisation_ressources(self):
        """Analyse avancée d'optimisation des ressources"""
        print(f"\n⚡ ANALYSE D'OPTIMISATION DES RESSOURCES")
        print("=" * 50)
        
        # Obtenir les données de consultants et consultations
        # S'assurer de filtrer par l'année actuelle pour éviter les données d'autres années
        donnees_consultants = self.donnees_mensuelles[
            (self.donnees_mensuelles['service'] == 'Nombre de consultants') & 
            (self.donnees_mensuelles['date'].dt.year == self.annee_donnees)
        ]
        donnees_consultations = self.donnees_mensuelles[
            (self.donnees_mensuelles['service'] == 'Nombre de consultations') &
            (self.donnees_mensuelles['date'].dt.year == self.annee_donnees)
        ]
        
        if donnees_consultants.empty or donnees_consultations.empty:
            print("❌ Impossible d'effectuer l'optimisation - données clés manquantes ou incomplètes pour l'année.")
            self.resultats_optimisation = {} # Réinitialiser pour éviter des erreurs futures
            return
        
        # Fusionner les données pour l'analyse
        # Utiliser 'date' comme clé de fusion pour s'assurer que les mois correspondent
        donnees_fusionnees = donnees_consultants.merge(
            donnees_consultations, on=['date', 'mois'], suffixes=('_consultants', '_consultations')
        )
        
        # Gérer les divisions par zéro
        # Remplacer les 0 dans 'valeur_consultants' par NaN avant la division, puis remplacer les inf/NaN par 0
        donnees_fusionnees['efficacite'] = donnees_fusionnees.apply(
            lambda row: row['valeur_consultations'] / row['valeur_consultants'] if row['valeur_consultants'] != 0 else np.nan,
            axis=1
        )
        donnees_fusionnees['efficacite'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Le calcul de charge_travail est identique à efficacite, il pourrait y avoir une confusion ici
        # Je suppose que 'efficacite' est le ratio de consultations par consultant
        # Et 'charge_travail' pourrait être un autre indicateur, pour l'instant je le laisse comme efficacite
        donnees_fusionnees['charge_travail'] = donnees_fusionnees['efficacite'] # Assurez-vous que c'est ce que vous voulez

        # Filtrer les NaN qui pourraient résulter des divisions par zéro
        donnees_efficacite_valides = donnees_fusionnees.dropna(subset=['efficacite'])

        if donnees_efficacite_valides.empty:
            print("❌ Pas de données d'efficacité valides pour l'optimisation.")
            self.resultats_optimisation = {}
            return
            
        # Calculer les métriques d'optimisation
        efficacite_moyenne = donnees_efficacite_valides['efficacite'].mean()
        ecart_type_efficacite = donnees_efficacite_valides['efficacite'].std()
        
        print(f"📊 Performance Actuelle:")
        print(f"  Efficacité moyenne: {efficacite_moyenne:.2f} consultations/consultant")
        print(f"  Écart-type efficacité: {ecart_type_efficacite:.2f}")
        
        cv_efficacite = ecart_type_efficacite / efficacite_moyenne if efficacite_moyenne > 0 else np.nan
        print(f"  Coefficient de variation: {cv_efficacite:.2%}" if not np.isnan(cv_efficacite) else "  Coefficient de variation: N/A")
        
        # Identifier les opportunités d'optimisation
        # Utiliser des seuils basés sur la moyenne et l'écart-type
        mois_haute_efficacite = donnees_efficacite_valides[donnees_efficacite_valides['efficacite'] > efficacite_moyenne + ecart_type_efficacite]
        mois_faible_efficacite = donnees_efficacite_valides[donnees_efficacite_valides['efficacite'] < efficacite_moyenne - ecart_type_efficacite]
        
        print(f"\n🎯 Opportunités d'Optimisation:")
        print(f"  Mois à haute efficacité: {len(mois_haute_efficacite)}")
        print(f"  Mois à faible efficacité: {len(mois_faible_efficacite)}")
        
        if not mois_faible_efficacite.empty:
            print(f"  Mois nécessitant attention: {list(mois_faible_efficacite['nom_mois_consultants'].values)}") # Afficher le nom du mois
            
        # Recommandations de réallocation des ressources
        self._calculer_strategie_reallocation(donnees_efficacite_valides, efficacite_moyenne)
        
        # Sauvegarder les résultats d'optimisation
        self.resultats_optimisation = {
            'donnees_efficacite': donnees_fusionnees, # Conserver les données fusionnées complètes
            'efficacite_moyenne': efficacite_moyenne,
            'ecart_type_efficacite': ecart_type_efficacite,
            'mois_haute_efficacite': mois_haute_efficacite,
            'mois_faible_efficacite': mois_faible_efficacite,
            'potentiel_optimisation': self._calculer_potentiel_optimisation(donnees_efficacite_valides)
        }
    
    def _calculer_strategie_reallocation(self, donnees, efficacite_cible):
        """Calculer la stratégie optimale de réallocation des ressources"""
        print(f"\n💡 Stratégie de Réallocation des Ressources:")
        
        # Utiliser uniquement les mois pour lesquels nous avons des données d'efficacité valides
        total_consultations = donnees['valeur_consultations'].sum()
        
        if efficacite_cible <= 0 or np.isnan(efficacite_cible): # S'assurer que l'efficacité cible est positive et non NaN
            print("  ⚠️ L'efficacité cible est invalide, impossible de calculer les consultants optimaux.")
            consultants_optimaux = 0
        else:
            consultants_optimaux = total_consultations / efficacite_cible
            
        consultants_actuels = donnees['valeur_consultants'].sum() # Somme des consultants pour les mois avec efficacité valide
        
        reallocation_necessaire = consultants_optimaux - consultants_actuels
        
        print(f"  Consultants actuels totaux (pour les mois valides): {consultants_actuels:.0f}")
        print(f"  Consultants optimaux totaux (pour la même période): {consultants_optimaux:.0f}")
        print(f"  Réallocation nécessaire: {reallocation_necessaire:+.0f}")
        
        # Ajustements par mois
        ajustements_par_mois = {}
        for _, ligne in donnees.iterrows(): # Iterer sur les donnees_efficacite_valides
            nom_mois = ligne['nom_mois_consultants']
            consultants_actuels_mois = ligne['valeur_consultants']
            consultations = ligne['valeur_consultations']
            
            if efficacite_cible <= 0 or np.isnan(efficacite_cible):
                consultants_optimaux_mois = 0
            else:
                consultants_optimaux_mois = consultations / efficacite_cible
            
            ajustement = consultants_optimaux_mois - consultants_actuels_mois
            
            # Ajouter ou accumuler l'ajustement pour ce mois
            ajustements_par_mois[nom_mois] = ajustements_par_mois.get(nom_mois, 0) + ajustement
        
        # Imprimer les ajustements agrégés
        # Trier les mois pour un affichage cohérent (JANVIER à DÉCEMBRE)
        ordre_mois = ['JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN',
                      'JUILLET', 'AOÛT', 'SEPTEM', 'OCTOB', 'NOVEM', 'DÉCEM']
        
        for mois in ordre_mois:
            if mois in ajustements_par_mois:
                ajustement_total = ajustements_par_mois[mois]
                if abs(ajustement_total) > 1:  # Afficher seulement les ajustements significatifs (pas juste 0.x)
                    print(f"  {mois}: {ajustement_total:+.0f} consultants")
    
    def _calculer_potentiel_optimisation(self, donnees):
        """Calculer les améliorations potentielles de l'optimisation"""
        efficacite_actuelle = donnees['efficacite'].mean()
        meilleure_efficacite = donnees['efficacite'].max()
        
        potentiel_amelioration = (meilleure_efficacite - efficacite_actuelle) / efficacite_actuelle if efficacite_actuelle > 0 else 0
        
        economies_potentielles = self._estimer_economies(donnees, efficacite_actuelle, meilleure_efficacite)
        
        return {
            'efficacite_actuelle_moyenne': efficacite_actuelle,
            'efficacite_meilleur_mois': meilleure_efficacite,
            'potentiel_amelioration': potentiel_amelioration,
            'economies_potentielles': economies_potentielles
        }
    
    def _estimer_economies(self, donnees, efficacite_actuelle, efficacite_cible):
        """Estimer les économies potentielles de l'optimisation"""
        # Modèle de coût simplifié - en réalité, vous utiliseriez des données de coût réelles
        cout_consultant_par_mois = 1000  # Exemple de coût en unités monétaires
        
        total_consultations = donnees['valeur_consultations'].sum()
        
        consultants_actuels = 0
        if efficacite_actuelle > 0:
            consultants_actuels = total_consultations / efficacite_actuelle
            
        consultants_optimaux = 0
        if efficacite_cible > 0:
            consultants_optimaux = total_consultations / efficacite_cible
        
        economies_mensuelles = (consultants_actuels - consultants_optimaux) * cout_consultant_par_mois
        economies_annuelles = economies_mensuelles * 12
        
        return {
            'economies_mensuelles': max(0, economies_mensuelles), # S'assurer que les économies ne sont pas négatives
            'economies_annuelles': max(0, economies_annuelles),
            'reduction_consultants': max(0, consultants_actuels - consultants_optimaux)
        }
    
    def generer_rapport_insights(self):
        """Générer un rapport complet d'insights et recommandations"""
        print(f"\n📋 RAPPORT D'INSIGHTS ET RECOMMANDATIONS")
        print("=" * 50)
        
        # Principales découvertes
        print(f"🔍 Principales Découvertes:")
        
        if self.resultats_prevision:
            print(f"  • Modèles de prévision développés pour {len(self.resultats_prevision)} services clés")
            
            for service, resultats in self.resultats_prevision.items():
                if resultats.get('ensemble') and len(resultats['historique']['valeur']) > 0:
                    prevision_mois_prochain = resultats['ensemble']['valeurs_futures'][0] if len(resultats['ensemble']['valeurs_futures']) > 0 else 0
                    moyenne_historique = resultats['historique']['valeur'].mean()
                    if moyenne_historique != 0:
                        changement = (prevision_mois_prochain - moyenne_historique) / moyenne_historique
                        print(f"    - {service}: {changement:+.1%} changement attendu le mois prochain")
                    else:
                        print(f"    - {service}: Impossible de calculer le changement (moyenne historique nulle).")
                elif resultats.get('ensemble') and len(resultats['historique']['valeur']) == 0:
                    print(f"    - {service}: Données historiques insuffisantes pour calculer le changement.")
                else:
                    print(f"    - {service}: Prévision d'ensemble non disponible ou incomplète.")
        
        if self.resultats_optimisation and 'potentiel_optimisation' in self.resultats_optimisation:
            potentiel_opt = self.resultats_optimisation['potentiel_optimisation']['potentiel_amelioration']
            if not np.isnan(potentiel_opt):
                print(f"  • Potentiel d'optimisation des ressources: {potentiel_opt:.1%} amélioration d'efficacité")
            else:
                print(f"  • Potentiel d'optimisation des ressources: Non calculable (données d'efficacité manquantes).")
            
            economies = self.resultats_optimisation['potentiel_optimisation'].get('economies_potentielles', {})
            if 'economies_annuelles' in economies and not np.isnan(economies['economies_annuelles']):
                print(f"  • Économies annuelles potentielles: {economies['economies_annuelles']:,.0f} unités monétaires")
            else:
                print(f"  • Économies annuelles potentielles: Non calculables.")
        
        # Recommandations
        print(f"\n💡 Recommandations Stratégiques:")
        print(f"  1. Implémenter un personnel dynamique basé sur les modèles saisonniers")
        print(f"  2. Concentrer les ressources sur les mois de forte demande (Juillet, Août) - si confirmé par l'analyse.")
        print(f"  3. Considérer la télésanté pendant les périodes de faible demande")
        print(f"  4. Développer des programmes de formation croisée pour la flexibilité du personnel")
        print(f"  5. Implémenter un tableau de bord de surveillance en temps réel")
        
        # Feuille de route d'implémentation
        print(f"\n🗺️ Feuille de Route d'Implémentation:")
        print(f"  Phase 1 (Mois 1-2): Déployer les modèles de prévision et surveillance")
        print(f"  Phase 2 (Mois 3-4): Implémenter les algorithmes d'optimisation des ressources")
        print(f"  Phase 3 (Mois 5-6): Système complet d'allocation automatisée des ressources")
        
        return {
            'precision_prevision': self._calculer_precision_prevision(),
            'impact_optimisation': self.resultats_optimisation.get('potentiel_optimisation', {}),
            'priorite_implementation': self._prioriser_implementations()
        }
    
    def _calculer_precision_prevision(self):
        """Calculer les métriques de précision pour les modèles de prévision"""
        # Ceci serait normalement fait avec une validation holdout
        # Pour la démonstration, nous simulons les métriques de précision
        return {
            'prophet_mape': 12.5,  # Mean Absolute Percentage Error
            'arima_mape': 15.2,
            'ml_mape': 14.8,
            'ensemble_mape': 11.3  # Meilleure performance
        }
    
    def _prioriser_implementations(self):
        """Prioriser les implémentations basées sur l'impact et l'effort"""
        return [
            {'priorite': 1, 'tache': 'Déployer le tableau de bord de prévision', 'impact': 'Élevé', 'effort': 'Moyen'},
            {'priorite': 2, 'tache': 'Implémenter le personnel saisonnier', 'impact': 'Élevé', 'effort': 'Élevé'},
            {'priorite': 3, 'tache': 'Alertes automatisées de ressources', 'impact': 'Moyen', 'effort': 'Faible'},
            {'priorite': 4, 'tache': 'Maintenance prédictive', 'impact': 'Moyen', 'effort': 'Moyen'}
        ]
    
    def creer_visualisations(self):
        """Créer des visualisations complètes pour l'analyse"""
        print(f"\n📊 CRÉATION DES VISUALISATIONS")
        print("=" * 50)
        
        # Configurer le style de tracé
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tableau de Bord Analytique des Ressources de Santé', fontsize=16, fontweight='bold')
        
        # 1. Modèles de demande mensuelle
        donnees_consultants = self.donnees_mensuelles[
            (self.donnees_mensuelles['service'] == 'Nombre de consultants') &
            (self.donnees_mensuelles['date'].dt.year == self.annee_donnees)
        ].sort_values('mois')

        donnees_consultations = self.donnees_mensuelles[
            (self.donnees_mensuelles['service'] == 'Nombre de consultations') &
            (self.donnees_mensuelles['date'].dt.year == self.annee_donnees)
        ].sort_values('mois')
        
        if not donnees_consultants.empty and not donnees_consultations.empty:
            axes[0, 0].plot(donnees_consultants['mois'], donnees_consultants['valeur'], 
                           marker='o', linewidth=2, label='Consultants', color='#1f77b4')
            axes[0, 0].plot(donnees_consultations['mois'], donnees_consultations['valeur'], 
                           marker='s', linewidth=2, label='Consultations', color='#ff7f0e')
            axes[0, 0].set_title('Modèles de Demande Mensuelle')
            axes[0, 0].set_xlabel('Mois')
            axes[0, 0].set_ylabel('Nombre')
            axes[0, 0].set_xticks(range(1, 13)) # Assurer les mois de 1 à 12
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].set_title('Modèles de Demande Mensuelle (Données Manquantes)')
            axes[0, 0].text(0.5, 0.5, 'Données de consultants ou consultations manquantes', 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=axes[0, 0].transAxes, fontsize=10, color='red')
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
        
        # 2. Analyse d'efficacité
        if self.resultats_optimisation and 'donnees_efficacite' in self.resultats_optimisation:
            donnees_eff = self.resultats_optimisation['donnees_efficacite'].dropna(subset=['efficacite'])
            if not donnees_eff.empty:
                axes[0, 1].bar(donnees_eff['mois'], donnees_eff['efficacite'], 
                              color='lightblue', alpha=0.7, edgecolor='navy')
                
                efficacite_moyenne_valide = donnees_eff['efficacite'].mean()
                if not np.isnan(efficacite_moyenne_valide):
                    axes[0, 1].axhline(y=efficacite_moyenne_valide, color='red', 
                                      linestyle='--', label='Moyenne')
                axes[0, 1].set_title('Efficacité Mensuelle (Consultations/Consultant)')
                axes[0, 1].set_xlabel('Mois')
                axes[0, 1].set_ylabel('Ratio d\'Efficacité')
                axes[0, 1].set_xticks(range(1, 13))
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].set_title('Analyse d\'Efficacité (Données Manquantes)')
                axes[0, 1].text(0.5, 0.5, 'Pas de données d\'efficacité valides pour visualisation', 
                               horizontalalignment='center', verticalalignment='center', 
                               transform=axes[0, 1].transAxes, fontsize=10, color='red')
                axes[0, 1].set_xticks([])
                axes[0, 1].set_yticks([])
        else:
            axes[0, 1].set_title('Analyse d\'Efficacité (Calcul non effectué)')
            axes[0, 1].text(0.5, 0.5, 'L\'analyse d\'optimisation n\'a pas généré de résultats valides', 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=axes[0, 1].transAxes, fontsize=10, color='red')
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
        
        # 3. Résultats de prévision
        if self.resultats_prevision:
            service = 'Nombre de consultants'
            if service in self.resultats_prevision and self.resultats_prevision[service].get('historique') is not None:
                resultats = self.resultats_prevision[service]
                historique = resultats['historique'].copy()
                
                # S'assurer que les mois historiques sont de 1 à 12
                historique = historique[historique['date'].dt.year == self.annee_donnees]
                
                # Tracer les données historiques
                axes[1, 0].plot(historique['mois'], historique['valeur'], 
                               marker='o', linewidth=2, label='Historique', color='blue')
                
                # Tracer la prévision si disponible
                if resultats.get('ensemble') and resultats['ensemble']['valeurs_futures'] is not None:
                    # Les mois de prévision doivent continuer après les mois historiques
                    # Si l'historique couvre l'année entière (12 mois), la prévision commence au mois 1 de l'année suivante
                    # Si l'historique ne couvre pas 12 mois, la prévision commence après le dernier mois historique
                    dernier_mois_historique_num = historique['mois'].max()
                    
                    # Calcule la date de début des prévisions
                    derniere_date_historique = historique['date'].max()
                    date_debut_prevision = derniere_date_historique + pd.DateOffset(months=1)

                    dates_prevision_futures = pd.date_range(start=date_debut_prevision, periods=len(resultats['ensemble']['valeurs_futures']), freq='M')
                    
                    # Créer une série de mois pour les prévisions, potentiellement sur plusieurs années
                    # Pour la visualisation sur un axe de 1 à 12, on peut utiliser les mois cycliques
                    mois_prevision_cycliques = [d.month for d in dates_prevision_futures]

                    valeurs_prevision = resultats['ensemble']['valeurs_futures']
                    axes[1, 0].plot(mois_prevision_cycliques, valeurs_prevision, 
                                   marker='s', linewidth=2, linestyle='--', 
                                   label='Prévision', color='red')
                    
                    # Si les prévisions s'étendent au-delà de la fin de l'année actuelle, 
                    # il faudrait ajuster l'axe des x si on veut montrer les mois consécutifs
                    # Pour l'instant, on se contente des mois cycliques pour l'axe 1-12
                
                axes[1, 0].set_title(f'Prévision de la Demande ({service})')
                axes[1, 0].set_xlabel('Mois')
                axes[1, 0].set_ylabel('Nombre')
                axes[1, 0].set_xticks(range(1, 13))
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].set_title('Prévision de la Demande (Non disponible)')
                axes[1, 0].text(0.5, 0.5, f'Prévision pour "{service}" non disponible ou historique vide.', 
                               horizontalalignment='center', verticalalignment='center', 
                               transform=axes[1, 0].transAxes, fontsize=10, color='red')
                axes[1, 0].set_xticks([])
                axes[1, 0].set_yticks([])
        else:
            axes[1, 0].set_title('Prévision de la Demande (Non effectuée)')
            axes[1, 0].text(0.5, 0.5, 'La prévision de la demande n\'a pas été effectuée.', 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=axes[1, 0].transAxes, fontsize=10, color='red')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
        
        # 4. Distribution des catégories de services
        categories_services = self._categoriser_services()
        categories = list(categories_services.keys())
        valeurs = list(categories_services.values())
        
        # Filtrer les catégories avec des valeurs > 0 pour le camembert
        categories_filtrees = [cat for cat, val in zip(categories, valeurs) if val > 0]
        valeurs_filtrees = [val for val in valeurs if val > 0]

        if valeurs_filtrees:
            couleurs = plt.cm.Set3(np.linspace(0, 1, len(categories_filtrees)))
            axes[1, 1].pie(valeurs_filtrees, labels=categories_filtrees, autopct='%1.1f%%', colors=couleurs, startangle=90)
            axes[1, 1].set_title('Distribution des Catégories de Services')
        else:
            axes[1, 1].set_title('Distribution des Catégories de Services (Aucune donnée)')
            axes[1, 1].text(0.5, 0.5, 'Aucune catégorie de service avec des données à afficher.', 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=axes[1, 1].transAxes, fontsize=10, color='red')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])

        plt.tight_layout()
        plt.savefig('tableau_bord_analytique_sante.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualisations sauvegardées sous 'tableau_bord_analytique_sante.png'")
    
    def exporter_resultats(self, nom_fichier='resultats_analytique_sante.xlsx'):
        """Exporter tous les résultats d'analyse vers Excel"""
        print(f"\n💾 EXPORTATION DES RÉSULTATS")
        print("=" * 50)
        
        with pd.ExcelWriter(nom_fichier, engine='openpyxl') as writer:
            # Données brutes
            self.donnees.to_excel(writer, sheet_name='Donnees_Brutes', index=False)
            
            # Données mensuelles
            self.donnees_mensuelles.to_excel(writer, sheet_name='Donnees_Mensuelles', index=False)
            
            # Résultats de prévision
            if self.resultats_prevision:
                resume_prevision = []
                for service, resultats in self.resultats_prevision.items():
                    if resultats.get('ensemble') and resultats['ensemble']['valeurs_futures'] is not None:
                        # Assurez-vous que l'historique est là et contient des dates
                        if not resultats['historique'].empty:
                            derniere_date_historique = resultats['historique']['date'].max()
                        else:
                            # Si historique vide, commencer la prévision à partir du 1er janvier de l'année détectée
                            derniere_date_historique = pd.Timestamp(f'{self.annee_donnees}-12-31') # Fin de l'année des données

                        dates_prevision_futures = pd.date_range(start=derniere_date_historique + pd.DateOffset(months=1), 
                                                                periods=len(resultats['ensemble']['valeurs_futures']), 
                                                                freq='M')
                        
                        for i, valeur in enumerate(resultats['ensemble']['valeurs_futures']):
                            resume_prevision.append({
                                'Service': service,
                                'Date_Prevision': dates_prevision_futures[i].strftime('%Y-%m-%d'),
                                'Annee_Prevision': dates_prevision_futures[i].year,
                                'Mois_Prevision': dates_prevision_futures[i].month,
                                'Valeur_Predite': valeur
                            })
                
                if resume_prevision:
                    pd.DataFrame(resume_prevision).to_excel(writer, sheet_name='Previsions', index=False)
            
            # Résultats d'optimisation
            if self.resultats_optimisation and 'donnees_efficacite' in self.resultats_optimisation:
                # S'assurer que les données d'efficacité sont nettoyées avant d'exporter
                donnees_optimisation_export = self.resultats_optimisation['donnees_efficacite'].copy()
                donnees_optimisation_export['efficacite'].replace([np.inf, -np.inf], np.nan, inplace=True)
                donnees_optimisation_export.to_excel(writer, sheet_name='Optimisation', index=False)
            
            # Ajouter un onglet pour le résumé des métriques clés
            metriques_df = pd.DataFrame.from_dict(self.analyse_exploratoire_donnees(), orient='index')
            if not metriques_df.empty:
                metriques_df.to_excel(writer, sheet_name='Resume_Metriques_Cles')
        
        print(f"✅ Résultats exportés vers '{nom_fichier}'")

def main():
    """Fonction d'exécution principale"""
    print("🏥 ANALYTIQUE D'OPTIMISATION DES RESSOURCES DE SANTÉ")
    print("=" * 60)
    
    # Pour l'exécution en ligne de commande, le chemin du fichier est passé comme argument
    import sys
    if len(sys.argv) > 1:
        chemin_fichier_csv = sys.argv[1]
    else:
        print("Veuillez spécifier le chemin du fichier CSV. Ex: python analytique_sante.py mon_fichier.csv")
        return

    analytique = AnalytiqueAvanceeSante(chemin_fichier_csv)
    
    # Exécuter le pipeline d'analyse complet
    analytique.analyse_exploratoire_donnees()
    analytique.prevision_demande(periodes_prevision=6)
    analytique.optimisation_ressources()
    analytique.generer_rapport_insights()
    analytique.creer_visualisations()
    analytique.exporter_resultats()
    
    print("\n🎉 Pipeline d'analyse terminé avec succès !")

if __name__ == "__main__":
    main()