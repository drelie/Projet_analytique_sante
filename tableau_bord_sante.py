#!/usr/bin/env python3
"""
Tableau de Bord d'Optimisation des Ressources Santé
==================================================
Dashboard interactif pour l'analyse des données de santé
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Pour la détection de l'année à partir du nom du fichier
import re

# For time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet non disponible. Certaines fonctionnalités de prévision seront limitées.")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Tableau de Bord d'Optimisation des Ressources Santé",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AnalytiqueSante:
    def __init__(self):
        self.donnees = None
        self.donnees_mensuelles = None
        self.modeles_prevision = {}
        self.annee_donnees = None # Nouvelle variable pour stocker l'année détectée
        self.resultats_optimisation = {} # Initialiser le dictionnaire pour les résultats d'optimisation
        
    def charger_donnees(self, fichier_telecharge):
        """Charger et préprocesser les données de santé"""
        try:
            self.donnees = pd.read_csv(fichier_telecharge)
            
            # Détecter l'année à partir du nom du fichier
            match = re.search(r'(\d{4})', fichier_telecharge.name) # Utilisez .name pour le nom du fichier Streamlit
            if match:
                self.annee_donnees = int(match.group(1))
            else:
                self.annee_donnees = datetime.now().year # Année par défaut si non trouvée
                
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
                    if pd.notna(ligne.get(nom_mois)):
                        enregistrements_mensuels.append({
                            'service': ligne['service'],
                            'mois': num_mois,
                            'nom_mois': nom_mois,
                            'valeur': ligne[nom_mois],
                            'date': datetime(self.annee_donnees, num_mois, 1), # Utilisation de l'année détectée
                            'trimestre': f'T{(num_mois-1)//3 + 1}',
                            'semestre': 'S1' if num_mois <= 6 else 'S2'
                        })
            
            self.donnees_mensuelles = pd.DataFrame(enregistrements_mensuels)
            st.success(f"Données préprocessées: {len(self.donnees)} services, {len(self.donnees_mensuelles)} enregistrements mensuels")
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement ou du préprocessing des données: {e}")
            return False

    def analyse_exploratoire_donnees(self):
        """Réaliser l'analyse exploratoire des données et afficher les KPIs"""
        if self.donnees_mensuelles is None or self.donnees.empty:
            st.warning("Veuillez charger des données pour l'analyse exploratoire.")
            return

        st.subheader("📊 Analyse Exploratoire des Données")
        col1, col2, col3 = st.columns(3)
        col1.metric("Services Uniques", self.donnees['service'].nunique())
        col2.metric("Enregistrements Mensuels", len(self.donnees_mensuelles))
        
        # Afficher l'année détectée
        if self.annee_donnees:
            col3.metric("Année des Données", self.annee_donnees)

        st.write("---")
        
        st.write("#### Catégories de Services")
        categories_services = self._categoriser_services()
        df_categories = pd.DataFrame(categories_services.items(), columns=['Catégorie', 'Nombre de Services'])
        
        # --- MODIFICATION : Remplacer le tableau par un Pie Chart ---
        fig_pie = px.pie(
            df_categories,
            values='Nombre de Services',
            names='Catégorie',
            title='Distribution des Services par Catégorie',
            hole=0.3 # Pour un donut chart
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        # -------------------------------------------------------------

        st.write("---")

        st.write("#### Résumé des Métriques Clés")
        services_cles = ['Nombre de consultants', 'Nombre de consultations']
        metrics_data = []

        for service in services_cles:
            # Utiliser str.contains pour une correspondance plus flexible
            if self.donnees_mensuelles['service'].str.contains(service, case=False, na=False).any():
                # Agrégation pour s'assurer d'avoir une seule valeur par mois pour ce service
                service_data = self.donnees_mensuelles[self.donnees_mensuelles['service'].str.contains(service, case=False, na=False)].copy()
                service_data['valeur'] = pd.to_numeric(service_data['valeur'], errors='coerce')
                service_data = service_data.dropna(subset=['valeur'])
                
                # S'assurer d'une seule valeur par mois en sommant si plusieurs sous-catégories existaient
                service_data_agg = service_data.groupby('date')['valeur'].sum().reset_index()

                if not service_data_agg.empty:
                    total_annuel = service_data_agg['valeur'].sum()
                    moyenne_mensuelle = service_data_agg['valeur'].mean()
                    std_mensuelle = service_data_agg['valeur'].std()
                    cv = (std_mensuelle / moyenne_mensuelle) if moyenne_mensuelle > 0 else 0
                    
                    metrics_data.append({
                        'Service': service,
                        'Total Annuel': f"{total_annuel:,.0f}",
                        'Moyenne Mensuelle': f"{moyenne_mensuelle:,.0f}",
                        'Coefficient de Variation': f"{cv:.2%}"
                    })
        
        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
        else:
            st.info("Aucune donnée pour les services clés 'Nombre de consultants' ou 'Nombre de consultations'.")


        # Visualisation de la demande mensuelle
        st.write("#### Demande Mensuelle des Services Clés")
        # Utiliser str.contains pour une correspondance plus flexible
        df_plot = self.donnees_mensuelles[self.donnees_mensuelles['service'].str.contains('consultant|consultation', case=False, na=False)].copy()
        
        # S'assurer que les valeurs sont numériques
        df_plot['valeur'] = pd.to_numeric(df_plot['valeur'], errors='coerce')
        df_plot.dropna(subset=['valeur'], inplace=True)

        # Agrégation pour s'assurer d'une seule valeur par mois pour chaque service clé
        df_plot_agg = df_plot.groupby(['date', 'service'])['valeur'].sum().reset_index()

        df_plot_agg['Mois_Nom'] = df_plot_agg['date'].dt.strftime('%B') # Nom complet du mois

        fig_demande = px.line(
            df_plot_agg,
            x='Mois_Nom',
            y='valeur',
            color='service',
            title='Tendance Mensuelle des Consultants et Consultations',
            labels={'valeur': 'Nombre', 'Mois_Nom': 'Mois'},
            markers=True
        )
        fig_demande.update_xaxes(categoryorder='array', categoryarray=[pd.to_datetime(f'2000-{m}-01').strftime('%B') for m in range(1, 13)])
        st.plotly_chart(fig_demande, use_container_width=True)


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
            'Autres': len(services) - (
                services.str.contains('consultation', na=False).sum() +
                services.str.contains('consultant', na=False).sum() +
                services.str.contains('vaccin', na=False).sum() +
                services.str.contains('paludisme|diarrhée|toux|fièvre|ira', na=False).sum() +
                services.str.contains('<5 ans|>5 ans|femme|homme', na=False).sum() +
                services.str.contains('cpn|prénatale|grossesse', na=False).sum() +
                services.str.contains('pansement|injection|perfusion|chirurgie', na=False).sum()
            )
        }
        # Filtrer les catégories avec 0 services pour ne pas les afficher dans le pie chart
        return {k: v for k, v in categories.items() if v > 0}

    def prevision_demande(self):
        """Réaliser la prévision de la demande pour les services clés"""
        if self.donnees_mensuelles is None or self.donnees_mensuelles.empty:
            st.warning("Veuillez charger des données pour la prévision de la demande.")
            return

        st.subheader("🔮 Prévision de la Demande")
        periodes_prevision = st.slider("Nombre de mois à prévoir :", 1, 12, 6)

        services_a_prevoir = ['Nombre de consultants', 'Nombre de consultations', 'Nombre de consultants <5 ans']
        
        self.modeles_prevision = {}

        for service in services_a_prevoir:
            # Utiliser str.contains pour une correspondance plus flexible
            donnees_service = self.donnees_mensuelles[self.donnees_mensuelles['service'].str.contains(service, case=False, na=False)].copy()
            
            if donnees_service.empty or len(donnees_service) < 12: # Nécessite au moins un an de données
                st.info(f"Données insuffisantes pour prévoir '{service}'. Nécessite au moins 12 mois de données.")
                continue

            # Assurez-vous que la colonne 'valeur' est numérique et agrégez par date
            donnees_service['valeur'] = pd.to_numeric(donnees_service['valeur'], errors='coerce')
            donnees_service.dropna(subset=['valeur'], inplace=True)
            donnees_service_agg = donnees_service.groupby('date')['valeur'].sum().reset_index()
            
            # Pour Prophet, renommer les colonnes
            df_prophet = donnees_service_agg[['date', 'valeur']].rename(columns={'date': 'ds', 'valeur': 'y'})
            
            if PROPHET_AVAILABLE:
                with st.spinner(f"Entraînement du modèle Prophet pour {service}..."):
                    try:
                        modele_prophet = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            seasonality_mode='multiplicative'
                        )
                        modele_prophet.fit(df_prophet)
                        
                        futur = modele_prophet.make_future_dataframe(periods=periodes_prevision, freq='M')
                        prevision = modele_prophet.predict(futur)
                        
                        self.modeles_prevision[service] = {
                            'modele': modele_prophet,
                            'prevision': prevision,
                            'historique': donnees_service_agg # Utiliser les données agrégées comme historique
                        }

                        st.success(f"Prévision générée pour '{service}'.")
                        
                        # Visualisation de la prévision
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historique'))
                        fig.add_trace(go.Scatter(x=prevision['ds'], y=prevision['yhat'], mode='lines', name='Prévision', line=dict(dash='dash')))
                        fig.add_trace(go.Scatter(x=prevision['ds'], y=prevision['yhat_lower'], mode='lines', name='Intervalle inférieur', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=prevision['ds'], y=prevision['yhat_upper'], mode='lines', name='Intervalle supérieur', fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False))
                        
                        fig.update_layout(title=f"Prévision de la Demande pour '{service}'",
                                        xaxis_title="Date",
                                        yaxis_title="Valeur")
                        st.plotly_chart(fig, use_container_width=True)

                        # Afficher les prévisions futures
                        # L'année des prévisions sera l'année suivante de la dernière année des données
                        if self.annee_donnees:
                             st.write(f"##### Prévisions pour '{service}' ({self.annee_donnees + 1})")
                        else:
                            st.write(f"##### Prévisions pour '{service}'") # Fallback si l'année n'est pas détectée
                        future_forecast = prevision[['ds', 'yhat']].tail(periodes_prevision).rename(columns={'ds': 'Date', 'yhat': 'Prévision'}).copy()
                        future_forecast['Date'] = future_forecast['Date'].dt.strftime('%Y-%m')
                        st.dataframe(future_forecast, hide_index=True)

                    except Exception as e:
                        st.error(f"Erreur lors de l'entraînement ou de la prévision pour '{service}' : {e}")
            else:
                st.info("Le modèle Prophet n'est pas disponible pour la prévision.")

    def optimisation_ressources(self):
        """Analyser l'optimisation des ressources"""
        if self.donnees_mensuelles is None or self.donnees_mensuelles.empty:
            st.warning("Veuillez charger des données pour l'analyse d'optimisation des ressources.")
            return

        st.subheader("⚡ Analyse d'Optimisation des Ressources")

        # Utiliser `str.contains` pour être plus robuste si les noms de service varient légèrement
        donnees_consultants = self.donnees_mensuelles[self.donnees_mensuelles['service'].str.contains('consultant', case=False, na=False)].copy()
        donnees_consultations = self.donnees_mensuelles[self.donnees_mensuelles['service'].str.contains('consultation', case=False, na=False)].copy()
        
        # S'assurer que les valeurs sont numériques avant l'agrégation
        donnees_consultants['valeur'] = pd.to_numeric(donnees_consultants['valeur'], errors='coerce')
        donnees_consultations['valeur'] = pd.to_numeric(donnees_consultations['valeur'], errors='coerce')

        # Agrégation par date pour éviter les duplicatas de mois pour chaque service
        # On somme toutes les entrées de valeur pour une même date et service
        donnees_consultants = donnees_consultants.groupby('date')['valeur'].sum().reset_index()
        donnees_consultations = donnees_consultations.groupby('date')['valeur'].sum().reset_index()

        if donnees_consultants.empty or donnees_consultations.empty:
            st.info("Données 'Nombre de consultants' ou 'Nombre de consultations' manquantes pour l'optimisation.")
            self.resultats_optimisation = {} # Réinitialiser si pas de données
            return

        
        donnees_fusionnees = pd.merge(
            donnees_consultants[['date', 'valeur']].rename(columns={'valeur': 'consultants'}),
            donnees_consultations[['date', 'valeur']].rename(columns={'valeur': 'consultations'}),
            on='date',
            how='inner'
        )
        
        # Supprimer les lignes où 'consultants' est 0 pour éviter la division par zéro
        donnees_fusionnees = donnees_fusionnees[donnees_fusionnees['consultants'] > 0].copy()
        if donnees_fusionnees.empty:
            st.warning("Aucune donnée de consultants positive pour calculer l'efficacité.")
            self.resultats_optimisation = {} # Réinitialiser si pas de données
            return

        donnees_fusionnees['efficacite'] = donnees_fusionnees['consultations'] / donnees_fusionnees['consultants']
        donnees_fusionnees['charge_travail'] = donnees_fusionnees['consultations'] / donnees_fusionnees['consultants'] # Similaire à efficacité pour cet exemple

        # Stocker les données d'efficacité fusionnées pour un accès ultérieur
        self.resultats_optimisation['donnees_efficacite'] = donnees_fusionnees.copy()

        efficacite_moyenne = donnees_fusionnees['efficacite'].mean()
        ecart_type_efficacite = donnees_fusionnees['efficacite'].std()

        st.write("#### Performance Actuelle")
        col1, col2 = st.columns(2)
        col1.metric("Efficacité Moyenne", f"{efficacite_moyenne:.2f} consultations/consultant")
        col2.metric("Coefficient de Variation", f"{ecart_type_efficacite/efficacite_moyenne:.2%}" if efficacite_moyenne > 0 else "N/A")

        st.write("#### Opportunités d'Optimisation")
        # Les calculs ici sont dynamiques et basés sur les données réelles
        mois_haute_efficacite = donnees_fusionnees[donnees_fusionnees['efficacite'] > efficacite_moyenne + ecart_type_efficacite]
        mois_faible_efficacite = donnees_fusionnees[donnees_fusionnees['efficacite'] < efficacite_moyenne - ecart_type_efficacite]
        
        st.info(f"Mois à haute efficacité: {len(mois_haute_efficacite)} / Mois à faible efficacité: {len(mois_faible_efficacite)}")
        
        if not mois_faible_efficacite.empty:
            st.write("Mois nécessitant une attention particulière :")
            # --- MODIFICATION : Afficher le nom du mois au lieu de la date complète ---
            mois_faible_efficacite_display = mois_faible_efficacite[['date', 'consultants', 'consultations', 'efficacite']].round(2).copy()
            mois_faible_efficacite_display['Mois'] = mois_faible_efficacite_display['date'].dt.strftime('%B')
            st.dataframe(mois_faible_efficacite_display[['Mois', 'consultants', 'consultations', 'efficacite']], hide_index=True)
            # -------------------------------------------------------------------------


        st.write("#### Stratégie de Réallocation des Ressources")
        total_consultations = donnees_fusionnees['consultations'].sum()
        consultants_actuels_total = donnees_fusionnees['consultants'].sum()

        if efficacite_moyenne > 0:
            consultants_optimaux_total = total_consultations / efficacite_moyenne
            reallocation_necessaire = consultants_optimaux_total - consultants_actuels_total
            st.info(f"Consultants actuels totaux: {consultants_actuels_total:,.0f} / Consultants optimaux totaux: {consultants_optimaux_total:,.0f}")
            st.success(f"Réallocation annuelle nécessaire: {reallocation_necessaire:+.0f} consultants")

            # Réallocation mensuelle basée sur l'efficacité cible moyenne
            df_reallocation = donnees_fusionnees.copy()
            df_reallocation['consultants_optimaux'] = df_reallocation['consultations'] / efficacite_moyenne
            df_reallocation['ajustement'] = df_reallocation['consultants_optimaux'] - df_reallocation['consultants']
            
            # Formater pour l'affichage
            df_reallocation['date_str'] = df_reallocation['date'].dt.strftime('%B')
            df_reallocation_display = df_reallocation[['date_str', 'consultants', 'consultations', 'efficacite', 'ajustement']].round(2)
            df_reallocation_display.rename(columns={
                'date_str': 'Mois',
                'consultants': 'Consultants Actuels',
                'consultations': 'Consultations',
                'efficacite': 'Efficacité',
                'ajustement': 'Ajustement Nécessaire'
            }, inplace=True)
            st.dataframe(df_reallocation_display, hide_index=True)

        else:
            st.warning("Impossible de calculer la réallocation : efficacité moyenne nulle.")

        st.write("---")
        st.write("#### Potentiel d'Optimisation et Économies")
        if efficacite_moyenne > 0:
            # Estimation simplifiée des économies
            cout_moyen_consultant_mois = 1000 # Exemple de coût
            
            # Nombre de consultants "idéal" si chaque mois atteignait la meilleure efficacité observée
            meilleure_efficacite_mois = donnees_fusionnees['efficacite'].max()
            
            if meilleure_efficacite_mois > 0:
                # Calcul des consultants nécessaires avec la meilleure efficacité possible
                consultants_necessaires_opt = total_consultations / meilleure_efficacite_mois
                reduction_consultants = consultants_actuels_total - consultants_necessaires_opt
                
                economies_annuelles_potentielles = reduction_consultants * cout_moyen_consultant_mois * 12
                
                st.metric("Économies Annuelles Potentielles", f"{economies_annuelles_potentielles:,.0f} Unités Monétaires")
                st.info(f"Réduction potentielle de consultants: {reduction_consultants:,.0f}")
            else:
                st.info("Efficacité maximale observée nulle, impossible d'estimer les économies.")
        else:
            st.info("Efficacité moyenne nulle, impossible d'estimer les économies.")


    def generer_rapport_insights(self):
        """Générer un rapport d'insights et recommandations"""
        st.subheader("📋 Rapport d'Insights et Recommandations")

        st.write("#### Principales Découvertes")
        if self.modeles_prevision:
            st.markdown("##### Prévisions de Demande :")
            for service, resultats in self.modeles_prevision.items():
                if resultats.get('prevision') is not None:
                    # Prendre la première prévision future
                    prevision_mois_prochain = resultats['prevision']['yhat'].tail(1).iloc[0]
                    
                    # Obtenir la dernière valeur historique
                    # Assurez-vous que l'historique a bien des données pour la dernière valeur
                    if not resultats['historique'].empty:
                        derniere_valeur_historique = resultats['historique']['valeur'].iloc[-1]
                    else:
                        derniere_valeur_historique = 0 # Valeur par défaut si historique vide
                    
                    if derniere_valeur_historique != 0:
                        changement = (prevision_mois_prochain - derniere_valeur_historique) / derniere_valeur_historique
                        st.write(f"- **{service}**: {changement:+.1%} changement attendu le mois prochain")
                    else:
                        st.write(f"- **{service}**: Prévision pour le mois prochain : {prevision_mois_prochain:,.0f} (Impossible de calculer le changement, dernière valeur historique nulle).")
        else:
            st.info("Aucune prévision générée.")

        # Potentiel d'optimisation
        if 'donnees_efficacite' in self.resultats_optimisation and not self.resultats_optimisation['donnees_efficacite'].empty:
            donnees_fusionnees = self.resultats_optimisation['donnees_efficacite']
            
            # S'assurer que 'efficacite' est numérique
            donnees_fusionnees['efficacite'] = pd.to_numeric(donnees_fusionnees['efficacite'], errors='coerce')
            donnees_fusionnees.dropna(subset=['efficacite'], inplace=True) # Supprimer les NaN après conversion

            if not donnees_fusionnees.empty and donnees_fusionnees['efficacite'].sum() > 0: # Vérifier après le dropna
                efficacite_moyenne = donnees_fusionnees['consultations'].sum() / donnees_fusionnees['consultants'].sum()
                
                # Estimer la meilleure efficacité réalisable (par exemple, le 75e percentile)
                meilleure_efficacite_cible = donnees_fusionnees['efficacite'].quantile(0.75)

                if efficacite_moyenne > 0 and meilleure_efficacite_cible > 0:
                    potentiel_amelioration = (meilleure_efficacite_cible - efficacite_moyenne) / efficacite_moyenne
                    st.write(f"- **Potentiel d'optimisation des ressources**: {potentiel_amelioration:.1%} amélioration d'efficacité si la moyenne atteint le 75e percentile d'efficacité mensuelle.")
                    
                    # Economies potentielles (simplifié pour le tableau de bord)
                    cout_moyen_consultant_mois = 1000 
                    consultants_actuels_total = donnees_fusionnees['consultants'].sum()
                    consultations_total = donnees_fusionnees['consultations'].sum()

                    consultants_optimaux_cible = consultations_total / meilleure_efficacite_cible
                    economies_annuelles = (consultants_actuels_total - consultants_optimaux_cible) * cout_moyen_consultant_mois * 12
                    
                    if economies_annuelles > 0:
                         st.write(f"- **Économies annuelles potentielles**: {economies_annuelles:,.0f} Unités Monétaires (estimation)")
                    else:
                        st.write("- Pas d'économies annuelles potentielles estimées ou valeur négative.")
                else:
                    st.info("Impossible de calculer le potentiel d'optimisation ou les économies (données d'efficacité insuffisantes ou nulles).")
            else:
                st.info("Données d'efficacité insuffisantes ou nulles pour l'optimisation.")
        else:
            st.info("Veuillez charger des données pour générer des insights d'optimisation.")

        st.write("#### Recommandations Stratégiques")
        st.markdown("""
        * **1. Allocation Dynamique du Personnel** : Utiliser les prévisions de demande pour ajuster les effectifs mensuellement.
        * **2. Focus sur les Périodes de Forte Demande** : Renforcer les ressources durant les mois identifiés comme à forte demande (ex: Juillet, Août) pour maintenir la qualité.
        * **3. Intégration de la Télésanté** : Promouvoir la télésanté pendant les périodes de faible demande pour optimiser l'utilisation des consultants et réduire les coûts.
        * **4. Programmes de Formation Croisée** : Développer la polyvalence du personnel pour une flexibilité accrue lors des fluctuations de la demande.
        * **5. Tableau de Bord en Temps Réel** : Mettre en place un outil de suivi pour ajuster rapidement les stratégies en fonction des performances et des prévisions.
        """)

        st.write("#### Feuille de Route d'Implémentation")
        st.markdown("""
        * **Phase 1 (Mois 1-2)** : Déploiement initial des modèles de prévision et mise en place d'un suivi des indicateurs clés.
        * **Phase 2 (Mois 3-4)** : Implémentation d'algorithmes d'optimisation pour la réallocation des ressources basée sur les données.
        * **Phase 3 (Mois 5-6)** : Développement d'un système automatisé d'allocation des ressources pour une efficacité maximale.
        """)

    def afficher_informations_fichier(self):
        """Affiche les informations sur le fichier chargé"""
        if self.donnees is not None:
            st.sidebar.subheader("ℹ️ Informations sur les Données")
            
            # Afficher l'année détectée une seule fois ici
            if self.annee_donnees:
                st.sidebar.success(f"Année détectée : {self.annee_donnees}")
            else:
                st.sidebar.warning(f"Année non détectée dans le nom du fichier. Utilisation de l'année actuelle : {datetime.now().year}")

            st.sidebar.write(f"**Forme du jeu de données:** {self.donnees.shape}")
            st.sidebar.write(f"**Colonnes:** {', '.join(self.donnees.columns.tolist())}")
            
            st.sidebar.write("---")
            if st.sidebar.checkbox("Afficher les 5 premières lignes"):
                st.sidebar.dataframe(self.donnees.head())
            if st.sidebar.checkbox("Afficher les statistiques descriptives"):
                st.sidebar.dataframe(self.donnees.describe())

# --- Application Streamlit ---
def main():
    st.title("🏥 Tableau de Bord d'Optimisation des Ressources Santé")
    st.markdown("""
    Cette application interactive vous permet d'analyser vos données de santé, de prévoir la demande et d'optimiser l'allocation de vos ressources.
    """)

    analytique_app = AnalytiqueSante()

    with st.sidebar:
        st.header("Upload Data")
        fichier_telecharge = st.file_uploader("Téléchargez votre fichier CSV ici", type=["csv"])
        
        if fichier_telecharge is not None:
            if analytique_app.charger_donnees(fichier_telecharge):
                analytique_app.afficher_informations_fichier()
            else:
                st.error("Échec du chargement des données. Veuillez vérifier le format du fichier.")
        else:
            st.info("👆 Veuillez télécharger votre fichier CSV de données de santé pour commencer l'analyse")
            
            # Afficher la structure de données d'exemple
            st.subheader("Format de Données Attendu")
            st.markdown("""
            Votre fichier CSV doit contenir une colonne 'service' pour le nom du service,
            et des colonnes pour chaque mois (JANVIER, FEVRIER, etc.) avec les valeurs correspondantes,
            ainsi qu'une colonne 'TOTAUX'. Exemple:
            """)
            donnees_exemple = {
                'service': ['Nombre de consultants', 'Nombre de consultations', 'Nombre de consultants <5 ans'],
                'JANVIER': [965, 991, 194],
                'FEVRIER': [844, 890, 173],
                'MARS': [806, 828, 169],
                'TOTAUX': [10388, 10589, 1951]
            }
            st.dataframe(pd.DataFrame(donnees_exemple))

    # --- Contenu principal du tableau de bord ---
    if analytique_app.donnees is not None:
        onglet1, onglet2, onglet3 = st.tabs(["Analyse Exploratoire", "Prévision de la Demande", "Optimisation & Recommandations"])

        with onglet1:
            analytique_app.analyse_exploratoire_donnees()

        with onglet2:
            analytique_app.prevision_demande()

        with onglet3:
            analytique_app.optimisation_ressources() # S'assurer que cette méthode est appelée avant generer_rapport_insights
            analytique_app.generer_rapport_insights()
        
        # Fonctionnalité d'export
        st.subheader("📥 Options d'Export")
        if st.button("Générer Rapport d'Analyse (PDF/HTML - à implémenter)"):
            st.info("Cette fonctionnalité générerait un rapport PDF ou HTML avec toutes les analyses.")
            st.warning("Pour l'instant, veuillez faire des captures d'écran des sections ou implémenter votre propre logique d'export.")

    else:
        st.info("Chargez vos données depuis la barre latérale pour commencer l'analyse.")


if __name__ == "__main__":
    main()