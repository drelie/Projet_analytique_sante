#!/usr/bin/env python3
"""
Tableau de Bord d'Optimisation des Ressources Santé
Version avec standardisation des services clés
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

# Import nécessaire pour les prévisions
from prophet import Prophet

# Utilitaires
import re
import os
from io import BytesIO

class AnalytiqueSante:
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
    
    def __init__(self):
        self.donnees = None
        self.donnees_mensuelles = None
        self.annee_donnees = None

    def _standardiser_nom_service(self, nom: str) -> str:
        """Standardise le nom du service selon le mapping défini."""
        nom = str(nom).strip()
        for pattern, standard in self.SERVICE_STANDARD.items():
            if re.search(pattern, nom, re.IGNORECASE):
                return standard
        return nom

    def charger_donnees(self, fichier_upload) -> bool:
        """Charge les données depuis un fichier uploadé."""
        try:
            # Lecture du fichier
            self.donnees = pd.read_csv(
                fichier_upload,
                dtype={'service': 'object'},
                engine='c'
            )

            # Détection de l'année
            match = re.search(r'(\d{4})', fichier_upload.name)
            self.annee_donnees = int(match.group(1)) if match else datetime.now().year

            # Standardiser les noms de services
            self.donnees['service'] = self.donnees['service'].apply(self._standardiser_nom_service)
            
            # Filtrer uniquement les services clés
            self.donnees = self.donnees[self.donnees['service'].isin(self.SERVICES_CLES)]
            
            # Nettoyage
            self.donnees.columns = self.donnees.columns.str.strip()
            self.donnees = self.donnees[self.donnees['service'].notna()].copy()

            # Transformation
            self.donnees_mensuelles = self._transformer_en_mensuel()
            
            st.success(f"Données chargées: {len(self.donnees)} services clés")
            return True
            
        except Exception as e:
            st.error(f"Erreur de chargement: {str(e)}")
            return False

    def _transformer_en_mensuel(self) -> pd.DataFrame:
        """Transforme les données en format mensuel en gérant les colonnes manquantes et les variantes de noms."""
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
            'AOÛT': ['AOUT', 'AOÛT', 'AUGUST', 'AUG'],
            'SEPTEMBRE': ['SEPT', 'SEPTEMBRE', 'SEPTEMBER', 'SEP'],
            'OCTOBRE': ['OCT', 'OCTOBRE', 'OCTOBER'],
            'NOVEMBRE': ['NOV', 'NOVEMBRE', 'NOVEMBER'],
            'DÉCEMBRE': ['DEC', 'DECEMBRE', 'DECEMBER', 'DÉCEMBRE']
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
            st.warning("Aucune colonne de mois trouvée dans les données")
            return pd.DataFrame()

        # Transformation avec melt
        donnees_long = self.donnees.melt(
            id_vars=['service'],
            value_vars=list(colonnes_trouvees.values()),
            var_name='nom_mois_brut',
            value_name='valeur'
        ).dropna(subset=['valeur'])

        # Normaliser les noms de mois
        mapping_normalisation = {v: k for k, v in colonnes_trouvees.items()}
        donnees_long['nom_mois'] = donnees_long['nom_mois_brut'].map(mapping_normalisation)

        # Mapping vers les numéros de mois
        mapping_mois = {m: i+1 for i, m in enumerate(mois_variantes.keys())}
        donnees_long['mois'] = donnees_long['nom_mois'].map(mapping_mois)

        # Ajout des dates
        donnees_long['date'] = donnees_long['mois'].apply(
            lambda m: datetime(self.annee_donnees, m, 1) if not pd.isna(m) else None
        )
        donnees_long['trimestre'] = donnees_long['mois'].apply(
            lambda m: f'T{(m-1)//3 + 1}' if not pd.isna(m) else None
        )
        donnees_long['semestre'] = donnees_long['mois'].apply(
            lambda m: 'S1' if m <= 6 else 'S2' if not pd.isna(m) else None
        )
        
        return donnees_long

    def analyse_exploratoire_donnees(self):
        """Réalise l'analyse exploratoire avec visualisations."""
        if self.donnees_mensuelles is None or self.donnees_mensuelles.empty:
            st.warning("Aucune donnée disponible pour l'analyse")
            return

        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Services Clés", len(self.SERVICES_CLES))
        col2.metric("Enregistrements Mensuels", len(self.donnees_mensuelles))
        col3.metric("Année des Données", self.annee_donnees)

        # Visualisation des services
        if not self.donnees_mensuelles.empty:
            fig_pie = px.pie(
                self.donnees_mensuelles.groupby('service').size().reset_index(name='count'),
                values='count',
                names='service',
                title='Répartition des Services Clés',
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Aucune donnée à afficher")

        # Tendances temporelles
        if not self.donnees_mensuelles.empty:
            fig_tendance = px.line(
                self.donnees_mensuelles,
                x='date',
                y='valeur',
                color='service',
                title='Tendance Mensuelle des Services Clés',
                markers=True
            )
            st.plotly_chart(fig_tendance, use_container_width=True)
        else:
            st.warning("Aucune donnée de tendance disponible")

    def prevision_demande(self, periodes: int = 6):
        """Prévoit la demande pour les services clés."""
        if self.donnees_mensuelles is None or self.donnees_mensuelles.empty:
            st.warning("Aucune donnée disponible pour la prévision")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Traiter tous les services clés
        for i, service in enumerate(self.SERVICES_CLES):
            status_text.text(f"Traitement du service {i+1}/{len(self.SERVICES_CLES)}: {service}")
            progress_bar.progress((i+1)/len(self.SERVICES_CLES))
            
            df_service = self.donnees_mensuelles[
                self.donnees_mensuelles['service'] == service
            ]
            
            if df_service.empty:
                continue
                
            try:
                # Préparation des données pour Prophet
                df_prophet = df_service[['date', 'valeur']].rename(columns={'date': 'ds', 'valeur': 'y'})
                
                # Création et entraînement du modèle
                model = Prophet(yearly_seasonality=True)
                model.fit(df_prophet)
                
                # Génération des prévisions
                future = model.make_future_dataframe(periods=periodes, freq='M')
                forecast = model.predict(future)
                
                # Visualisation - prévisions en rouge
                fig = go.Figure()
                
                # Historique en bleu
                fig.add_trace(go.Scatter(
                    x=df_prophet['ds'], 
                    y=df_prophet['y'], 
                    name='Historique', 
                    mode='lines+markers',
                    line=dict(color='blue')
                ))
                
                # Prévision en rouge
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], 
                    y=forecast['yhat'], 
                    name='Prévision', 
                    line=dict(dash='dash', color='red')
                ))
                
                fig.update_layout(
                    title=f"Prévision pour {service}",
                    xaxis_title="Date",
                    yaxis_title="Valeur"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lors de la prévision pour {service}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()

    def generer_rapport_insights(self):
        """Génère un rapport d'analyse téléchargeable."""
        if self.donnees_mensuelles is None or self.donnees_mensuelles.empty:
            st.warning("Aucune donnée disponible pour générer le rapport")
            return

        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Données mensuelles
            self.donnees_mensuelles.to_excel(writer, sheet_name='Donnees_Mensuelles')
            
            # Statistiques descriptives
            stats = self.donnees_mensuelles.groupby('service')['valeur'].agg(['mean', 'median', 'std', 'sum'])
            stats.to_excel(writer, sheet_name='Statistiques')
        
        st.download_button(
            label="Télécharger le rapport complet",
            data=output.getvalue(),
            file_name=f"rapport_optimisation_{self.annee_donnees}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def main():
    st.title("🏥 Tableau de Bord d'Optimisation des Ressources Santé")
    
    # Initialisation
    analytique = AnalytiqueSante()
    
    # Sidebar - Upload
    with st.sidebar:
        st.header("Chargement des données")
        fichier = st.file_uploader("Téléverser un fichier CSV", type=["csv"])
        
        if fichier is not None:
            if analytique.charger_donnees(fichier):
                st.success("Données chargées avec succès!")
            else:
                st.error("Erreur lors du chargement")

    # Main content
    if analytique.donnees is not None:
        tabs = st.tabs(["Analyse Exploratoire", "Prévisions", "Optimisation"])
        
        with tabs[0]:
            analytique.analyse_exploratoire_donnees()
        
        with tabs[1]:
            periodes = st.slider("Mois à prévoir", 1, 12, 6)
            analytique.prevision_demande(periodes)
        
        with tabs[2]:
            analytique.generer_rapport_insights()

if __name__ == "__main__":
    main()