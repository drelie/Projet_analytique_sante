# Documentation des Modèles

## Hypothèses Principales

### Modèle d'Efficacité
- **Formule**: `Efficacité = Nombre consultations / Nombre consultants`
- **Sources**:
  - Rapport OMS 2022 sur les indicateurs de performance
  - Étude nationale sur l'allocation des ressources (2021)
- **Limitations**:
  - Ne tient pas compte de la complexité des cas
  - Suppose une homogénéité des compétences

### Paramètres d'Optimisation
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Coût consultant/mois | 1000 UM | Moyenne nationale |
| Seuil haute efficacité | >1.1x moyenne | Analyse historique |
| Période minimale | 12 mois | Saisonnalité annuelle |

## Modèles de Prévision

### Prophet
- **Configuration**:
  - Saisonnalité annuelle
  - Mode multiplicatif
  - Changepoint prior scale: 0.05
- **Performance**: MAPE de 12.5%

### ARIMA
- **Ordre**: (1,1,1)
- **Période**: Mensuelle
- **Performance**: MAPE de 15.2%