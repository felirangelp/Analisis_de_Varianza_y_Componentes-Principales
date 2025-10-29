# Análisis de Varianza y Componentes Principales

Este proyecto contiene un análisis completo de varianza y componentes principales (PCA) aplicado al conjunto de datos de reconocimiento de actividades humanas con smartphones de Kaggle.

## 📊 Contenido

- **report.html**: Dashboard interactivo con visualizaciones de Plotly que presenta todos los análisis del taller.
- **generate_report.py**: Script de Python que genera el reporte HTML completo.

## 🚀 Ver el Reporte

### Opción 1: GitHub Pages
El reporte está disponible en:
**https://felirangelp.github.io/Analisis여_de_Varianza_y_Componentes-Principales/report.html**

### Opción 2: Localmente
1. Clona este repositorio:
   ```bash
   git clone https://github.com/felirangelp/Analisis_de_Varianza_y_Componentes-Principales.git
   cd Analisis_de_Varianza_y_Componentes-Principales
   ```

2. Abre directamente `report.html` en tu navegador web.

## 🔧 Configuración Local

Si deseas regenerar el reporte:

1. Crea un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Descarga los datos de Kaggle en el directorio `data/`:
   - Dataset: Human Activity Recognition with Smartphones
   - URL: https://www.kaggle.com/datasets/ucdb/human-activity-recognition-with-smartphones

4. Ejecuta el script de generación:
   ```bash
   python generate_report.py
   ```

## 📈 Análisis Incluidos

1. **Análisis de Varianza**: Visualización de la varianza de las 50 características más importantes
2. **Visualización 3D**: Gráficos interactivos de las características con mayor varianza
3. **Aplicación de PCA**: Análisis de componentes principales con Scree Plot
4. **Reducción de Dimensionalidad**: Conclusiones sobre el número óptimo de componentes
5. **Visualización PCA 2D y 3D**: Proyecciones de los datos en el espacio de componentes principales
6. **Análisis de Loadings**: Interpretación de las características más influyentes en cada componente

## 📝 Licencia

Este proyecto es parte de un taller académico sobre análisis de señales biológicas.

## 🔗 Referencias

- [Dataset en Kaggle](https://www.kaggle.com/datasets/ucdb/human-activity-recognition-with-smartphones)
- [Documentación de Plotly](https://plotly.com/python/)
- [Documentación de scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

