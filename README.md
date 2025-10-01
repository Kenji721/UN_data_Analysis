# Análisis de Indicadores de Desarrollo de Países (UN Data)

## Descripción del Proyecto

Este proyecto analiza una base de datos de la **Organización de las Naciones Unidas (UNdata)** que contiene **indicadores socioeconómicos, demográficos, de salud y educación** de distintos países y años[cite: 613]. [cite\_start]El objetivo es transformar los datos crudos en datasets listos para el análisis, con el fin de responder a preguntas clave sobre el desarrollo y apoyar la toma de decisiones en políticas públicas.

El repositorio incluye un *pipeline* de limpieza que procesa los datos y los divide en categorías para facilitar su uso en análisis exploratorio (EDA), pruebas de hipótesis (t-tests), análisis de componentes principales (PCA) y modelos de aprendizaje automático.

## Estructura del Proyecto

El repositorio está organizado en las siguientes carpetas:

  - `data/`: Contiene los datos crudos (`raw/`) y los datasets procesados (`processed/`).
  - `src/`: Incluye el módulo de limpieza de datos (`UNCountryDataCleaner`) y otros scripts de utilidad.
  - `notebooks/`: Contiene los Jupyter Notebooks con el análisis exploratorio, las pruebas estadísticas y la construcción de modelos.
  - `tests/`: Contiene pruebas para asegurar la calidad y consistencia del código.

Además, se incluyen los siguientes archivos principales:

  - `run_data_cleaning.py`: Script para ejecutar el pipeline de limpieza de datos de principio a fin.
  - `requirements.txt`: Archivo con las dependencias necesarias para ejecutar el proyecto.

## Instalación

Para configurar el entorno de desarrollo y ejecutar el proyecto, sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/tu-repositorio.git
    cd tu-repositorio
    ```
2.  **Crea y activa un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para ejecutar el pipeline de limpieza y generar los datasets procesados, utiliza el siguiente comando:

```bash
python run_data_cleaning.py
```

Los datasets limpios se guardarán en la carpeta `data/processed/` y estarán listos para ser utilizados en los notebooks de análisis.

## Enfoque Analítico

El análisis se centra en responder a **cinco preguntas de negocio** clave a través de:

  - **Pruebas de hipótesis (t-tests de Welch):** Para comparar las medias entre diferentes grupos de países y determinar si existen diferencias estadísticamente significativas.
  - **Análisis de Componentes Principales (PCA):** Para reducir la dimensionalidad de los datos, identificar patrones multivariados y facilitar la interpretación de las relaciones entre las variables.
  - **Modelos de Machine Learning:**
      - **Clasificación:** Para predecir el nivel de desarrollo de un país (clasificado en cuartiles de PIB per cápita) a partir de indicadores socioeconómicos.
      - **Regresión:** Para predecir la esperanza de vida promedio al nacer con base en indicadores socioeconómicos, ambientales y de infraestructura.

### Preguntas de Negocio

El proyecto busca responder a las siguientes preguntas:

1.  ¿Los países con mayor gasto en salud muestran una mayor esperanza de vida? 
2.  ¿Los países con alta fertilidad difieren en esperanza de vida respecto a los de baja fertilidad? 
3.  ¿Los países desarrollados presentan tasas de escolarización superiores a los que están en desarrollo? 
4.  ¿Un mayor gasto en salud se asocia con un mayor desarrollo económico (medido por el PIB per cápita)? 
5.  ¿Una mayor cantidad de médicos por cada 1,000 habitantes se asocia con una menor mortalidad en menores de 5 años? 

## Datasets

El proyecto utiliza los siguientes datasets:

  - **Crudo:** `data/raw/un_country_data_raw.csv` 
  - **Procesados:**
      - `data/processed/general_info_clean.csv` 
      - `data/processed/economic_indicators_clean.csv` 
      - `data/processed/social_indicators_clean.csv` 
      - `data/processed/environment_infrastructure_clean.csv` 
      - `data/processed/complete_merged.csv` 
      - `data/processed/timeseries_merged.csv` 

## Autores

  - Leonardo Kenji Minemura Suazo
  - Sara Rocío Miranda Mateos

## Agradecimientos

Los datos utilizados en este proyecto fueron obtenidos de **UNdata**, un servicio de la **Organización de las Naciones Unidas**.