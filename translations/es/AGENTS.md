<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T10:57:00+00:00",
  "source_file": "AGENTS.md",
  "language_code": "es"
}
-->
# AGENTS.md

## Resumen del Proyecto

Este es **Machine Learning para Principiantes**, un completo plan de estudios de 12 semanas y 26 lecciones que cubre conceptos clásicos de aprendizaje automático utilizando Python (principalmente con Scikit-learn) y R. El repositorio está diseñado como un recurso de aprendizaje autodidacta con proyectos prácticos, cuestionarios y tareas. Cada lección explora conceptos de ML utilizando datos reales de diversas culturas y regiones del mundo.

Componentes clave:
- **Contenido Educativo**: 26 lecciones que cubren introducción al ML, regresión, clasificación, clustering, NLP, series temporales y aprendizaje por refuerzo.
- **Aplicación de Cuestionarios**: Aplicación de cuestionarios basada en Vue.js con evaluaciones antes y después de las lecciones.
- **Soporte Multilingüe**: Traducciones automáticas a más de 40 idiomas mediante GitHub Actions.
- **Soporte Dual de Lenguajes**: Lecciones disponibles tanto en Python (notebooks Jupyter) como en R (archivos R Markdown).
- **Aprendizaje Basado en Proyectos**: Cada tema incluye proyectos prácticos y tareas.

## Estructura del Repositorio

```
ML-For-Beginners/
├── 1-Introduction/         # ML basics, history, fairness, techniques
├── 2-Regression/          # Regression models with Python/R
├── 3-Web-App/            # Flask web app for ML model deployment
├── 4-Classification/      # Classification algorithms
├── 5-Clustering/         # Clustering techniques
├── 6-NLP/               # Natural Language Processing
├── 7-TimeSeries/        # Time series forecasting
├── 8-Reinforcement/     # Reinforcement learning
├── 9-Real-World/        # Real-world ML applications
├── quiz-app/           # Vue.js quiz application
├── translations/       # Auto-generated translations
└── sketchnotes/       # Visual learning aids
```

Cada carpeta de lección típicamente contiene:
- `README.md` - Contenido principal de la lección.
- `notebook.ipynb` - Notebook Jupyter en Python.
- `solution/` - Código de solución (versiones en Python y R).
- `assignment.md` - Ejercicios prácticos.
- `images/` - Recursos visuales.

## Comandos de Configuración

### Para Lecciones en Python

La mayoría de las lecciones utilizan notebooks Jupyter. Instala las dependencias necesarias:

```bash
# Install Python 3.8+ if not already installed
python --version

# Install Jupyter
pip install jupyter

# Install common ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# For specific lessons, check lesson-specific requirements
# Example: Web App lesson
pip install flask
```

### Para Lecciones en R

Las lecciones en R están en las carpetas `solution/R/` como archivos `.rmd` o `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Para la Aplicación de Cuestionarios

La aplicación de cuestionarios es una aplicación Vue.js ubicada en el directorio `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Para el Sitio de Documentación

Para ejecutar la documentación localmente:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Flujo de Trabajo de Desarrollo

### Trabajando con los Notebooks de las Lecciones

1. Navega al directorio de la lección (por ejemplo, `2-Regression/1-Tools/`).
2. Abre el notebook Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Trabaja en el contenido y los ejercicios de la lección.
4. Consulta las soluciones en la carpeta `solution/` si es necesario.

### Desarrollo en Python

- Las lecciones utilizan bibliotecas estándar de ciencia de datos en Python.
- Notebooks Jupyter para aprendizaje interactivo.
- Código de solución disponible en la carpeta `solution/` de cada lección.

### Desarrollo en R

- Las lecciones en R están en formato `.rmd` (R Markdown).
- Soluciones ubicadas en subdirectorios `solution/R/`.
- Usa RStudio o Jupyter con el kernel de R para ejecutar los notebooks en R.

### Desarrollo de la Aplicación de Cuestionarios

```bash
cd quiz-app

# Start development server
npm run serve
# Access at http://localhost:8080

# Build for production
npm run build

# Lint and fix files
npm run lint
```

## Instrucciones de Pruebas

### Pruebas de la Aplicación de Cuestionarios

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Nota**: Este es principalmente un repositorio de currículo educativo. No hay pruebas automatizadas para el contenido de las lecciones. La validación se realiza mediante:
- Completar los ejercicios de las lecciones.
- Ejecutar las celdas de los notebooks con éxito.
- Comparar los resultados con las soluciones esperadas.

## Guías de Estilo de Código

### Código en Python
- Sigue las guías de estilo PEP 8.
- Usa nombres de variables claros y descriptivos.
- Incluye comentarios para operaciones complejas.
- Los notebooks Jupyter deben tener celdas de markdown explicando conceptos.

### JavaScript/Vue.js (Aplicación de Cuestionarios)
- Sigue la guía de estilo de Vue.js.
- Configuración de ESLint en `quiz-app/package.json`.
- Ejecuta `npm run lint` para verificar y corregir problemas automáticamente.

### Documentación
- Los archivos markdown deben ser claros y bien estructurados.
- Incluye ejemplos de código en bloques de código delimitados.
- Usa enlaces relativos para referencias internas.
- Sigue las convenciones de formato existentes.

## Construcción y Despliegue

### Despliegue de la Aplicación de Cuestionarios

La aplicación de cuestionarios puede desplegarse en Azure Static Web Apps:

1. **Requisitos Previos**:
   - Cuenta de Azure.
   - Repositorio de GitHub (ya bifurcado).

2. **Desplegar en Azure**:
   - Crea un recurso de Azure Static Web App.
   - Conéctalo al repositorio de GitHub.
   - Configura la ubicación de la aplicación: `/quiz-app`.
   - Configura la ubicación de salida: `dist`.
   - Azure crea automáticamente un flujo de trabajo de GitHub Actions.

3. **Flujo de Trabajo de GitHub Actions**:
   - Archivo de flujo de trabajo creado en `.github/workflows/azure-static-web-apps-*.yml`.
   - Se construye y despliega automáticamente al hacer push en la rama principal.

### Documentación en PDF

Genera un PDF a partir de la documentación:

```bash
npm install
npm run convert
```

## Flujo de Trabajo de Traducción

**Importante**: Las traducciones son automáticas mediante GitHub Actions utilizando Co-op Translator.

- Las traducciones se generan automáticamente cuando se realizan cambios en la rama `main`.
- **NO traduzcas manualmente el contenido**: el sistema se encarga de esto.
- Flujo de trabajo definido en `.github/workflows/co-op-translator.yml`.
- Utiliza servicios de Azure AI/OpenAI para la traducción.
- Soporta más de 40 idiomas.

## Guías para Contribuir

### Para Contribuyentes de Contenido

1. **Bifurca el repositorio** y crea una rama de características.
2. **Realiza cambios en el contenido de las lecciones** si estás agregando o actualizando lecciones.
3. **No modifiques archivos traducidos**: se generan automáticamente.
4. **Prueba tu código**: asegúrate de que todas las celdas de los notebooks se ejecuten con éxito.
5. **Verifica que los enlaces e imágenes** funcionen correctamente.
6. **Envía un pull request** con una descripción clara.

### Guías para Pull Requests

- **Formato del título**: `[Sección] Breve descripción de los cambios`.
  - Ejemplo: `[Regresión] Corregir error tipográfico en la lección 5`.
  - Ejemplo: `[Quiz-App] Actualizar dependencias`.
- **Antes de enviar**:
  - Asegúrate de que todas las celdas de los notebooks se ejecuten sin errores.
  - Ejecuta `npm run lint` si modificas la aplicación de cuestionarios.
  - Verifica el formato del markdown.
  - Prueba cualquier nuevo ejemplo de código.
- **El PR debe incluir**:
  - Descripción de los cambios.
  - Razón de los cambios.
  - Capturas de pantalla si hay cambios en la interfaz.
- **Código de Conducta**: Sigue el [Código de Conducta de Código Abierto de Microsoft](CODE_OF_CONDUCT.md).
- **CLA**: Necesitarás firmar el Acuerdo de Licencia de Contribuyente.

## Estructura de las Lecciones

Cada lección sigue un patrón consistente:

1. **Cuestionario previo a la lección** - Evalúa conocimientos iniciales.
2. **Contenido de la lección** - Instrucciones y explicaciones escritas.
3. **Demostraciones de código** - Ejemplos prácticos en notebooks.
4. **Verificaciones de conocimiento** - Confirma la comprensión a lo largo de la lección.
5. **Desafío** - Aplica los conceptos de forma independiente.
6. **Tarea** - Práctica extendida.
7. **Cuestionario posterior a la lección** - Evalúa los resultados del aprendizaje.

## Referencia de Comandos Comunes

```bash
# Python/Jupyter
jupyter notebook                    # Start Jupyter server
jupyter notebook notebook.ipynb     # Open specific notebook
pip install -r requirements.txt     # Install dependencies (where available)

# Quiz App
cd quiz-app
npm install                        # Install dependencies
npm run serve                      # Development server
npm run build                      # Production build
npm run lint                       # Lint and fix

# Documentation
docsify serve                      # Serve documentation locally
npm run convert                    # Generate PDF

# Git workflow
git checkout -b feature/my-change  # Create feature branch
git add .                         # Stage changes
git commit -m "Description"       # Commit changes
git push origin feature/my-change # Push to remote
```

## Recursos Adicionales

- **Colección de Microsoft Learn**: [Módulos de ML para Principiantes](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplicación de Cuestionarios**: [Cuestionarios en línea](https://ff-quizzes.netlify.app/en/ml/)
- **Foro de Discusión**: [Discusiones en GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Tutoriales en Video**: [Lista de reproducción en YouTube](https://aka.ms/ml-beginners-videos)

## Tecnologías Clave

- **Python**: Lenguaje principal para las lecciones de ML (Scikit-learn, Pandas, NumPy, Matplotlib).
- **R**: Implementación alternativa utilizando tidyverse, tidymodels, caret.
- **Jupyter**: Notebooks interactivos para lecciones en Python.
- **R Markdown**: Documentos para lecciones en R.
- **Vue.js 3**: Framework de la aplicación de cuestionarios.
- **Flask**: Framework de aplicaciones web para despliegue de modelos de ML.
- **Docsify**: Generador de sitios de documentación.
- **GitHub Actions**: CI/CD y traducciones automáticas.

## Consideraciones de Seguridad

- **No incluir secretos en el código**: Nunca comprometas claves API o credenciales.
- **Dependencias**: Mantén actualizados los paquetes npm y pip.
- **Entrada del usuario**: Los ejemplos de aplicaciones web Flask incluyen validación básica de entrada.
- **Datos sensibles**: Los conjuntos de datos de ejemplo son públicos y no sensibles.

## Resolución de Problemas

### Notebooks Jupyter

- **Problemas con el kernel**: Reinicia el kernel si las celdas se quedan colgadas: Kernel → Reiniciar.
- **Errores de importación**: Asegúrate de que todos los paquetes necesarios estén instalados con pip.
- **Problemas de ruta**: Ejecuta los notebooks desde su directorio contenedor.

### Aplicación de Cuestionarios

- **npm install falla**: Limpia la caché de npm: `npm cache clean --force`.
- **Conflictos de puerto**: Cambia el puerto con: `npm run serve -- --port 8081`.
- **Errores de construcción**: Elimina `node_modules` y reinstala: `rm -rf node_modules && npm install`.

### Lecciones en R

- **Paquete no encontrado**: Instálalo con: `install.packages("nombre-del-paquete")`.
- **Renderizado de RMarkdown**: Asegúrate de que el paquete rmarkdown esté instalado.
- **Problemas con el kernel**: Puede ser necesario instalar IRkernel para Jupyter.

## Notas Específicas del Proyecto

- Este es principalmente un **currículo de aprendizaje**, no código de producción.
- El enfoque está en **comprender conceptos de ML** mediante práctica práctica.
- Los ejemplos de código priorizan **claridad sobre optimización**.
- La mayoría de las lecciones son **autónomas** y pueden completarse de forma independiente.
- **Se proporcionan soluciones**, pero los estudiantes deben intentar los ejercicios primero.
- El repositorio utiliza **Docsify** para documentación web sin pasos de construcción.
- **Sketchnotes** ofrecen resúmenes visuales de conceptos.
- **Soporte multilingüe** hace que el contenido sea accesible globalmente.

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.