<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:35:27+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "es"
}
-->
# Guía de Solución de Problemas

Esta guía te ayudará a resolver problemas comunes al trabajar con el plan de estudios de Machine Learning para Principiantes. Si no encuentras una solución aquí, consulta nuestras [Discusiones en Discord](https://aka.ms/foundry/discord) o [abre un problema](https://github.com/microsoft/ML-For-Beginners/issues).

## Tabla de Contenidos

- [Problemas de Instalación](../..)
- [Problemas con Jupyter Notebook](../..)
- [Problemas con Paquetes de Python](../..)
- [Problemas con el Entorno R](../..)
- [Problemas con la Aplicación de Cuestionarios](../..)
- [Problemas con Datos y Rutas de Archivos](../..)
- [Mensajes de Error Comunes](../..)
- [Problemas de Rendimiento](../..)
- [Entorno y Configuración](../..)

---

## Problemas de Instalación

### Instalación de Python

**Problema**: `python: command not found`

**Solución**:
1. Instala Python 3.8 o superior desde [python.org](https://www.python.org/downloads/)
2. Verifica la instalación: `python --version` o `python3 --version`
3. En macOS/Linux, puede que necesites usar `python3` en lugar de `python`

**Problema**: Conflictos entre múltiples versiones de Python

**Solución**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Instalación de Jupyter

**Problema**: `jupyter: command not found`

**Solución**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problema**: Jupyter no se abre en el navegador

**Solución**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Instalación de R

**Problema**: Los paquetes de R no se instalan

**Solución**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problema**: IRkernel no está disponible en Jupyter

**Solución**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problemas con Jupyter Notebook

### Problemas con el Kernel

**Problema**: El kernel sigue fallando o reiniciándose

**Solución**:
1. Reinicia el kernel: `Kernel → Restart`
2. Borra la salida y reinicia: `Kernel → Restart & Clear Output`
3. Verifica problemas de memoria (consulta [Problemas de Rendimiento](../..))
4. Ejecuta las celdas individualmente para identificar el código problemático

**Problema**: Kernel de Python incorrecto seleccionado

**Solución**:
1. Verifica el kernel actual: `Kernel → Change Kernel`
2. Selecciona la versión correcta de Python
3. Si falta el kernel, créalo:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problema**: El kernel no se inicia

**Solución**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problemas con las Celdas del Notebook

**Problema**: Las celdas se ejecutan pero no muestran salida

**Solución**:
1. Verifica si la celda sigue ejecutándose (busca el indicador `[*]`)
2. Reinicia el kernel y ejecuta todas las celdas: `Kernel → Restart & Run All`
3. Revisa la consola del navegador para errores de JavaScript (F12)

**Problema**: No se pueden ejecutar celdas - no hay respuesta al hacer clic en "Run"

**Solución**:
1. Verifica si el servidor de Jupyter sigue ejecutándose en la terminal
2. Refresca la página del navegador
3. Cierra y vuelve a abrir el notebook
4. Reinicia el servidor de Jupyter

---

## Problemas con Paquetes de Python

### Errores de Importación

**Problema**: `ModuleNotFoundError: No module named 'sklearn'`

**Solución**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problema**: `ImportError: cannot import name 'X' from 'sklearn'`

**Solución**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Conflictos de Versiones

**Problema**: Errores de incompatibilidad de versiones de paquetes

**Solución**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problema**: `pip install` falla con errores de permisos

**Solución**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problemas al Cargar Datos

**Problema**: `FileNotFoundError` al cargar archivos CSV

**Solución**:
```python
import os
# Check current working directory
print(os.getcwd())

# Use relative paths from notebook location
df = pd.read_csv('../../data/filename.csv')

# Or use absolute paths
df = pd.read_csv('/full/path/to/data/filename.csv')
```

---

## Problemas con el Entorno R

### Instalación de Paquetes

**Problema**: La instalación de paquetes falla con errores de compilación

**Solución**:
```r
# Install binary version (Windows/macOS)
install.packages("package-name", type = "binary")

# Update R to latest version if packages require it
# Check R version
R.version.string

# Install system dependencies (Linux)
# For Ubuntu/Debian, in terminal:
# sudo apt-get install r-base-dev
```

**Problema**: `tidyverse` no se instala

**Solución**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problemas con RMarkdown

**Problema**: RMarkdown no se renderiza

**Solución**:
```r
# Install/update rmarkdown
install.packages("rmarkdown")

# Install pandoc if needed
install.packages("pandoc")

# For PDF output, install tinytex
install.packages("tinytex")
tinytex::install_tinytex()
```

---

## Problemas con la Aplicación de Cuestionarios

### Construcción e Instalación

**Problema**: `npm install` falla

**Solución**:
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install

# If still fails, try with legacy peer deps
npm install --legacy-peer-deps
```

**Problema**: El puerto 8080 ya está en uso

**Solución**:
```bash
# Use different port
npm run serve -- --port 8081

# Or find and kill process using port 8080
# On Linux/macOS:
lsof -ti:8080 | xargs kill -9

# On Windows:
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### Errores de Construcción

**Problema**: `npm run build` falla

**Solución**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problema**: Errores de linting que impiden la construcción

**Solución**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problemas con Datos y Rutas de Archivos

### Problemas de Rutas

**Problema**: Archivos de datos no encontrados al ejecutar notebooks

**Solución**:
1. **Ejecuta siempre los notebooks desde su directorio contenedor**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Verifica las rutas relativas en el código**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Usa rutas absolutas si es necesario**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Archivos de Datos Faltantes

**Problema**: Faltan archivos de conjuntos de datos

**Solución**:
1. Verifica si los datos deberían estar en el repositorio - la mayoría de los conjuntos de datos están incluidos
2. Algunas lecciones pueden requerir descargar datos - revisa el README de la lección
3. Asegúrate de haber descargado los últimos cambios:
   ```bash
   git pull origin main
   ```

---

## Mensajes de Error Comunes

### Errores de Memoria

**Error**: `MemoryError` o el kernel falla al procesar datos

**Solución**:
```python
# Load data in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)

# Or read only needed columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])

# Free memory when done
del large_dataframe
import gc
gc.collect()
```

### Advertencias de Convergencia

**Advertencia**: `ConvergenceWarning: Maximum number of iterations reached`

**Solución**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problemas al Graficar

**Problema**: Las gráficas no se muestran en Jupyter

**Solución**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problema**: Las gráficas de Seaborn se ven diferentes o generan errores

**Solución**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Errores de Unicode/Codificación

**Problema**: `UnicodeDecodeError` al leer archivos

**Solución**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Problemas de Rendimiento

### Ejecución Lenta de Notebooks

**Problema**: Los notebooks son muy lentos al ejecutarse

**Solución**:
1. **Reinicia el kernel para liberar memoria**: `Kernel → Restart`
2. **Cierra notebooks no utilizados** para liberar recursos
3. **Usa muestras de datos más pequeñas para pruebas**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Perfila tu código** para encontrar cuellos de botella:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Uso Alto de Memoria

**Problema**: El sistema se queda sin memoria

**Solución**:
```python
# Check memory usage
df.info(memory_usage='deep')

# Optimize data types
df['column'] = df['column'].astype('int32')  # Instead of int64

# Drop unnecessary columns
df = df[['col1', 'col2']]  # Keep only needed columns

# Process in batches
for batch in np.array_split(df, 10):
    process(batch)
```

---

## Entorno y Configuración

### Problemas con Entornos Virtuales

**Problema**: El entorno virtual no se activa

**Solución**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Check if activated (should show venv name in prompt)
which python  # Should point to venv python
```

**Problema**: Los paquetes instalados no se encuentran en el notebook

**Solución**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problemas con Git

**Problema**: No se pueden descargar los últimos cambios - conflictos de fusión

**Solución**:
```bash
# Stash your changes
git stash

# Pull latest
git pull origin main

# Reapply your changes
git stash pop

# If conflicts, resolve manually or:
git checkout --theirs path/to/file  # Take remote version
git checkout --ours path/to/file    # Keep your version
```

### Integración con VS Code

**Problema**: Los notebooks de Jupyter no se abren en VS Code

**Solución**:
1. Instala la extensión de Python en VS Code
2. Instala la extensión de Jupyter en VS Code
3. Selecciona el intérprete de Python correcto: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Reinicia VS Code

---

## Recursos Adicionales

- **Discusiones en Discord**: [Haz preguntas y comparte soluciones en el canal #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Módulos de ML para Principiantes](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Tutoriales en Video**: [Lista de Reproducción en YouTube](https://aka.ms/ml-beginners-videos)
- **Seguimiento de Problemas**: [Reporta errores](https://github.com/microsoft/ML-For-Beginners/issues)

---

## ¿Sigues Teniendo Problemas?

Si has intentado las soluciones anteriores y sigues teniendo problemas:

1. **Busca problemas existentes**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Consulta las discusiones en Discord**: [Discusiones en Discord](https://aka.ms/foundry/discord)
3. **Abre un nuevo problema**: Incluye:
   - Tu sistema operativo y versión
   - Versión de Python/R
   - Mensaje de error (traza completa)
   - Pasos para reproducir el problema
   - Lo que ya has intentado

¡Estamos aquí para ayudarte! 🚀

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.