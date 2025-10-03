<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:43:39+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "pt"
}
-->
# Guia de Resolução de Problemas

Este guia ajuda a resolver problemas comuns ao trabalhar com o currículo de Machine Learning para Iniciantes. Se não encontrar uma solução aqui, consulte as nossas [Discussões no Discord](https://aka.ms/foundry/discord) ou [abra um problema](https://github.com/microsoft/ML-For-Beginners/issues).

## Índice

- [Problemas de Instalação](../..)
- [Problemas com o Jupyter Notebook](../..)
- [Problemas com Pacotes Python](../..)
- [Problemas com o Ambiente R](../..)
- [Problemas com a Aplicação de Questionários](../..)
- [Problemas com Dados e Caminhos de Ficheiros](../..)
- [Mensagens de Erro Comuns](../..)
- [Problemas de Desempenho](../..)
- [Ambiente e Configuração](../..)

---

## Problemas de Instalação

### Instalação do Python

**Problema**: `python: command not found`

**Solução**:
1. Instale o Python 3.8 ou superior a partir de [python.org](https://www.python.org/downloads/)
2. Verifique a instalação: `python --version` ou `python3 --version`
3. No macOS/Linux, pode ser necessário usar `python3` em vez de `python`

**Problema**: Múltiplas versões do Python a causar conflitos

**Solução**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Instalação do Jupyter

**Problema**: `jupyter: command not found`

**Solução**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problema**: O Jupyter não abre no navegador

**Solução**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Instalação do R

**Problema**: Os pacotes R não instalam

**Solução**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problema**: IRkernel não está disponível no Jupyter

**Solução**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problemas com o Jupyter Notebook

### Problemas com o Kernel

**Problema**: O kernel continua a falhar ou reiniciar

**Solução**:
1. Reinicie o kernel: `Kernel → Restart`
2. Limpe a saída e reinicie: `Kernel → Restart & Clear Output`
3. Verifique problemas de memória (consulte [Problemas de Desempenho](../..))
4. Tente executar as células individualmente para identificar o código problemático

**Problema**: Kernel Python errado selecionado

**Solução**:
1. Verifique o kernel atual: `Kernel → Change Kernel`
2. Selecione a versão correta do Python
3. Se o kernel estiver ausente, crie-o:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problema**: O kernel não inicia

**Solução**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problemas com Células do Notebook

**Problema**: As células estão a executar, mas não mostram saída

**Solução**:
1. Verifique se a célula ainda está a executar (procure o indicador `[*]`)
2. Reinicie o kernel e execute todas as células: `Kernel → Restart & Run All`
3. Verifique o console do navegador para erros de JavaScript (F12)

**Problema**: Não é possível executar células - sem resposta ao clicar em "Run"

**Solução**:
1. Verifique se o servidor Jupyter ainda está a executar no terminal
2. Atualize a página do navegador
3. Feche e reabra o notebook
4. Reinicie o servidor Jupyter

---

## Problemas com Pacotes Python

### Erros de Importação

**Problema**: `ModuleNotFoundError: No module named 'sklearn'`

**Solução**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problema**: `ImportError: cannot import name 'X' from 'sklearn'`

**Solução**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Conflitos de Versão

**Problema**: Erros de incompatibilidade de versão de pacotes

**Solução**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problema**: `pip install` falha com erros de permissão

**Solução**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problemas ao Carregar Dados

**Problema**: `FileNotFoundError` ao carregar ficheiros CSV

**Solução**:
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

## Problemas com o Ambiente R

### Instalação de Pacotes

**Problema**: A instalação de pacotes falha com erros de compilação

**Solução**:
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

**Problema**: `tidyverse` não instala

**Solução**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problemas com RMarkdown

**Problema**: O RMarkdown não renderiza

**Solução**:
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

## Problemas com a Aplicação de Questionários

### Construção e Instalação

**Problema**: `npm install` falha

**Solução**:
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

**Problema**: Porta 8080 já está em uso

**Solução**:
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

### Erros de Construção

**Problema**: `npm run build` falha

**Solução**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problema**: Erros de linting impedem a construção

**Solução**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problemas com Dados e Caminhos de Ficheiros

### Problemas com Caminhos

**Problema**: Ficheiros de dados não encontrados ao executar notebooks

**Solução**:
1. **Execute sempre os notebooks a partir do diretório onde estão localizados**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Verifique os caminhos relativos no código**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Use caminhos absolutos, se necessário**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Ficheiros de Dados em Falta

**Problema**: Ficheiros de conjuntos de dados estão em falta

**Solução**:
1. Verifique se os dados deveriam estar no repositório - a maioria dos conjuntos de dados está incluída
2. Algumas lições podem exigir o download de dados - consulte o README da lição
3. Certifique-se de que puxou as alterações mais recentes:
   ```bash
   git pull origin main
   ```

---

## Mensagens de Erro Comuns

### Erros de Memória

**Erro**: `MemoryError` ou kernel falha ao processar dados

**Solução**:
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

### Avisos de Convergência

**Aviso**: `ConvergenceWarning: Maximum number of iterations reached`

**Solução**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problemas com Gráficos

**Problema**: Gráficos não aparecem no Jupyter

**Solução**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problema**: Gráficos do Seaborn aparecem diferentes ou geram erros

**Solução**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Erros de Unicode/Codificação

**Problema**: `UnicodeDecodeError` ao ler ficheiros

**Solução**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Problemas de Desempenho

### Execução Lenta de Notebooks

**Problema**: Notebooks muito lentos para executar

**Solução**:
1. **Reinicie o kernel para liberar memória**: `Kernel → Restart`
2. **Feche notebooks não utilizados** para liberar recursos
3. **Use amostras de dados menores para testes**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Faça o perfil do seu código** para encontrar gargalos:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Uso Elevado de Memória

**Problema**: Sistema a ficar sem memória

**Solução**:
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

## Ambiente e Configuração

### Problemas com Ambientes Virtuais

**Problema**: Ambiente virtual não ativa

**Solução**:
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

**Problema**: Pacotes instalados, mas não encontrados no notebook

**Solução**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problemas com Git

**Problema**: Não é possível puxar as alterações mais recentes - conflitos de merge

**Solução**:
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

### Integração com o VS Code

**Problema**: Notebooks Jupyter não abrem no VS Code

**Solução**:
1. Instale a extensão Python no VS Code
2. Instale a extensão Jupyter no VS Code
3. Selecione o interpretador Python correto: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Reinicie o VS Code

---

## Recursos Adicionais

- **Discussões no Discord**: [Faça perguntas e partilhe soluções no canal #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Módulos de ML para Iniciantes](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Tutoriais em Vídeo**: [Playlist no YouTube](https://aka.ms/ml-beginners-videos)
- **Rastreador de Problemas**: [Reporte bugs](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Ainda com Problemas?

Se tentou as soluções acima e ainda está a enfrentar problemas:

1. **Pesquise problemas existentes**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Verifique discussões no Discord**: [Discussões no Discord](https://aka.ms/foundry/discord)
3. **Abra um novo problema**: Inclua:
   - O seu sistema operativo e versão
   - Versão do Python/R
   - Mensagem de erro (stack trace completo)
   - Passos para reproduzir o problema
   - O que já tentou

Estamos aqui para ajudar! 🚀

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.