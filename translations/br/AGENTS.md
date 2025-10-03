<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:06:33+00:00",
  "source_file": "AGENTS.md",
  "language_code": "br"
}
-->
# AGENTS.md

## Visão Geral do Projeto

Este é o **Machine Learning para Iniciantes**, um currículo abrangente de 12 semanas e 26 aulas que cobre conceitos clássicos de aprendizado de máquina usando Python (principalmente com Scikit-learn) e R. O repositório foi projetado como um recurso de aprendizado autônomo com projetos práticos, questionários e tarefas. Cada aula explora conceitos de aprendizado de máquina através de dados reais de diversas culturas e regiões ao redor do mundo.

Componentes principais:
- **Conteúdo Educacional**: 26 aulas cobrindo introdução ao aprendizado de máquina, regressão, classificação, clustering, NLP, séries temporais e aprendizado por reforço
- **Aplicativo de Questionário**: Aplicativo de questionário baseado em Vue.js com avaliações antes e depois das aulas
- **Suporte Multilíngue**: Traduções automáticas para mais de 40 idiomas via GitHub Actions
- **Suporte a Duas Linguagens**: Aulas disponíveis em Python (notebooks Jupyter) e R (arquivos R Markdown)
- **Aprendizado Baseado em Projetos**: Cada tópico inclui projetos práticos e tarefas

## Estrutura do Repositório

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

Cada pasta de aula geralmente contém:
- `README.md` - Conteúdo principal da aula
- `notebook.ipynb` - Notebook Jupyter em Python
- `solution/` - Código de solução (versões em Python e R)
- `assignment.md` - Exercícios práticos
- `images/` - Recursos visuais

## Comandos de Configuração

### Para Aulas em Python

A maioria das aulas utiliza notebooks Jupyter. Instale as dependências necessárias:

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

### Para Aulas em R

As aulas em R estão nas pastas `solution/R/` como arquivos `.rmd` ou `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Para o Aplicativo de Questionário

O aplicativo de questionário é uma aplicação Vue.js localizada no diretório `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Para o Site de Documentação

Para executar a documentação localmente:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Fluxo de Trabalho de Desenvolvimento

### Trabalhando com Notebooks de Aula

1. Navegue até o diretório da aula (ex.: `2-Regression/1-Tools/`)
2. Abra o notebook Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Trabalhe no conteúdo e nos exercícios da aula
4. Verifique as soluções na pasta `solution/` se necessário

### Desenvolvimento em Python

- As aulas utilizam bibliotecas padrão de ciência de dados em Python
- Notebooks Jupyter para aprendizado interativo
- Código de solução disponível na pasta `solution/` de cada aula

### Desenvolvimento em R

- As aulas em R estão no formato `.rmd` (R Markdown)
- Soluções localizadas em subdiretórios `solution/R/`
- Use RStudio ou Jupyter com kernel R para executar os notebooks em R

### Desenvolvimento do Aplicativo de Questionário

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

## Instruções de Teste

### Teste do Aplicativo de Questionário

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Nota**: Este é principalmente um repositório de currículo educacional. Não há testes automatizados para o conteúdo das aulas. A validação é feita através de:
- Conclusão dos exercícios das aulas
- Execução bem-sucedida das células dos notebooks
- Verificação dos resultados contra as soluções esperadas

## Diretrizes de Estilo de Código

### Código em Python
- Siga as diretrizes de estilo PEP 8
- Use nomes de variáveis claros e descritivos
- Inclua comentários para operações complexas
- Os notebooks Jupyter devem ter células markdown explicando os conceitos

### JavaScript/Vue.js (Aplicativo de Questionário)
- Segue o guia de estilo do Vue.js
- Configuração do ESLint em `quiz-app/package.json`
- Execute `npm run lint` para verificar e corrigir problemas automaticamente

### Documentação
- Arquivos markdown devem ser claros e bem estruturados
- Inclua exemplos de código em blocos de código delimitados
- Use links relativos para referências internas
- Siga as convenções de formatação existentes

## Construção e Implantação

### Implantação do Aplicativo de Questionário

O aplicativo de questionário pode ser implantado no Azure Static Web Apps:

1. **Pré-requisitos**:
   - Conta no Azure
   - Repositório GitHub (já bifurcado)

2. **Implantar no Azure**:
   - Crie um recurso Azure Static Web App
   - Conecte ao repositório GitHub
   - Defina a localização do aplicativo: `/quiz-app`
   - Defina a localização de saída: `dist`
   - O Azure cria automaticamente o workflow do GitHub Actions

3. **Workflow do GitHub Actions**:
   - Arquivo de workflow criado em `.github/workflows/azure-static-web-apps-*.yml`
   - Constrói e implanta automaticamente ao fazer push na branch principal

### PDF da Documentação

Gerar PDF a partir da documentação:

```bash
npm install
npm run convert
```

## Fluxo de Trabalho de Tradução

**Importante**: As traduções são automatizadas via GitHub Actions usando o Co-op Translator.

- As traduções são geradas automaticamente quando alterações são feitas na branch `main`
- **NÃO traduza o conteúdo manualmente** - o sistema cuida disso
- Workflow definido em `.github/workflows/co-op-translator.yml`
- Utiliza serviços Azure AI/OpenAI para tradução
- Suporta mais de 40 idiomas

## Diretrizes de Contribuição

### Para Contribuidores de Conteúdo

1. **Bifurque o repositório** e crie uma branch de recurso
2. **Faça alterações no conteúdo das aulas** se estiver adicionando/atualizando aulas
3. **Não modifique arquivos traduzidos** - eles são gerados automaticamente
4. **Teste seu código** - certifique-se de que todas as células dos notebooks sejam executadas com sucesso
5. **Verifique se os links e imagens** funcionam corretamente
6. **Envie um pull request** com uma descrição clara

### Diretrizes para Pull Request

- **Formato do título**: `[Seção] Breve descrição das alterações`
  - Exemplo: `[Regression] Corrigir erro de digitação na aula 5`
  - Exemplo: `[Quiz-App] Atualizar dependências`
- **Antes de enviar**:
  - Certifique-se de que todas as células dos notebooks sejam executadas sem erros
  - Execute `npm run lint` se estiver modificando o quiz-app
  - Verifique a formatação do markdown
  - Teste quaisquer novos exemplos de código
- **O PR deve incluir**:
  - Descrição das alterações
  - Motivo das alterações
  - Capturas de tela se houver alterações na interface
- **Código de Conduta**: Siga o [Código de Conduta de Código Aberto da Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: Você precisará assinar o Acordo de Licença de Contribuidor

## Estrutura das Aulas

Cada aula segue um padrão consistente:

1. **Questionário pré-aula** - Teste de conhecimento inicial
2. **Conteúdo da aula** - Instruções e explicações escritas
3. **Demonstrações de código** - Exemplos práticos em notebooks
4. **Verificações de conhecimento** - Confirme a compreensão ao longo da aula
5. **Desafio** - Aplique os conceitos de forma independente
6. **Tarefa** - Prática estendida
7. **Questionário pós-aula** - Avalie os resultados do aprendizado

## Referência de Comandos Comuns

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

## Recursos Adicionais

- **Coleção Microsoft Learn**: [Módulos de ML para Iniciantes](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplicativo de Questionário**: [Questionários online](https://ff-quizzes.netlify.app/en/ml/)
- **Fórum de Discussão**: [Discussões no GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Tutoriais em Vídeo**: [Playlist no YouTube](https://aka.ms/ml-beginners-videos)

## Tecnologias Principais

- **Python**: Linguagem principal para aulas de aprendizado de máquina (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Implementação alternativa usando tidyverse, tidymodels, caret
- **Jupyter**: Notebooks interativos para aulas em Python
- **R Markdown**: Documentos para aulas em R
- **Vue.js 3**: Framework do aplicativo de questionário
- **Flask**: Framework de aplicação web para implantação de modelos de aprendizado de máquina
- **Docsify**: Gerador de site de documentação
- **GitHub Actions**: CI/CD e traduções automatizadas

## Considerações de Segurança

- **Sem segredos no código**: Nunca comprometa chaves de API ou credenciais
- **Dependências**: Mantenha os pacotes npm e pip atualizados
- **Entrada do usuário**: Exemplos de aplicativos web Flask incluem validação básica de entrada
- **Dados sensíveis**: Os conjuntos de dados de exemplo são públicos e não sensíveis

## Solução de Problemas

### Notebooks Jupyter

- **Problemas no kernel**: Reinicie o kernel se as células travarem: Kernel → Restart
- **Erros de importação**: Certifique-se de que todos os pacotes necessários estão instalados com pip
- **Problemas de caminho**: Execute os notebooks a partir do diretório onde estão localizados

### Aplicativo de Questionário

- **npm install falha**: Limpe o cache do npm: `npm cache clean --force`
- **Conflitos de porta**: Altere a porta com: `npm run serve -- --port 8081`
- **Erros de build**: Exclua `node_modules` e reinstale: `rm -rf node_modules && npm install`

### Aulas em R

- **Pacote não encontrado**: Instale com: `install.packages("nome-do-pacote")`
- **Renderização de RMarkdown**: Certifique-se de que o pacote rmarkdown está instalado
- **Problemas no kernel**: Pode ser necessário instalar IRkernel para Jupyter

## Notas Específicas do Projeto

- Este é principalmente um **currículo de aprendizado**, não código de produção
- O foco está em **compreender conceitos de aprendizado de máquina** através de prática prática
- Exemplos de código priorizam **clareza em vez de otimização**
- A maioria das aulas é **autossuficiente** e pode ser concluída de forma independente
- **Soluções fornecidas**, mas os alunos devem tentar os exercícios primeiro
- O repositório utiliza **Docsify** para documentação web sem etapa de construção
- **Sketchnotes** fornecem resumos visuais dos conceitos
- **Suporte multilíngue** torna o conteúdo acessível globalmente

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.