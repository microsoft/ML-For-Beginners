<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:06:05+00:00",
  "source_file": "AGENTS.md",
  "language_code": "pt"
}
-->
# AGENTS.md

## Visão Geral do Projeto

Este é o **Machine Learning para Principiantes**, um currículo abrangente de 12 semanas e 26 lições que cobre conceitos clássicos de machine learning utilizando Python (principalmente com Scikit-learn) e R. O repositório foi concebido como um recurso de aprendizagem autodidata, com projetos práticos, questionários e tarefas. Cada lição explora conceitos de ML através de dados reais provenientes de várias culturas e regiões do mundo.

Componentes principais:
- **Conteúdo Educacional**: 26 lições que abrangem introdução ao ML, regressão, classificação, clustering, NLP, séries temporais e aprendizagem por reforço
- **Aplicação de Questionários**: Aplicação de questionários baseada em Vue.js com avaliações antes e depois das lições
- **Suporte Multilíngue**: Traduções automáticas para mais de 40 idiomas via GitHub Actions
- **Suporte a Duas Linguagens**: Lições disponíveis tanto em Python (notebooks Jupyter) quanto em R (ficheiros R Markdown)
- **Aprendizagem Baseada em Projetos**: Cada tópico inclui projetos práticos e tarefas

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

Cada pasta de lição geralmente contém:
- `README.md` - Conteúdo principal da lição
- `notebook.ipynb` - Notebook Jupyter em Python
- `solution/` - Código de solução (versões em Python e R)
- `assignment.md` - Exercícios práticos
- `images/` - Recursos visuais

## Comandos de Configuração

### Para Lições em Python

A maioria das lições utiliza notebooks Jupyter. Instale as dependências necessárias:

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

### Para Lições em R

As lições em R estão nas pastas `solution/R/` como ficheiros `.rmd` ou `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Para a Aplicação de Questionários

A aplicação de questionários é uma aplicação Vue.js localizada no diretório `quiz-app/`:

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

### Trabalhar com Notebooks de Lição

1. Navegue até ao diretório da lição (ex.: `2-Regression/1-Tools/`)
2. Abra o notebook Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Trabalhe no conteúdo e nos exercícios da lição
4. Consulte as soluções na pasta `solution/` se necessário

### Desenvolvimento em Python

- As lições utilizam bibliotecas padrão de ciência de dados em Python
- Notebooks Jupyter para aprendizagem interativa
- Código de solução disponível na pasta `solution/` de cada lição

### Desenvolvimento em R

- As lições em R estão no formato `.rmd` (R Markdown)
- Soluções localizadas em subdiretórios `solution/R/`
- Utilize RStudio ou Jupyter com kernel R para executar os notebooks em R

### Desenvolvimento da Aplicação de Questionários

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

### Teste da Aplicação de Questionários

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Nota**: Este é principalmente um repositório de currículo educacional. Não há testes automatizados para o conteúdo das lições. A validação é feita através de:
- Conclusão dos exercícios das lições
- Execução bem-sucedida das células dos notebooks
- Verificação dos resultados contra as soluções esperadas

## Diretrizes de Estilo de Código

### Código em Python
- Siga as diretrizes de estilo PEP 8
- Utilize nomes de variáveis claros e descritivos
- Inclua comentários para operações complexas
- Os notebooks Jupyter devem conter células markdown explicando os conceitos

### JavaScript/Vue.js (Aplicação de Questionários)
- Siga o guia de estilo Vue.js
- Configuração ESLint em `quiz-app/package.json`
- Execute `npm run lint` para verificar e corrigir automaticamente problemas

### Documentação
- Os ficheiros markdown devem ser claros e bem estruturados
- Inclua exemplos de código em blocos de código delimitados
- Utilize links relativos para referências internas
- Siga as convenções de formatação existentes

## Construção e Implementação

### Implementação da Aplicação de Questionários

A aplicação de questionários pode ser implementada no Azure Static Web Apps:

1. **Pré-requisitos**:
   - Conta Azure
   - Repositório GitHub (já bifurcado)

2. **Implementar no Azure**:
   - Crie um recurso Azure Static Web App
   - Conecte ao repositório GitHub
   - Defina a localização da aplicação: `/quiz-app`
   - Defina a localização de saída: `dist`
   - O Azure cria automaticamente o workflow do GitHub Actions

3. **Workflow do GitHub Actions**:
   - Ficheiro de workflow criado em `.github/workflows/azure-static-web-apps-*.yml`
   - Constrói e implementa automaticamente ao fazer push para a branch principal

### PDF da Documentação

Gerar PDF a partir da documentação:

```bash
npm install
npm run convert
```

## Fluxo de Trabalho de Tradução

**Importante**: As traduções são automatizadas via GitHub Actions utilizando o Co-op Translator.

- As traduções são geradas automaticamente quando alterações são feitas na branch `main`
- **NÃO traduza o conteúdo manualmente** - o sistema trata disso
- Workflow definido em `.github/workflows/co-op-translator.yml`
- Utiliza serviços Azure AI/OpenAI para tradução
- Suporta mais de 40 idiomas

## Diretrizes de Contribuição

### Para Contribuidores de Conteúdo

1. **Bifurque o repositório** e crie uma branch de funcionalidade
2. **Faça alterações no conteúdo das lições** se estiver a adicionar/atualizar lições
3. **Não modifique ficheiros traduzidos** - eles são gerados automaticamente
4. **Teste o seu código** - certifique-se de que todas as células dos notebooks são executadas com sucesso
5. **Verifique se os links e imagens** funcionam corretamente
6. **Submeta um pull request** com uma descrição clara

### Diretrizes para Pull Requests

- **Formato do título**: `[Seção] Breve descrição das alterações`
  - Exemplo: `[Regression] Corrigir erro na lição 5`
  - Exemplo: `[Quiz-App] Atualizar dependências`
- **Antes de submeter**:
  - Certifique-se de que todas as células dos notebooks são executadas sem erros
  - Execute `npm run lint` se estiver a modificar quiz-app
  - Verifique a formatação markdown
  - Teste quaisquer novos exemplos de código
- **O PR deve incluir**:
  - Descrição das alterações
  - Razão para as alterações
  - Capturas de ecrã se houver alterações na interface
- **Código de Conduta**: Siga o [Código de Conduta de Código Aberto da Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: Será necessário assinar o Acordo de Licença de Contribuidor

## Estrutura das Lições

Cada lição segue um padrão consistente:

1. **Questionário pré-aula** - Testar conhecimento inicial
2. **Conteúdo da lição** - Instruções e explicações escritas
3. **Demonstrações de código** - Exemplos práticos em notebooks
4. **Verificações de conhecimento** - Confirmar compreensão ao longo da lição
5. **Desafio** - Aplicar conceitos de forma independente
6. **Tarefa** - Prática estendida
7. **Questionário pós-aula** - Avaliar resultados de aprendizagem

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

- **Coleção Microsoft Learn**: [Módulos de ML para Principiantes](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplicação de Questionários**: [Questionários online](https://ff-quizzes.netlify.app/en/ml/)
- **Fórum de Discussão**: [Discussões no GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Tutoriais em Vídeo**: [Playlist no YouTube](https://aka.ms/ml-beginners-videos)

## Tecnologias Principais

- **Python**: Linguagem principal para lições de ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Implementação alternativa utilizando tidyverse, tidymodels, caret
- **Jupyter**: Notebooks interativos para lições em Python
- **R Markdown**: Documentos para lições em R
- **Vue.js 3**: Framework da aplicação de questionários
- **Flask**: Framework de aplicação web para implementação de modelos de ML
- **Docsify**: Gerador de sites de documentação
- **GitHub Actions**: CI/CD e traduções automatizadas

## Considerações de Segurança

- **Sem segredos no código**: Nunca comprometa chaves de API ou credenciais
- **Dependências**: Mantenha os pacotes npm e pip atualizados
- **Entrada do utilizador**: Exemplos de aplicações web Flask incluem validação básica de entrada
- **Dados sensíveis**: Os conjuntos de dados de exemplo são públicos e não sensíveis

## Resolução de Problemas

### Notebooks Jupyter

- **Problemas com o kernel**: Reinicie o kernel se as células ficarem pendentes: Kernel → Reiniciar
- **Erros de importação**: Certifique-se de que todos os pacotes necessários estão instalados com pip
- **Problemas de caminho**: Execute os notebooks a partir do diretório onde estão localizados

### Aplicação de Questionários

- **npm install falha**: Limpe a cache do npm: `npm cache clean --force`
- **Conflitos de porta**: Altere a porta com: `npm run serve -- --port 8081`
- **Erros de construção**: Elimine `node_modules` e reinstale: `rm -rf node_modules && npm install`

### Lições em R

- **Pacote não encontrado**: Instale com: `install.packages("nome-do-pacote")`
- **Renderização de RMarkdown**: Certifique-se de que o pacote rmarkdown está instalado
- **Problemas com o kernel**: Pode ser necessário instalar IRkernel para Jupyter

## Notas Específicas do Projeto

- Este é principalmente um **currículo de aprendizagem**, não código de produção
- O foco está em **compreender conceitos de ML** através de prática prática
- Os exemplos de código priorizam **clareza em vez de otimização**
- A maioria das lições é **autossuficiente** e pode ser concluída de forma independente
- **Soluções fornecidas**, mas os alunos devem tentar os exercícios primeiro
- O repositório utiliza **Docsify** para documentação web sem etapa de construção
- **Sketchnotes** fornecem resumos visuais dos conceitos
- **Suporte multilíngue** torna o conteúdo acessível globalmente

---

**Aviso**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante notar que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.