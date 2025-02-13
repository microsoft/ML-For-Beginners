# Questionários

Esses questionários são os questionários de pré e pós-aula para o currículo de ML em https://aka.ms/ml-beginners

## Configuração do projeto

```
npm install
```

### Compila e recarrega rapidamente para desenvolvimento

```
npm run serve
```

### Compila e minifica para produção

```
npm run build
```

### Verifica e corrige arquivos

```
npm run lint
```

### Personalizar configuração

Veja [Referência de Configuração](https://cli.vuejs.org/config/).

Créditos: Agradecimentos à versão original deste aplicativo de questionário: https://github.com/arpan45/simple-quiz-vue

## Implantando no Azure

Aqui está um guia passo a passo para ajudá-lo a começar:

1. Fork o repositório do GitHub
Certifique-se de que o código do seu aplicativo web estático esteja no seu repositório do GitHub. Faça um fork deste repositório.

2. Crie um Aplicativo Web Estático do Azure
- Crie uma [conta no Azure](http://azure.microsoft.com)
- Vá para o [portal do Azure](https://portal.azure.com) 
- Clique em “Criar um recurso” e procure por “Aplicativo Web Estático”.
- Clique em “Criar”.

3. Configure o Aplicativo Web Estático
- Básicos: Assinatura: Selecione sua assinatura do Azure.
- Grupo de Recursos: Crie um novo grupo de recursos ou use um existente.
- Nome: Forneça um nome para seu aplicativo web estático.
- Região: Escolha a região mais próxima dos seus usuários.

- #### Detalhes da Implantação:
- Fonte: Selecione “GitHub”.
- Conta do GitHub: Autorize o Azure a acessar sua conta do GitHub.
- Organização: Selecione sua organização do GitHub.
- Repositório: Escolha o repositório que contém seu aplicativo web estático.
- Branch: Selecione o branch do qual você deseja implantar.

- #### Detalhes da Construção:
- Predefinições de Construção: Escolha o framework com o qual seu aplicativo foi construído (por exemplo, React, Angular, Vue, etc.).
- Localização do Aplicativo: Especifique a pasta que contém o código do seu aplicativo (por exemplo, / se estiver na raiz).
- Localização da API: Se você tiver uma API, especifique sua localização (opcional).
- Localização da Saída: Especifique a pasta onde a saída da construção é gerada (por exemplo, build ou dist).

4. Revise e Crie
Revise suas configurações e clique em “Criar”. O Azure configurará os recursos necessários e criará um fluxo de trabalho do GitHub Actions em seu repositório.

5. Fluxo de Trabalho do GitHub Actions
O Azure criará automaticamente um arquivo de fluxo de trabalho do GitHub Actions em seu repositório (.github/workflows/azure-static-web-apps-<name>.yml). Este fluxo de trabalho lidará com o processo de construção e implantação.

6. Monitore a Implantação
Vá para a aba “Ações” em seu repositório do GitHub.
Você deve ver um fluxo de trabalho em execução. Este fluxo de trabalho irá construir e implantar seu aplicativo web estático no Azure.
Assim que o fluxo de trabalho for concluído, seu aplicativo estará ao vivo na URL do Azure fornecida.

### Exemplo de Arquivo de Fluxo de Trabalho

Aqui está um exemplo de como o arquivo de fluxo de trabalho do GitHub Actions pode parecer:
name: Azure Static Web Apps CI/CD
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### Recursos Adicionais
- [Documentação do Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [Documentação do GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**Aviso**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em sua língua nativa deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional feita por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações erradas decorrentes do uso desta tradução.