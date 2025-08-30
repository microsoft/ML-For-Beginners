<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-08-29T21:39:09+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "br"
}
-->
# Questionários

Esses questionários são os quizzes pré e pós-aula para o currículo de ML em https://aka.ms/ml-beginners

## Configuração do projeto

```
npm install
```

### Compila e recarrega automaticamente para desenvolvimento

```
npm run serve
```

### Compila e minimiza para produção

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

1. Faça um fork do repositório no GitHub  
Certifique-se de que o código do seu aplicativo web estático esteja no seu repositório do GitHub. Faça um fork deste repositório.

2. Crie um Azure Static Web App  
- Crie uma [conta no Azure](http://azure.microsoft.com)  
- Acesse o [portal do Azure](https://portal.azure.com)  
- Clique em “Criar um recurso” e procure por “Static Web App”.  
- Clique em “Criar”.

3. Configure o Static Web App  
- Básico:  
  - Assinatura: Selecione sua assinatura do Azure.  
  - Grupo de Recursos: Crie um novo grupo de recursos ou use um existente.  
  - Nome: Forneça um nome para seu aplicativo web estático.  
  - Região: Escolha a região mais próxima dos seus usuários.  

- #### Detalhes de Implantação:  
  - Fonte: Selecione “GitHub”.  
  - Conta do GitHub: Autorize o Azure a acessar sua conta do GitHub.  
  - Organização: Selecione sua organização no GitHub.  
  - Repositório: Escolha o repositório que contém seu aplicativo web estático.  
  - Branch: Selecione o branch do qual deseja implantar.  

- #### Detalhes de Build:  
  - Presets de Build: Escolha o framework com o qual seu aplicativo foi construído (por exemplo, React, Angular, Vue, etc.).  
  - Localização do App: Especifique a pasta que contém o código do seu aplicativo (por exemplo, / se estiver na raiz).  
  - Localização da API: Se você tiver uma API, especifique sua localização (opcional).  
  - Localização de Saída: Especifique a pasta onde a saída do build é gerada (por exemplo, build ou dist).  

4. Revisar e Criar  
Revise suas configurações e clique em “Criar”. O Azure configurará os recursos necessários e criará um workflow do GitHub Actions no seu repositório.

5. Workflow do GitHub Actions  
O Azure criará automaticamente um arquivo de workflow do GitHub Actions no seu repositório (.github/workflows/azure-static-web-apps-<nome>.yml). Esse workflow cuidará do processo de build e implantação.

6. Monitorar a Implantação  
Acesse a aba “Actions” no seu repositório do GitHub.  
Você verá um workflow em execução. Esse workflow irá construir e implantar seu aplicativo web estático no Azure.  
Assim que o workflow for concluído, seu aplicativo estará ativo no URL fornecido pelo Azure.

### Exemplo de Arquivo de Workflow

Aqui está um exemplo de como pode ser o arquivo de workflow do GitHub Actions:  
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

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.