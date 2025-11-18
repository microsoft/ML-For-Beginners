<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-11-18T18:12:16+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "pcm"
}
-->
# Quizzes

Dis quizzes na di pre- and post-lecture quizzes for di ML curriculum wey dey https://aka.ms/ml-beginners

## Project setup

```
npm install
```

### E dey compile and dey reload for development

```
npm run serve
```

### E dey compile and dey minify for production

```
npm run build
```

### E dey check code and dey fix files

```
npm run lint
```

### Customize configuration

Check [Configuration Reference](https://cli.vuejs.org/config/).

Credits: Big thanks to di original version of dis quiz app: https://github.com/arpan45/simple-quiz-vue

## How to deploy am for Azure

Dis na step-by-step guide to help you start:

1. Fork di GitHub Repository
Make sure say your static web app code dey inside your GitHub repository. Fork dis repository.

2. Create Azure Static Web App
- Create [Azure account](http://azure.microsoft.com)
- Go di [Azure portal](https://portal.azure.com) 
- Click “Create a resource” and search for “Static Web App”.
- Click “Create”.

3. Configure di Static Web App
- Basics: Subscription: Choose your Azure subscription.
- Resource Group: Create new resource group or use di one wey you don already get.
- Name: Give name for your static web app.
- Region: Choose di region wey dey near your users.

- #### Deployment Details:
- Source: Choose “GitHub”.
- GitHub Account: Allow Azure to access your GitHub account.
- Organization: Choose your GitHub organization.
- Repository: Pick di repository wey get your static web app.
- Branch: Choose di branch wey you wan deploy from.

- #### Build Details:
- Build Presets: Pick di framework wey your app dey use (e.g., React, Angular, Vue, etc.).
- App Location: Put di folder wey get your app code (e.g., / if e dey for root).
- API Location: If you get API, put di location (optional).
- Output Location: Put di folder wey di build output dey generate (e.g., build or dist).

4. Review and Create
Check your settings and click “Create”. Azure go set up di resources wey you need and go create GitHub Actions workflow for your repository.

5. GitHub Actions Workflow
Azure go automatically create GitHub Actions workflow file for your repository (.github/workflows/azure-static-web-apps-<name>.yml). Dis workflow go handle di build and deployment process.

6. Monitor di Deployment
Go di “Actions” tab for your GitHub repository.
You go see workflow wey dey run. Dis workflow go build and deploy your static web app for Azure.
Once di workflow finish, your app go dey live for di Azure URL wey dem give you.

### Example Workflow File

Dis na example of how di GitHub Actions workflow file fit look like:
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

### Additional Resources
- [Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis document don dey translate wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even though we dey try make am accurate, abeg sabi say automated translations fit get mistake or no correct well. Di original document for im native language na di main correct source. For important information, e better make una use professional human translation. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because of dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->