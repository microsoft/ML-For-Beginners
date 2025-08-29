<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-08-29T17:48:16+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "pa"
}
-->
# ਕਵਿਜ਼

ਇਹ ਕਵਿਜ਼ ML ਪਾਠਕ੍ਰਮ ਲਈ ਲੈਕਚਰ ਤੋਂ ਪਹਿਲਾਂ ਅਤੇ ਬਾਅਦ ਦੇ ਕਵਿਜ਼ ਹਨ: https://aka.ms/ml-beginners

## ਪ੍ਰੋਜੈਕਟ ਸੈਟਅੱਪ

```
npm install
```

### ਡਿਵੈਲਪਮੈਂਟ ਲਈ ਕੰਪਾਇਲ ਅਤੇ ਹੌਟ-ਰੀਲੋਡ ਕਰਦਾ ਹੈ

```
npm run serve
```

### ਪ੍ਰੋਡਕਸ਼ਨ ਲਈ ਕੰਪਾਇਲ ਅਤੇ ਮਿਨੀਫਾਈ ਕਰਦਾ ਹੈ

```
npm run build
```

### ਫਾਈਲਾਂ ਨੂੰ ਲਿੰਟ ਅਤੇ ਫਿਕਸ ਕਰਦਾ ਹੈ

```
npm run lint
```

### ਕਨਫਿਗਰੇਸ਼ਨ ਨੂੰ ਕਸਟਮਾਈਜ਼ ਕਰੋ

[ਕਨਫਿਗਰੇਸ਼ਨ ਰੈਫਰੈਂਸ](https://cli.vuejs.org/config/) ਵੇਖੋ।

ਸ਼੍ਰੇਯ: ਇਸ ਕਵਿਜ਼ ਐਪ ਦੇ ਮੂਲ ਸੰਸਕਰਣ ਲਈ ਧੰਨਵਾਦ: https://github.com/arpan45/simple-quiz-vue

## ਐਜ਼ਰ 'ਤੇ ਡਿਪਲੌਇ ਕਰਨਾ

ਇੱਥੇ ਇੱਕ ਕਦਮ-ਦਰ-ਕਦਮ ਗਾਈਡ ਹੈ ਜੋ ਤੁਹਾਨੂੰ ਸ਼ੁਰੂਆਤ ਕਰਨ ਵਿੱਚ ਮਦਦ ਕਰੇਗੀ:

1. GitHub ਰਿਪੋਜ਼ਟਰੀ ਨੂੰ ਫੋਰਕ ਕਰੋ  
ਸੁਨਿਸ਼ਚਿਤ ਕਰੋ ਕਿ ਤੁਹਾਡੀ ਸਟੈਟਿਕ ਵੈੱਬ ਐਪ ਕੋਡ ਤੁਹਾਡੇ GitHub ਰਿਪੋਜ਼ਟਰੀ ਵਿੱਚ ਹੈ। ਇਸ ਰਿਪੋਜ਼ਟਰੀ ਨੂੰ ਫੋਰਕ ਕਰੋ।

2. ਇੱਕ ਐਜ਼ਰ ਸਟੈਟਿਕ ਵੈੱਬ ਐਪ ਬਣਾਓ  
- ਇੱਕ [ਐਜ਼ਰ ਅਕਾਊਂਟ](http://azure.microsoft.com) ਬਣਾਓ।  
- [ਐਜ਼ਰ ਪੋਰਟਲ](https://portal.azure.com) 'ਤੇ ਜਾਓ।  
- "Create a resource" 'ਤੇ ਕਲਿੱਕ ਕਰੋ ਅਤੇ "Static Web App" ਖੋਜੋ।  
- "Create" 'ਤੇ ਕਲਿੱਕ ਕਰੋ।  

3. ਸਟੈਟਿਕ ਵੈੱਬ ਐਪ ਨੂੰ ਕਨਫਿਗਰ ਕਰੋ  
- #### ਬੇਸਿਕਸ:  
  - Subscription: ਆਪਣੀ ਐਜ਼ਰ ਸਬਸਕ੍ਰਿਪਸ਼ਨ ਚੁਣੋ।  
  - Resource Group: ਇੱਕ ਨਵਾਂ ਰਿਸੋਰਸ ਗਰੁੱਪ ਬਣਾਓ ਜਾਂ ਮੌਜੂਦਾ ਵਰਤੋ।  
  - Name: ਆਪਣੀ ਸਟੈਟਿਕ ਵੈੱਬ ਐਪ ਲਈ ਇੱਕ ਨਾਮ ਦਿਓ।  
  - Region: ਆਪਣੇ ਯੂਜ਼ਰਾਂ ਦੇ ਸਭ ਤੋਂ ਨੇੜੇ ਖੇਤਰ ਦੀ ਚੋਣ ਕਰੋ।  

- #### ਡਿਪਲੌਇਮੈਂਟ ਵੇਰਵੇ:  
  - Source: "GitHub" ਚੁਣੋ।  
  - GitHub Account: ਐਜ਼ਰ ਨੂੰ ਤੁਹਾਡੇ GitHub ਅਕਾਊਂਟ ਤੱਕ ਪਹੁੰਚ ਦੀ ਆਗਿਆ ਦਿਓ।  
  - Organization: ਆਪਣੀ GitHub ਸੰਗਠਨ ਚੁਣੋ।  
  - Repository: ਉਹ ਰਿਪੋਜ਼ਟਰੀ ਚੁਣੋ ਜਿਸ ਵਿੱਚ ਤੁਹਾਡੀ ਸਟੈਟਿਕ ਵੈੱਬ ਐਪ ਹੈ।  
  - Branch: ਉਹ ਬ੍ਰਾਂਚ ਚੁਣੋ ਜਿਸ ਤੋਂ ਤੁਸੀਂ ਡਿਪਲੌਇ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ।  

- #### ਬਿਲਡ ਵੇਰਵੇ:  
  - Build Presets: ਉਹ ਫਰੇਮਵਰਕ ਚੁਣੋ ਜਿਸ ਨਾਲ ਤੁਹਾਡੀ ਐਪ ਬਣਾਈ ਗਈ ਹੈ (ਜਿਵੇਂ React, Angular, Vue ਆਦਿ)।  
  - App Location: ਉਹ ਫੋਲਡਰ ਦਰਸਾਓ ਜਿਸ ਵਿੱਚ ਤੁਹਾਡੀ ਐਪ ਕੋਡ ਹੈ (ਜਿਵੇਂ ਕਿ / ਜੇਕਰ ਇਹ ਰੂਟ ਵਿੱਚ ਹੈ)।  
  - API Location: ਜੇ ਤੁਹਾਡੇ ਕੋਲ API ਹੈ, ਤਾਂ ਇਸ ਦਾ ਸਥਾਨ ਦਰਸਾਓ (ਵਿਕਲਪਿਕ)।  
  - Output Location: ਉਹ ਫੋਲਡਰ ਦਰਸਾਓ ਜਿੱਥੇ ਬਿਲਡ ਆਉਟਪੁੱਟ ਤਿਆਰ ਹੁੰਦੀ ਹੈ (ਜਿਵੇਂ ਕਿ build ਜਾਂ dist)।  

4. ਸਮੀਖਿਆ ਕਰੋ ਅਤੇ ਬਣਾਓ  
ਆਪਣੀ ਸੈਟਿੰਗ ਦੀ ਸਮੀਖਿਆ ਕਰੋ ਅਤੇ "Create" 'ਤੇ ਕਲਿੱਕ ਕਰੋ। ਐਜ਼ਰ ਜ਼ਰੂਰੀ ਰਿਸੋਰਸ ਸੈਟਅੱਪ ਕਰੇਗਾ ਅਤੇ ਤੁਹਾਡੇ ਰਿਪੋਜ਼ਟਰੀ ਵਿੱਚ ਇੱਕ GitHub Actions ਵਰਕਫਲੋ ਬਣਾਏਗਾ।  

5. GitHub Actions ਵਰਕਫਲੋ  
ਐਜ਼ਰ ਤੁਹਾਡੇ ਰਿਪੋਜ਼ਟਰੀ ਵਿੱਚ ਆਟੋਮੈਟਿਕ ਤੌਰ 'ਤੇ ਇੱਕ GitHub Actions ਵਰਕਫਲੋ ਫਾਈਲ ਬਣਾਏਗਾ (.github/workflows/azure-static-web-apps-<name>.yml)। ਇਹ ਵਰਕਫਲੋ ਬਿਲਡ ਅਤੇ ਡਿਪਲੌਇਮੈਂਟ ਪ੍ਰਕਿਰਿਆ ਨੂੰ ਸੰਭਾਲੇਗਾ।  

6. ਡਿਪਲੌਇਮੈਂਟ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ  
ਤੁਹਾਡੇ GitHub ਰਿਪੋਜ਼ਟਰੀ ਵਿੱਚ "Actions" ਟੈਬ 'ਤੇ ਜਾਓ।  
ਤੁਹਾਨੂੰ ਇੱਕ ਵਰਕਫਲੋ ਚਲਦਾ ਹੋਇਆ ਦਿਖਾਈ ਦੇਵੇਗਾ। ਇਹ ਵਰਕਫਲੋ ਤੁਹਾਡੀ ਸਟੈਟਿਕ ਵੈੱਬ ਐਪ ਨੂੰ ਐਜ਼ਰ 'ਤੇ ਬਿਲਡ ਅਤੇ ਡਿਪਲੌਇ ਕਰੇਗਾ।  
ਜਦੋਂ ਵਰਕਫਲੋ ਪੂਰਾ ਹੋ ਜਾਵੇ, ਤੁਹਾਡੀ ਐਪ ਪ੍ਰਦਾਨ ਕੀਤੇ ਐਜ਼ਰ URL 'ਤੇ ਲਾਈਵ ਹੋਵੇਗੀ।  

### ਉਦਾਹਰਣ ਵਰਕਫਲੋ ਫਾਈਲ

ਇੱਥੇ ਇੱਕ ਉਦਾਹਰਣ ਹੈ ਕਿ GitHub Actions ਵਰਕਫਲੋ ਫਾਈਲ ਕਿਵੇਂ ਲੱਗ ਸਕਦੀ ਹੈ:  
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

### ਵਾਧੂ ਸਰੋਤ  
- [ਐਜ਼ਰ ਸਟੈਟਿਕ ਵੈੱਬ ਐਪਸ ਡੌਕਯੂਮੈਂਟੇਸ਼ਨ](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions ਡੌਕਯੂਮੈਂਟੇਸ਼ਨ](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**ਅਸਵੀਕਾਰਨਾ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚਨਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।