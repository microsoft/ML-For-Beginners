<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-12-19T13:01:04+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "ml"
}
-->
# ക്വിസുകൾ

ഈ ക്വിസുകൾ https://aka.ms/ml-beginners ലെ ML പാഠ്യപദ്ധതിക്കുള്ള പ്രീ-ലക്ചർ, പോസ്റ്റ്-ലക്ചർ ക്വിസുകളാണ്

## പ്രോജക്ട് സജ്ജീകരണം

```
npm install
```

### വികസനത്തിനായി കോമ്പൈൽ ചെയ്ത് ഹോട്ട്-റീലോഡ് ചെയ്യുന്നു

```
npm run serve
```

### ഉത്പാദനത്തിനായി കോമ്പൈൽ ചെയ്ത് മിനിഫൈ ചെയ്യുന്നു

```
npm run build
```

### ഫയലുകൾ ലിന്റ് ചെയ്ത് ശരിയാക്കുന്നു

```
npm run lint
```

### കോൺഫിഗറേഷൻ ഇഷ്ടാനുസൃതമാക്കുക

കാണുക [Configuration Reference](https://cli.vuejs.org/config/) .

ക്രെഡിറ്റുകൾ: ഈ ക്വിസ് ആപ്പിന്റെ ഒറിജിനൽ വേർഷനിന് നന്ദി: https://github.com/arpan45/simple-quiz-vue

## Azure-ലേക്ക് ഡിപ്ലോയ് ചെയ്യൽ

തുടങ്ങാൻ സഹായിക്കുന്ന ഘട്ടം-ഘട്ടമായ മാർഗ്ഗനിർദ്ദേശം:

1. GitHub റിപോസിറ്ററി ഫോർക്ക് ചെയ്യുക  
നിങ്ങളുടെ സ്റ്റാറ്റിക് വെബ് ആപ്പ് കോഡ് നിങ്ങളുടെ GitHub റിപോസിറ്ററിയിൽ ഉണ്ടെന്ന് ഉറപ്പാക്കുക. ഈ റിപോസിറ്ററി ഫോർക്ക് ചെയ്യുക.

2. Azure സ്റ്റാറ്റിക് വെബ് ആപ്പ് സൃഷ്ടിക്കുക  
- ഒരു [Azure അക്കൗണ്ട്](http://azure.microsoft.com) സൃഷ്ടിക്കുക  
- [Azure പോർട്ടൽ](https://portal.azure.com) സന്ദർശിക്കുക  
- “Create a resource” ക്ലിക്ക് ചെയ്ത് “Static Web App” തിരയുക.  
- “Create” ക്ലിക്ക് ചെയ്യുക.

3. സ്റ്റാറ്റിക് വെബ് ആപ്പ് കോൺഫിഗർ ചെയ്യുക  
- അടിസ്ഥാനങ്ങൾ: സബ്സ്ക്രിപ്ഷൻ: നിങ്ങളുടെ Azure സബ്സ്ക്രിപ്ഷൻ തിരഞ്ഞെടുക്കുക.  
- റിസോഴ്‌സ് ഗ്രൂപ്പ്: പുതിയ റിസോഴ്‌സ് ഗ്രൂപ്പ് സൃഷ്ടിക്കുക അല്ലെങ്കിൽ നിലവിലുള്ളത് ഉപയോഗിക്കുക.  
- പേര്: നിങ്ങളുടെ സ്റ്റാറ്റിക് വെബ് ആപ്പിന് ഒരു പേര് നൽകുക.  
- പ്രദേശം: നിങ്ങളുടെ ഉപയോക്താക്കൾക്ക് ഏറ്റവും അടുത്ത പ്രദേശം തിരഞ്ഞെടുക്കുക.

- #### ഡിപ്ലോയ്മെന്റ് വിശദാംശങ്ങൾ:  
- സോഴ്‌സ്: “GitHub” തിരഞ്ഞെടുക്കുക.  
- GitHub അക്കൗണ്ട്: Azure-ന് നിങ്ങളുടെ GitHub അക്കൗണ്ടിൽ പ്രവേശനം അനുവദിക്കുക.  
- ഓർഗനൈസേഷൻ: നിങ്ങളുടെ GitHub ഓർഗനൈസേഷൻ തിരഞ്ഞെടുക്കുക.  
- റിപോസിറ്ററി: നിങ്ങളുടെ സ്റ്റാറ്റിക് വെബ് ആപ്പ് ഉള്ള റിപോസിറ്ററി തിരഞ്ഞെടുക്കുക.  
- ബ്രാഞ്ച്: ഡിപ്ലോയ് ചെയ്യാൻ ആഗ്രഹിക്കുന്ന ബ്രാഞ്ച് തിരഞ്ഞെടുക്കുക.

- #### ബിൽഡ് വിശദാംശങ്ങൾ:  
- ബിൽഡ് പ്രീസെറ്റുകൾ: നിങ്ങളുടെ ആപ്പ് നിർമ്മിച്ച ഫ്രെയിംവർക്ക് തിരഞ്ഞെടുക്കുക (ഉദാ: React, Angular, Vue, മുതലായവ).  
- ആപ്പ് ലൊക്കേഷൻ: നിങ്ങളുടെ ആപ്പ് കോഡ് ഉള്ള ഫോൾഡർ വ്യക്തമാക്കുക (ഉദാ: റൂട്ട്-ൽ ആണെങ്കിൽ /).  
- API ലൊക്കേഷൻ: API ഉണ്ടെങ്കിൽ, അതിന്റെ സ്ഥലം വ്യക്തമാക്കുക (ഐച്ഛികം).  
- ഔട്ട്പുട്ട് ലൊക്കേഷൻ: ബിൽഡ് ഔട്ട്പുട്ട് സൃഷ്ടിക്കുന്ന ഫോൾഡർ വ്യക്തമാക്കുക (ഉദാ: build അല്ലെങ്കിൽ dist).

4. അവലോകനം ചെയ്ത് സൃഷ്ടിക്കുക  
നിങ്ങളുടെ ക്രമീകരണങ്ങൾ അവലോകനം ചെയ്ത് “Create” ക്ലിക്ക് ചെയ്യുക. Azure ആവശ്യമായ റിസോഴ്‌സുകൾ സജ്ജമാക്കി GitHub Actions വർക്ക്‌ഫ്ലോ നിങ്ങളുടെ റിപോസിറ്ററിയിൽ സൃഷ്ടിക്കും.

5. GitHub Actions വർക്ക്‌ഫ്ലോ  
Azure സ്വയം GitHub Actions വർക്ക്‌ഫ്ലോ ഫയൽ നിങ്ങളുടെ റിപോസിറ്ററിയിൽ (.github/workflows/azure-static-web-apps-<name>.yml) സൃഷ്ടിക്കും. ഈ വർക്ക്‌ഫ്ലോ ബിൽഡ്, ഡിപ്ലോയ്മെന്റ് പ്രക്രിയ കൈകാര്യം ചെയ്യും.

6. ഡിപ്ലോയ്മെന്റ് നിരീക്ഷിക്കുക  
നിങ്ങളുടെ GitHub റിപോസിറ്ററിയിലെ “Actions” ടാബിലേക്ക് പോകുക.  
ഒരു വർക്ക്‌ഫ്ലോ പ്രവർത്തിക്കുന്നതായി കാണണം. ഈ വർക്ക്‌ഫ്ലോ നിങ്ങളുടെ സ്റ്റാറ്റിക് വെബ് ആപ്പ് Azure-ലേക്ക് ബിൽഡ് ചെയ്ത് ഡിപ്ലോയ് ചെയ്യും.  
വർക്ക്‌ഫ്ലോ പൂർത്തിയായാൽ, നിങ്ങളുടെ ആപ്പ് നൽകിയ Azure URL-ൽ ലൈവായി കാണാം.

### ഉദാഹരണ വർക്ക്‌ഫ്ലോ ഫയൽ

GitHub Actions വർക്ക്‌ഫ്ലോ ഫയൽ എങ്ങനെ കാണാമെന്ന് ഉദാഹരണം:  
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

### അധിക സ്രോതസുകൾ  
- [Azure Static Web Apps ഡോക്യുമെന്റേഷൻ](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions ഡോക്യുമെന്റേഷൻ](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**അസൂയാ**:  
ഈ രേഖ AI വിവർത്തന സേവനം [Co-op Translator](https://github.com/Azure/co-op-translator) ഉപയോഗിച്ച് വിവർത്തനം ചെയ്തതാണ്. നാം കൃത്യതയ്ക്ക് ശ്രമിച്ചിട്ടുണ്ടെങ്കിലും, യന്ത്രം ചെയ്ത വിവർത്തനങ്ങളിൽ പിശകുകൾ അല്ലെങ്കിൽ തെറ്റുകൾ ഉണ്ടാകാമെന്ന് ദയവായി ശ്രദ്ധിക്കുക. അതിന്റെ മാതൃഭാഷയിലുള്ള യഥാർത്ഥ രേഖയാണ് പ്രാമാണികമായ ഉറവിടം എന്ന് പരിഗണിക്കേണ്ടതാണ്. നിർണായകമായ വിവരങ്ങൾക്ക്, പ്രൊഫഷണൽ മനുഷ്യ വിവർത്തനം ശുപാർശ ചെയ്യപ്പെടുന്നു. ഈ വിവർത്തനം ഉപയോഗിക്കുന്നതിൽ നിന്നുണ്ടാകുന്ന ഏതെങ്കിലും തെറ്റിദ്ധാരണകൾക്കോ വ്യാഖ്യാനക്കേടുകൾക്കോ ഞങ്ങൾ ഉത്തരവാദികളല്ല.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->