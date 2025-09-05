<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T13:01:42+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "sr"
}
-->
# Квизови

Ови квизови су предавања пре и после лекција за ML наставни план на https://aka.ms/ml-beginners

## Постављање пројекта

```
npm install
```

### Компилирање и брзо учитавање за развој

```
npm run serve
```

### Компилирање и минимизација за продукцију

```
npm run build
```

### Провера и исправке датотека

```
npm run lint
```

### Прилагођавање конфигурације

Погледајте [Референцу конфигурације](https://cli.vuejs.org/config/).

Заслуге: Захвалност оригиналној верзији ове апликације за квиз: https://github.com/arpan45/simple-quiz-vue

## Деплојовање на Azure

Ево корак-по-корак упутства које ће вам помоћи да започнете:

1. Форкујте GitHub репозиторијум  
Уверите се да је ваш код за статичну веб апликацију у вашем GitHub репозиторијуму. Форкујте овај репозиторијум.

2. Направите Azure статичну веб апликацију  
- Направите [Azure налог](http://azure.microsoft.com)  
- Идите на [Azure портал](https://portal.azure.com)  
- Кликните на „Create a resource“ и потражите „Static Web App“.  
- Кликните „Create“.  

3. Конфигуришите статичну веб апликацију  
- Основно:  
  - Претплата: Изаберите вашу Azure претплату.  
  - Група ресурса: Направите нову групу ресурса или користите постојећу.  
  - Назив: Унесите назив за вашу статичну веб апликацију.  
  - Регион: Изаберите регион најближи вашим корисницима.  

- #### Детаљи о деплојовању:  
  - Извор: Изаберите „GitHub“.  
  - GitHub налог: Овластите Azure да приступи вашем GitHub налогу.  
  - Организација: Изаберите вашу GitHub организацију.  
  - Репозиторијум: Изаберите репозиторијум који садржи вашу статичну веб апликацију.  
  - Грана: Изаберите грану са које желите да деплојујете.  

- #### Детаљи о изградњи:  
  - Пресети изградње: Изаберите оквир у коме је ваша апликација направљена (нпр. React, Angular, Vue, итд.).  
  - Локација апликације: Наведите фасциклу која садржи код ваше апликације (нпр. / ако је у корену).  
  - Локација API-ја: Ако имате API, наведите његову локацију (опционо).  
  - Локација излазног фолдера: Наведите фасциклу где се генерише излаз изградње (нпр. build или dist).  

4. Преглед и креирање  
Прегледајте своја подешавања и кликните „Create“. Azure ће поставити неопходне ресурсе и креирати GitHub Actions workflow у вашем репозиторијуму.

5. GitHub Actions Workflow  
Azure ће аутоматски креирати GitHub Actions workflow датотеку у вашем репозиторијуму (.github/workflows/azure-static-web-apps-<name>.yml). Овај workflow ће обрађивати процес изградње и деплојовања.

6. Праћење деплојовања  
Идите на картицу „Actions“ у вашем GitHub репозиторијуму.  
Требало би да видите workflow који се извршава. Овај workflow ће изградити и деплојовати вашу статичну веб апликацију на Azure.  
Када workflow буде завршен, ваша апликација ће бити активна на обезбеђеном Azure URL-у.

### Пример датотеке workflow-а

Ево примера како би GitHub Actions workflow датотека могла изгледати:  
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

### Додатни ресурси  
- [Документација за Azure статичне веб апликације](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Документација за GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако се трудимо да обезбедимо тачност, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на његовом изворном језику треба сматрати меродавним извором. За критичне информације препоручује се професионални превод од стране људи. Не преузимамо одговорност за било каква погрешна тумачења или неспоразуме који могу настати услед коришћења овог превода.