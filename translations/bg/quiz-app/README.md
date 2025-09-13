<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T00:40:07+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "bg"
}
-->
# Тестове

Тези тестове са предварителни и последващи тестове към лекциите от учебната програма за машинно обучение на https://aka.ms/ml-beginners

## Настройка на проекта

```
npm install
```

### Компилиране и автоматично презареждане за разработка

```
npm run serve
```

### Компилиране и минимизиране за продукция

```
npm run build
```

### Проверка и поправка на файлове

```
npm run lint
```

### Персонализиране на конфигурацията

Вижте [Configuration Reference](https://cli.vuejs.org/config/).

Кредити: Благодарности към оригиналната версия на приложението за тестове: https://github.com/arpan45/simple-quiz-vue

## Деплойване в Azure

Ето стъпка по стъпка ръководство, което ще ви помогне да започнете:

1. Форкнете GitHub хранилище  
Уверете се, че кодът на вашето статично уеб приложение е във вашето GitHub хранилище. Форкнете това хранилище.

2. Създайте Azure Static Web App  
- Създайте [Azure акаунт](http://azure.microsoft.com)  
- Отидете на [Azure портала](https://portal.azure.com)  
- Кликнете върху „Create a resource“ и потърсете „Static Web App“.  
- Кликнете върху „Create“.  

3. Конфигурирайте статичното уеб приложение  
- Основни настройки:  
  - Subscription: Изберете вашия Azure абонамент.  
  - Resource Group: Създайте нова ресурсна група или използвайте съществуваща.  
  - Name: Дайте име на вашето статично уеб приложение.  
  - Region: Изберете региона, който е най-близо до вашите потребители.  

- #### Детайли за деплойване:  
  - Source: Изберете „GitHub“.  
  - GitHub Account: Авторизирайте Azure да достъпи вашия GitHub акаунт.  
  - Organization: Изберете вашата GitHub организация.  
  - Repository: Изберете хранилището, което съдържа вашето статично уеб приложение.  
  - Branch: Изберете клона, от който искате да деплойвате.  

- #### Детайли за билд:  
  - Build Presets: Изберете фреймуърка, с който е създадено вашето приложение (например React, Angular, Vue и др.).  
  - App Location: Укажете папката, която съдържа кода на вашето приложение (например / ако е в корена).  
  - API Location: Ако имате API, укажете неговото местоположение (по избор).  
  - Output Location: Укажете папката, където се генерира изходният билд (например build или dist).  

4. Преглед и създаване  
Прегледайте настройките си и кликнете върху „Create“. Azure ще създаде необходимите ресурси и ще създаде GitHub Actions workflow във вашето хранилище.

5. GitHub Actions Workflow  
Azure автоматично ще създаде GitHub Actions workflow файл във вашето хранилище (.github/workflows/azure-static-web-apps-<name>.yml). Този workflow ще се грижи за процеса на билд и деплойване.

6. Мониторинг на деплойването  
Отидете в таба „Actions“ във вашето GitHub хранилище.  
Трябва да видите работещ workflow. Този workflow ще билдне и деплойне вашето статично уеб приложение в Azure.  
След като workflow приключи, вашето приложение ще бъде достъпно на предоставения Azure URL.

### Примерен файл за workflow

Ето пример за това как може да изглежда GitHub Actions workflow файлът:  
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

### Допълнителни ресурси  
- [Документация за Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Документация за GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.