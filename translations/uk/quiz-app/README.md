<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T13:02:22+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "uk"
}
-->
# Вікторини

Ці вікторини є попередніми та підсумковими тестами для курсу машинного навчання за адресою https://aka.ms/ml-beginners

## Налаштування проєкту

```
npm install
```

### Компіляція та автоматичне оновлення для розробки

```
npm run serve
```

### Компіляція та мінімізація для продакшну

```
npm run build
```

### Перевірка та виправлення файлів

```
npm run lint
```

### Налаштування конфігурації

Дивіться [Довідник конфігурації](https://cli.vuejs.org/config/).

Подяка: Дякуємо за оригінальну версію цього додатку для вікторин: https://github.com/arpan45/simple-quiz-vue

## Розгортання на Azure

Ось покрокова інструкція, яка допоможе вам розпочати:

1. Форкніть GitHub репозиторій  
Переконайтеся, що код вашого статичного веб-додатку знаходиться у вашому GitHub репозиторії. Форкніть цей репозиторій.

2. Створіть статичний веб-додаток на Azure  
- Створіть [обліковий запис Azure](http://azure.microsoft.com)  
- Перейдіть до [порталу Azure](https://portal.azure.com)  
- Натисніть "Створити ресурс" і знайдіть "Static Web App".  
- Натисніть "Створити".  

3. Налаштуйте статичний веб-додаток  
- Основне:  
  - Підписка: Виберіть вашу підписку Azure.  
  - Група ресурсів: Створіть нову групу ресурсів або використайте існуючу.  
  - Назва: Вкажіть назву для вашого статичного веб-додатку.  
  - Регіон: Виберіть регіон, найближчий до ваших користувачів.  

- #### Деталі розгортання:  
  - Джерело: Виберіть "GitHub".  
  - Обліковий запис GitHub: Авторизуйте Azure для доступу до вашого облікового запису GitHub.  
  - Організація: Виберіть вашу GitHub організацію.  
  - Репозиторій: Виберіть репозиторій, що містить ваш статичний веб-додаток.  
  - Гілка: Виберіть гілку, з якої ви хочете розгортати.  

- #### Деталі збірки:  
  - Пресети збірки: Виберіть фреймворк, на якому побудовано ваш додаток (наприклад, React, Angular, Vue тощо).  
  - Розташування додатку: Вкажіть папку, що містить код вашого додатку (наприклад, /, якщо він знаходиться в корені).  
  - Розташування API: Якщо у вас є API, вкажіть його розташування (опціонально).  
  - Розташування вихідних даних: Вкажіть папку, де генерується вихідний код збірки (наприклад, build або dist).  

4. Перевірте та створіть  
Перевірте ваші налаштування та натисніть "Створити". Azure налаштує необхідні ресурси та створить GitHub Actions workflow у вашому репозиторії.

5. GitHub Actions Workflow  
Azure автоматично створить файл GitHub Actions workflow у вашому репозиторії (.github/workflows/azure-static-web-apps-<name>.yml). Цей workflow буде обробляти процес збірки та розгортання.

6. Моніторинг розгортання  
Перейдіть до вкладки "Actions" у вашому GitHub репозиторії.  
Ви повинні побачити запущений workflow. Цей workflow збере та розгорне ваш статичний веб-додаток на Azure.  
Після завершення workflow ваш додаток буде доступний за наданою URL-адресою Azure.

### Приклад файлу Workflow

Ось приклад того, як може виглядати файл GitHub Actions workflow:  
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

### Додаткові ресурси  
- [Документація Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Документація GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Відмова від відповідальності**:  
Цей документ був перекладений за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, будь ласка, майте на увазі, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ на його рідній мові слід вважати авторитетним джерелом. Для критичної інформації рекомендується професійний людський переклад. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникають внаслідок використання цього перекладу.