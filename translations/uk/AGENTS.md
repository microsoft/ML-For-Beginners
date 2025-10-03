<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:20:15+00:00",
  "source_file": "AGENTS.md",
  "language_code": "uk"
}
-->
# AGENTS.md

## Огляд проєкту

Це **Machine Learning for Beginners** — комплексна навчальна програма на 12 тижнів, що складається з 26 уроків, які охоплюють класичні концепції машинного навчання за допомогою Python (переважно з використанням Scikit-learn) та R. Репозиторій створений як ресурс для самостійного навчання з практичними проєктами, тестами та завданнями. Кожен урок досліджує концепції ML через реальні дані з різних культур і регіонів світу.

Основні компоненти:
- **Навчальний контент**: 26 уроків, що охоплюють введення в ML, регресію, класифікацію, кластеризацію, NLP, часові ряди та навчання з підкріпленням
- **Додаток для тестів**: Додаток для тестування на основі Vue.js з оцінюванням до та після уроків
- **Підтримка багатьох мов**: Автоматичні переклади на 40+ мов за допомогою GitHub Actions
- **Двомовна підтримка**: Уроки доступні як у Python (Jupyter notebooks), так і в R (R Markdown файли)
- **Навчання через проєкти**: Кожна тема включає практичні проєкти та завдання

## Структура репозиторію

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

Кожна папка уроку зазвичай містить:
- `README.md` - Основний контент уроку
- `notebook.ipynb` - Jupyter notebook для Python
- `solution/` - Код рішення (версії для Python і R)
- `assignment.md` - Практичні вправи
- `images/` - Візуальні ресурси

## Команди для налаштування

### Для уроків на Python

Більшість уроків використовують Jupyter notebooks. Встановіть необхідні залежності:

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

### Для уроків на R

Уроки на R знаходяться в папках `solution/R/` у форматі `.rmd` або `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Для додатка тестів

Додаток тестів — це Vue.js-додаток, розташований у папці `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Для сайту документації

Щоб запустити документацію локально:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Робочий процес розробки

### Робота з ноутбуками уроків

1. Перейдіть до папки уроку (наприклад, `2-Regression/1-Tools/`)
2. Відкрийте Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Пройдіть через контент уроку та вправи
4. Перевірте рішення в папці `solution/`, якщо потрібно

### Розробка на Python

- Уроки використовують стандартні бібліотеки Python для аналізу даних
- Jupyter notebooks для інтерактивного навчання
- Код рішень доступний у папці `solution/` кожного уроку

### Розробка на R

- Уроки на R представлені у форматі `.rmd` (R Markdown)
- Рішення знаходяться в підкаталогах `solution/R/`
- Використовуйте RStudio або Jupyter з ядром R для запуску ноутбуків на R

### Розробка додатка тестів

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

## Інструкції з тестування

### Тестування додатка тестів

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Примітка**: Це переважно навчальний репозиторій. Автоматизовані тести для контенту уроків не передбачені. Перевірка здійснюється через:
- Виконання вправ уроків
- Успішне виконання комірок ноутбуків
- Зіставлення результатів з очікуваними у рішеннях

## Рекомендації щодо стилю коду

### Код на Python
- Дотримуйтесь стилю PEP 8
- Використовуйте зрозумілі, описові назви змінних
- Додавайте коментарі до складних операцій
- У Jupyter notebooks повинні бути markdown-комірки з поясненнями концепцій

### JavaScript/Vue.js (додаток тестів)
- Дотримуйтесь стилю Vue.js
- Конфігурація ESLint у `quiz-app/package.json`
- Використовуйте `npm run lint` для перевірки та автоматичного виправлення проблем

### Документація
- Markdown-файли повинні бути зрозумілими та добре структурованими
- Включайте приклади коду у блоки з огородженням
- Використовуйте відносні посилання для внутрішніх референсів
- Дотримуйтесь існуючих форматувальних конвенцій

## Збірка та розгортання

### Розгортання додатка тестів

Додаток тестів можна розгорнути на Azure Static Web Apps:

1. **Попередні умови**:
   - Обліковий запис Azure
   - Репозиторій GitHub (вже форкований)

2. **Розгортання на Azure**:
   - Створіть ресурс Azure Static Web App
   - Підключіть до репозиторію GitHub
   - Вкажіть розташування додатка: `/quiz-app`
   - Вкажіть розташування вихідних даних: `dist`
   - Azure автоматично створює GitHub Actions workflow

3. **GitHub Actions Workflow**:
   - Файл workflow створюється у `.github/workflows/azure-static-web-apps-*.yml`
   - Автоматично збирає та розгортає при пуші в основну гілку

### PDF документація

Генерація PDF з документації:

```bash
npm install
npm run convert
```

## Робочий процес перекладу

**Важливо**: Переклади автоматизовані через GitHub Actions за допомогою Co-op Translator.

- Переклади генеруються автоматично при внесенні змін у гілку `main`
- **НЕ перекладайте контент вручну** — система це робить автоматично
- Робочий процес визначений у `.github/workflows/co-op-translator.yml`
- Використовує сервіси Azure AI/OpenAI для перекладу
- Підтримує 40+ мов

## Рекомендації для внесення змін

### Для контентних авторів

1. **Форкніть репозиторій** і створіть гілку для змін
2. **Внесіть зміни до контенту уроків**, якщо додаєте/оновлюєте уроки
3. **Не змінюйте перекладені файли** — вони генеруються автоматично
4. **Протестуйте ваш код** — переконайтеся, що всі комірки ноутбуків виконуються успішно
5. **Перевірте посилання та зображення** на коректність
6. **Надішліть pull request** з чітким описом

### Рекомендації для pull request

- **Формат заголовка**: `[Розділ] Короткий опис змін`
  - Приклад: `[Regression] Виправлення помилки в уроці 5`
  - Приклад: `[Quiz-App] Оновлення залежностей`
- **Перед надсиланням**:
  - Переконайтеся, що всі комірки ноутбуків виконуються без помилок
  - Виконайте `npm run lint`, якщо змінюєте додаток тестів
  - Перевірте форматування markdown
  - Протестуйте будь-які нові приклади коду
- **PR повинен включати**:
  - Опис змін
  - Причину змін
  - Скриншоти, якщо є зміни в UI
- **Кодекс поведінки**: Дотримуйтесь [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Вам потрібно буде підписати Contributor License Agreement

## Структура уроку

Кожен урок має послідовну структуру:

1. **Тест перед лекцією** - Перевірка базових знань
2. **Контент уроку** - Письмові інструкції та пояснення
3. **Демонстрація коду** - Практичні приклади в ноутбуках
4. **Перевірка знань** - Перевірка розуміння протягом уроку
5. **Завдання** - Самостійне застосування концепцій
6. **Домашнє завдання** - Розширена практика
7. **Тест після лекції** - Оцінка результатів навчання

## Довідник команд

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

## Додаткові ресурси

- **Колекція Microsoft Learn**: [Модулі ML для початківців](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Додаток тестів**: [Онлайн-тести](https://ff-quizzes.netlify.app/en/ml/)
- **Дошка обговорень**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Відео-огляди**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Основні технології

- **Python**: Основна мова для уроків ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Альтернативна реалізація з використанням tidyverse, tidymodels, caret
- **Jupyter**: Інтерактивні ноутбуки для уроків на Python
- **R Markdown**: Документи для уроків на R
- **Vue.js 3**: Фреймворк для додатка тестів
- **Flask**: Фреймворк для веб-додатків для розгортання моделей ML
- **Docsify**: Генератор сайту документації
- **GitHub Actions**: CI/CD та автоматизовані переклади

## Міркування щодо безпеки

- **Ніяких секретів у коді**: Ніколи не додавайте API-ключі або облікові дані
- **Залежності**: Оновлюйте пакети npm і pip
- **Введення користувача**: Приклади веб-додатків Flask включають базову перевірку введення
- **Чутливі дані**: Прикладові набори даних є публічними та не містять конфіденційної інформації

## Вирішення проблем

### Jupyter Notebooks

- **Проблеми з ядром**: Перезапустіть ядро, якщо комірки зависають: Kernel → Restart
- **Помилки імпорту**: Переконайтеся, що всі необхідні пакети встановлені через pip
- **Проблеми з шляхами**: Запускайте ноутбуки з їхньої папки

### Додаток тестів

- **npm install не працює**: Очистіть кеш npm: `npm cache clean --force`
- **Конфлікти портів**: Змініть порт за допомогою: `npm run serve -- --port 8081`
- **Помилки збірки**: Видаліть `node_modules` і перевстановіть: `rm -rf node_modules && npm install`

### Уроки на R

- **Пакет не знайдено**: Встановіть за допомогою: `install.packages("package-name")`
- **Рендеринг RMarkdown**: Переконайтеся, що пакет rmarkdown встановлений
- **Проблеми з ядром**: Можливо, потрібно встановити IRkernel для Jupyter

## Примітки щодо проєкту

- Це переважно **навчальна програма**, а не код для продакшну
- Основна увага приділяється **розумінню концепцій ML** через практику
- Приклади коду пріоритетно **зрозумілі, а не оптимізовані**
- Більшість уроків **самодостатні** і можуть бути виконані незалежно
- **Рішення надаються**, але учні повинні спочатку спробувати виконати вправи
- Репозиторій використовує **Docsify** для веб-документації без етапу збірки
- **Скетчноти** забезпечують візуальні резюме концепцій
- **Підтримка багатьох мов** робить контент доступним глобально

---

**Відмова від відповідальності**:  
Цей документ було перекладено за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, звертаємо вашу увагу, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ мовою оригіналу слід вважати авторитетним джерелом. Для отримання критично важливої інформації рекомендується професійний людський переклад. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникли внаслідок використання цього перекладу.