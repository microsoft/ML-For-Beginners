<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:17:32+00:00",
  "source_file": "AGENTS.md",
  "language_code": "bg"
}
-->
# AGENTS.md

## Преглед на проекта

Това е **Машинно обучение за начинаещи**, цялостна 12-седмична програма с 26 урока, обхващаща класически концепции за машинно обучение с помощта на Python (основно със Scikit-learn) и R. Репозиторият е създаден като ресурс за самостоятелно обучение с практически проекти, тестове и задачи. Всеки урок разглежда концепции за машинно обучение чрез реални данни от различни култури и региони по света.

Основни компоненти:
- **Образователно съдържание**: 26 урока, обхващащи въведение в машинното обучение, регресия, класификация, клъстеризация, NLP, времеви серии и обучение чрез подсилване
- **Приложение за тестове**: Приложение за тестове, базирано на Vue.js, с предварителни и последващи оценки за уроците
- **Поддръжка на много езици**: Автоматични преводи на над 40 езика чрез GitHub Actions
- **Двуезична поддръжка**: Уроците са достъпни както на Python (Jupyter notebooks), така и на R (R Markdown файлове)
- **Обучение чрез проекти**: Всеки раздел включва практически проекти и задачи

## Структура на репозитория

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

Всеки папка с урок обикновено съдържа:
- `README.md` - Основно съдържание на урока
- `notebook.ipynb` - Jupyter notebook за Python
- `solution/` - Решения на задачите (версии за Python и R)
- `assignment.md` - Упражнения за практика
- `images/` - Визуални ресурси

## Команди за настройка

### За уроци на Python

Повечето уроци използват Jupyter notebooks. Инсталирайте необходимите зависимости:

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

### За уроци на R

Уроците на R се намират в папките `solution/R/` като `.rmd` или `.ipynb` файлове:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### За приложението за тестове

Приложението за тестове е Vue.js приложение, разположено в директорията `quiz-app/`:

```bash
cd quiz-app
npm install
```

### За сайта с документация

За да стартирате документацията локално:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Работен процес за разработка

### Работа с учебни notebooks

1. Навигирайте до директорията на урока (например, `2-Regression/1-Tools/`)
2. Отворете Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Работете през съдържанието на урока и упражненията
4. Проверете решенията в папката `solution/`, ако е необходимо

### Разработка на Python

- Уроците използват стандартни библиотеки за обработка на данни в Python
- Jupyter notebooks за интерактивно обучение
- Кодът с решения е наличен в папката `solution/` на всеки урок

### Разработка на R

- Уроците на R са във формат `.rmd` (R Markdown)
- Решенията се намират в поддиректориите `solution/R/`
- Използвайте RStudio или Jupyter с R kernel за изпълнение на R notebooks

### Разработка на приложението за тестове

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

## Инструкции за тестване

### Тестване на приложението за тестове

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Забележка**: Това е основно репозитория за образователна програма. Няма автоматизирани тестове за съдържанието на уроците. Валидацията се извършва чрез:
- Завършване на упражненията от уроците
- Успешно изпълнение на клетките в notebook
- Проверка на резултатите спрямо очакваните в решенията

## Насоки за стил на код

### Python код
- Следвайте насоките за стил PEP 8
- Използвайте ясни и описателни имена на променливи
- Включвайте коментари за сложни операции
- Jupyter notebooks трябва да имат markdown клетки, обясняващи концепции

### JavaScript/Vue.js (Приложение за тестове)
- Следва насоките за стил на Vue.js
- Конфигурация на ESLint в `quiz-app/package.json`
- Изпълнете `npm run lint`, за да проверите и автоматично коригирате проблеми

### Документация
- Markdown файловете трябва да бъдат ясни и добре структурирани
- Включвайте примери за код в оградени блокове
- Използвайте относителни връзки за вътрешни препратки
- Следвайте съществуващите конвенции за форматиране

## Създаване и разгръщане

### Разгръщане на приложението за тестове

Приложението за тестове може да бъде разположено в Azure Static Web Apps:

1. **Предварителни условия**:
   - Azure акаунт
   - GitHub репозитория (вече клонирана)

2. **Разгръщане в Azure**:
   - Създайте ресурс Azure Static Web App
   - Свържете с GitHub репозитория
   - Задайте местоположение на приложението: `/quiz-app`
   - Задайте местоположение на изхода: `dist`
   - Azure автоматично създава GitHub Actions workflow

3. **GitHub Actions Workflow**:
   - Файлът за workflow се създава в `.github/workflows/azure-static-web-apps-*.yml`
   - Автоматично се изгражда и разгръща при push към основния branch

### PDF документация

Генерирайте PDF от документацията:

```bash
npm install
npm run convert
```

## Работен процес за превод

**Важно**: Преводите се извършват автоматично чрез GitHub Actions с помощта на Co-op Translator.

- Преводите се генерират автоматично при промени, изпратени към `main` branch
- **НЕ превеждайте съдържанието ръчно** - системата се грижи за това
- Workflow е дефиниран в `.github/workflows/co-op-translator.yml`
- Използва Azure AI/OpenAI услуги за превод
- Поддържа над 40 езика

## Насоки за принос

### За сътрудници на съдържание

1. **Клонирайте репозиторията** и създайте branch за нова функционалност
2. **Направете промени в съдържанието на урока**, ако добавяте/актуализирате уроци
3. **Не модифицирайте преведените файлове** - те се генерират автоматично
4. **Тествайте кода си** - уверете се, че всички клетки в notebook се изпълняват успешно
5. **Проверете връзките и изображенията** дали работят правилно
6. **Изпратете pull request** с ясно описание

### Насоки за pull request

- **Формат на заглавието**: `[Раздел] Кратко описание на промените`
  - Пример: `[Regression] Поправка на грешка в урок 5`
  - Пример: `[Quiz-App] Актуализация на зависимости`
- **Преди изпращане**:
  - Уверете се, че всички клетки в notebook се изпълняват без грешки
  - Изпълнете `npm run lint`, ако модифицирате quiz-app
  - Проверете форматирането на markdown
  - Тествайте всички нови примери за код
- **PR трябва да включва**:
  - Описание на промените
  - Причина за промените
  - Скрийншоти, ако има промени в интерфейса
- **Кодекс на поведение**: Следвайте [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Ще трябва да подпишете Contributor License Agreement

## Структура на урока

Всеки урок следва последователен модел:

1. **Тест преди лекцията** - Проверка на базовите знания
2. **Съдържание на урока** - Писмени инструкции и обяснения
3. **Демонстрации на код** - Практически примери в notebooks
4. **Проверка на знанията** - Проверка на разбирането по време на урока
5. **Предизвикателство** - Прилагане на концепциите самостоятелно
6. **Задача** - Разширена практика
7. **Тест след лекцията** - Оценка на резултатите от обучението

## Референция за общи команди

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

## Допълнителни ресурси

- **Microsoft Learn Collection**: [Модули за ML за начинаещи](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Приложение за тестове**: [Онлайн тестове](https://ff-quizzes.netlify.app/en/ml/)
- **Дискусионен форум**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Видео уроци**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Основни технологии

- **Python**: Основен език за уроците по машинно обучение (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Алтернативна имплементация с tidyverse, tidymodels, caret
- **Jupyter**: Интерактивни notebooks за уроците на Python
- **R Markdown**: Документи за уроците на R
- **Vue.js 3**: Framework за приложението за тестове
- **Flask**: Framework за уеб приложения за разгръщане на ML модели
- **Docsify**: Генератор на сайт за документация
- **GitHub Actions**: CI/CD и автоматични преводи

## Съображения за сигурност

- **Без тайни в кода**: Никога не качвайте API ключове или идентификационни данни
- **Зависимости**: Поддържайте npm и pip пакетите актуални
- **Потребителски вход**: Примерите за уеб приложения с Flask включват основна валидация на входа
- **Чувствителни данни**: Примерните набори от данни са публични и нечувствителни

## Отстраняване на проблеми

### Jupyter Notebooks

- **Проблеми с kernel**: Рестартирайте kernel, ако клетките блокират: Kernel → Restart
- **Грешки при импортиране**: Уверете се, че всички необходими пакети са инсталирани с pip
- **Проблеми с пътя**: Стартирайте notebooks от тяхната съдържаща директория

### Приложение за тестове

- **npm install не успява**: Изчистете кеша на npm: `npm cache clean --force`
- **Конфликти на портове**: Променете порта с: `npm run serve -- --port 8081`
- **Грешки при изграждане**: Изтрийте `node_modules` и преинсталирайте: `rm -rf node_modules && npm install`

### Уроци на R

- **Пакетът не е намерен**: Инсталирайте с: `install.packages("package-name")`
- **Рендиране на RMarkdown**: Уверете се, че пакетът rmarkdown е инсталиран
- **Проблеми с kernel**: Може да се наложи да инсталирате IRkernel за Jupyter

## Бележки, специфични за проекта

- Това е основно **учебна програма**, а не продукционен код
- Фокусът е върху **разбирането на концепциите за машинно обучение** чрез практическа работа
- Примерите за код приоритизират **яснотата пред оптимизацията**
- Повечето уроци са **самостоятелни** и могат да бъдат завършени независимо
- **Решенията са предоставени**, но обучаващите се трябва първо да опитат упражненията
- Репозиторият използва **Docsify** за уеб документация без стъпка за изграждане
- **Скетчбележки** предоставят визуални обобщения на концепциите
- **Поддръжката на много езици** прави съдържанието достъпно глобално

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.