<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T10:58:03+00:00",
  "source_file": "AGENTS.md",
  "language_code": "ru"
}
-->
# AGENTS.md

## Обзор проекта

Это **Machine Learning for Beginners** — комплексная 12-недельная программа из 26 уроков, охватывающая классические концепции машинного обучения с использованием Python (в основном Scikit-learn) и R. Репозиторий разработан как ресурс для самостоятельного обучения с практическими проектами, тестами и заданиями. Каждый урок изучает концепции ML через реальные данные из различных культур и регионов мира.

Основные компоненты:
- **Образовательный контент**: 26 уроков, охватывающих введение в ML, регрессию, классификацию, кластеризацию, NLP, временные ряды и обучение с подкреплением
- **Приложение для тестов**: Приложение на Vue.js с тестами до и после уроков
- **Поддержка нескольких языков**: Автоматический перевод на более чем 40 языков через GitHub Actions
- **Двойная языковая поддержка**: Уроки доступны как на Python (Jupyter notebooks), так и на R (файлы R Markdown)
- **Обучение на основе проектов**: Каждая тема включает практические проекты и задания

## Структура репозитория

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

Каждая папка урока обычно содержит:
- `README.md` - Основной контент урока
- `notebook.ipynb` - Jupyter notebook на Python
- `solution/` - Код решений (версии на Python и R)
- `assignment.md` - Практические упражнения
- `images/` - Визуальные ресурсы

## Команды для настройки

### Для уроков на Python

Большинство уроков используют Jupyter notebooks. Установите необходимые зависимости:

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

### Для уроков на R

Уроки на R находятся в папках `solution/R/` в формате `.rmd` или `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Для приложения тестов

Приложение для тестов — это приложение на Vue.js, расположенное в папке `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Для сайта документации

Чтобы запустить документацию локально:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Рабочий процесс разработки

### Работа с ноутбуками уроков

1. Перейдите в папку урока (например, `2-Regression/1-Tools/`)
2. Откройте Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Пройдите через контент урока и упражнения
4. Проверьте решения в папке `solution/`, если потребуется

### Разработка на Python

- Уроки используют стандартные библиотеки Python для анализа данных
- Jupyter notebooks для интерактивного обучения
- Код решений доступен в папке `solution/` каждого урока

### Разработка на R

- Уроки на R представлены в формате `.rmd` (R Markdown)
- Решения находятся в подпапках `solution/R/`
- Используйте RStudio или Jupyter с ядром R для запуска ноутбуков на R

### Разработка приложения тестов

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

## Инструкции по тестированию

### Тестирование приложения тестов

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Примечание**: Это в первую очередь образовательный репозиторий. Автоматические тесты для контента уроков отсутствуют. Проверка осуществляется через:
- Выполнение упражнений урока
- Успешное выполнение ячеек ноутбуков
- Сравнение вывода с ожидаемыми результатами в решениях

## Руководство по стилю кода

### Код на Python
- Следуйте рекомендациям стиля PEP 8
- Используйте понятные и описательные имена переменных
- Добавляйте комментарии для сложных операций
- Jupyter notebooks должны содержать markdown ячейки с объяснениями концепций

### JavaScript/Vue.js (приложение тестов)
- Следуйте руководству по стилю Vue.js
- Конфигурация ESLint находится в `quiz-app/package.json`
- Запустите `npm run lint` для проверки и автоматического исправления ошибок

### Документация
- Markdown файлы должны быть четкими и хорошо структурированными
- Включайте примеры кода в огражденных блоках кода
- Используйте относительные ссылки для внутренних ссылок
- Следуйте существующим форматам оформления

## Сборка и развертывание

### Развертывание приложения тестов

Приложение тестов можно развернуть на Azure Static Web Apps:

1. **Предварительные условия**:
   - Аккаунт Azure
   - Репозиторий GitHub (уже форкнутый)

2. **Развертывание на Azure**:
   - Создайте ресурс Azure Static Web App
   - Подключите репозиторий GitHub
   - Укажите расположение приложения: `/quiz-app`
   - Укажите расположение вывода: `dist`
   - Azure автоматически создаст workflow GitHub Actions

3. **Workflow GitHub Actions**:
   - Файл workflow создается в `.github/workflows/azure-static-web-apps-*.yml`
   - Автоматически собирает и развертывает при пуше в основную ветку

### PDF документации

Создайте PDF из документации:

```bash
npm install
npm run convert
```

## Рабочий процесс перевода

**Важно**: Переводы выполняются автоматически через GitHub Actions с использованием Co-op Translator.

- Переводы создаются автоматически при внесении изменений в ветку `main`
- **НЕ переводите контент вручную** — система справляется с этим
- Workflow определен в `.github/workflows/co-op-translator.yml`
- Используются сервисы Azure AI/OpenAI для перевода
- Поддерживается более 40 языков

## Руководство по внесению изменений

### Для контентных участников

1. **Сделайте форк репозитория** и создайте ветку для изменений
2. **Внесите изменения в контент уроков**, если добавляете или обновляете уроки
3. **Не изменяйте переведенные файлы** — они создаются автоматически
4. **Протестируйте ваш код** — убедитесь, что все ячейки ноутбуков выполняются успешно
5. **Проверьте ссылки и изображения** на корректность
6. **Отправьте pull request** с четким описанием

### Руководство по pull request

- **Формат заголовка**: `[Раздел] Краткое описание изменений`
  - Пример: `[Regression] Исправление опечатки в уроке 5`
  - Пример: `[Quiz-App] Обновление зависимостей`
- **Перед отправкой**:
  - Убедитесь, что все ячейки ноутбуков выполняются без ошибок
  - Запустите `npm run lint`, если изменяли quiz-app
  - Проверьте форматирование markdown
  - Протестируйте любые новые примеры кода
- **PR должен включать**:
  - Описание изменений
  - Причину изменений
  - Скриншоты, если изменения касаются интерфейса
- **Кодекс поведения**: Следуйте [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Вам потребуется подписать Contributor License Agreement

## Структура урока

Каждый урок следует единому шаблону:

1. **Тест перед лекцией** — проверка базовых знаний
2. **Контент урока** — письменные инструкции и объяснения
3. **Демонстрация кода** — практические примеры в ноутбуках
4. **Проверка знаний** — проверка понимания в процессе
5. **Задание** — самостоятельное применение концепций
6. **Практическое задание** — расширенная практика
7. **Тест после лекции** — оценка результатов обучения

## Справочник команд

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

## Дополнительные ресурсы

- **Коллекция Microsoft Learn**: [Модули ML для начинающих](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Приложение тестов**: [Онлайн тесты](https://ff-quizzes.netlify.app/en/ml/)
- **Форум обсуждений**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Видеообзоры**: [Плейлист на YouTube](https://aka.ms/ml-beginners-videos)

## Основные технологии

- **Python**: Основной язык для уроков ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Альтернативная реализация с использованием tidyverse, tidymodels, caret
- **Jupyter**: Интерактивные ноутбуки для уроков на Python
- **R Markdown**: Документы для уроков на R
- **Vue.js 3**: Фреймворк для приложения тестов
- **Flask**: Фреймворк для веб-приложений для развертывания моделей ML
- **Docsify**: Генератор сайта документации
- **GitHub Actions**: CI/CD и автоматические переводы

## Соображения по безопасности

- **Без секретов в коде**: Никогда не добавляйте API-ключи или учетные данные в код
- **Зависимости**: Обновляйте пакеты npm и pip
- **Ввод пользователя**: Примеры веб-приложений Flask включают базовую проверку ввода
- **Чувствительные данные**: Примерные наборы данных являются публичными и не содержат конфиденциальной информации

## Устранение неполадок

### Jupyter Notebooks

- **Проблемы с ядром**: Перезапустите ядро, если ячейки зависают: Kernel → Restart
- **Ошибки импорта**: Убедитесь, что все необходимые пакеты установлены через pip
- **Проблемы с путями**: Запускайте ноутбуки из их содержащей папки

### Приложение тестов

- **npm install не работает**: Очистите кэш npm: `npm cache clean --force`
- **Конфликты портов**: Измените порт с помощью: `npm run serve -- --port 8081`
- **Ошибки сборки**: Удалите `node_modules` и переустановите: `rm -rf node_modules && npm install`

### Уроки на R

- **Пакет не найден**: Установите с помощью: `install.packages("package-name")`
- **Рендеринг RMarkdown**: Убедитесь, что пакет rmarkdown установлен
- **Проблемы с ядром**: Возможно, потребуется установить IRkernel для Jupyter

## Примечания к проекту

- Это в первую очередь **учебная программа**, а не производственный код
- Основное внимание уделяется **пониманию концепций ML** через практику
- Примеры кода ориентированы на **понятность, а не оптимизацию**
- Большинство уроков **самодостаточны** и могут быть выполнены независимо
- **Решения предоставлены**, но учащиеся должны сначала попытаться выполнить упражнения
- Репозиторий использует **Docsify** для веб-документации без этапа сборки
- **Скетчноуты** предоставляют визуальные резюме концепций
- **Поддержка нескольких языков** делает контент доступным для глобальной аудитории

---

**Отказ от ответственности**:  
Этот документ был переведен с помощью сервиса автоматического перевода [Co-op Translator](https://github.com/Azure/co-op-translator). Несмотря на наши усилия обеспечить точность, автоматические переводы могут содержать ошибки или неточности. Оригинальный документ на его родном языке следует считать авторитетным источником. Для получения критически важной информации рекомендуется профессиональный перевод человеком. Мы не несем ответственности за любые недоразумения или неправильные интерпретации, возникшие в результате использования данного перевода.