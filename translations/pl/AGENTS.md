<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:07:34+00:00",
  "source_file": "AGENTS.md",
  "language_code": "pl"
}
-->
# AGENTS.md

## Przegląd projektu

To jest **Machine Learning dla początkujących**, kompleksowy 12-tygodniowy, 26-lekcyjny kurs obejmujący klasyczne koncepcje uczenia maszynowego z użyciem Pythona (głównie Scikit-learn) i R. Repozytorium zostało zaprojektowane jako zasób do samodzielnej nauki, zawierający projekty praktyczne, quizy i zadania. Każda lekcja bada koncepcje ML na podstawie danych z różnych kultur i regionów świata.

Kluczowe elementy:
- **Treści edukacyjne**: 26 lekcji obejmujących wprowadzenie do ML, regresję, klasyfikację, klasteryzację, NLP, szereg czasowy i uczenie przez wzmacnianie
- **Aplikacja quizowa**: Aplikacja quizowa oparta na Vue.js z ocenami przed i po lekcji
- **Wsparcie wielojęzyczne**: Automatyczne tłumaczenia na ponad 40 języków za pomocą GitHub Actions
- **Wsparcie dla dwóch języków**: Lekcje dostępne zarówno w Pythonie (notatniki Jupyter), jak i R (pliki R Markdown)
- **Nauka oparta na projektach**: Każdy temat zawiera praktyczne projekty i zadania

## Struktura repozytorium

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

Każdy folder lekcji zazwyczaj zawiera:
- `README.md` - Główna treść lekcji
- `notebook.ipynb` - Notatnik Jupyter w Pythonie
- `solution/` - Kod rozwiązania (wersje w Pythonie i R)
- `assignment.md` - Ćwiczenia praktyczne
- `images/` - Zasoby wizualne

## Polecenia konfiguracji

### Dla lekcji w Pythonie

Większość lekcji korzysta z notatników Jupyter. Zainstaluj wymagane zależności:

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

### Dla lekcji w R

Lekcje w R znajdują się w folderach `solution/R/` jako pliki `.rmd` lub `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Dla aplikacji quizowej

Aplikacja quizowa to aplikacja Vue.js znajdująca się w katalogu `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Dla strony dokumentacji

Aby uruchomić dokumentację lokalnie:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Przepływ pracy deweloperskiej

### Praca z notatnikami lekcji

1. Przejdź do katalogu lekcji (np. `2-Regression/1-Tools/`)
2. Otwórz notatnik Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Pracuj nad treścią lekcji i ćwiczeniami
4. Sprawdź rozwiązania w folderze `solution/`, jeśli to konieczne

### Rozwój w Pythonie

- Lekcje korzystają ze standardowych bibliotek do analizy danych w Pythonie
- Notatniki Jupyter do interaktywnej nauki
- Kod rozwiązania dostępny w folderze `solution/` każdej lekcji

### Rozwój w R

- Lekcje w R są w formacie `.rmd` (R Markdown)
- Rozwiązania znajdują się w podkatalogach `solution/R/`
- Użyj RStudio lub Jupyter z jądrem R, aby uruchomić notatniki w R

### Rozwój aplikacji quizowej

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

## Instrukcje testowania

### Testowanie aplikacji quizowej

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Uwaga**: To repozytorium jest głównie zasobem edukacyjnym. Nie ma zautomatyzowanych testów dla treści lekcji. Walidacja odbywa się poprzez:
- Wykonywanie ćwiczeń lekcyjnych
- Pomyślne uruchamianie komórek notatnika
- Porównywanie wyników z oczekiwanymi rezultatami w rozwiązaniach

## Wytyczne dotyczące stylu kodu

### Kod w Pythonie
- Przestrzegaj wytycznych stylu PEP 8
- Używaj jasnych, opisowych nazw zmiennych
- Dodawaj komentarze do skomplikowanych operacji
- Notatniki Jupyter powinny zawierać komórki markdown wyjaśniające koncepcje

### JavaScript/Vue.js (Aplikacja quizowa)
- Przestrzegaj wytycznych stylu Vue.js
- Konfiguracja ESLint w `quiz-app/package.json`
- Uruchom `npm run lint`, aby sprawdzić i automatycznie poprawić problemy

### Dokumentacja
- Pliki markdown powinny być jasne i dobrze zorganizowane
- Dodawaj przykłady kodu w blokach kodu
- Używaj względnych linków dla odwołań wewnętrznych
- Przestrzegaj istniejących konwencji formatowania

## Budowa i wdrożenie

### Wdrożenie aplikacji quizowej

Aplikacja quizowa może być wdrożona na Azure Static Web Apps:

1. **Wymagania wstępne**:
   - Konto Azure
   - Repozytorium GitHub (już zforkowane)

2. **Wdrożenie na Azure**:
   - Utwórz zasób Azure Static Web App
   - Połącz z repozytorium GitHub
   - Ustaw lokalizację aplikacji: `/quiz-app`
   - Ustaw lokalizację wynikową: `dist`
   - Azure automatycznie tworzy workflow GitHub Actions

3. **Workflow GitHub Actions**:
   - Plik workflow utworzony w `.github/workflows/azure-static-web-apps-*.yml`
   - Automatycznie buduje i wdraża przy pushu do głównej gałęzi

### Dokumentacja PDF

Generowanie PDF z dokumentacji:

```bash
npm install
npm run convert
```

## Przepływ pracy tłumaczeń

**Ważne**: Tłumaczenia są automatyzowane za pomocą GitHub Actions z użyciem Co-op Translator.

- Tłumaczenia są generowane automatycznie po wprowadzeniu zmian do gałęzi `main`
- **NIE tłumacz treści ręcznie** - system zajmuje się tym
- Workflow zdefiniowany w `.github/workflows/co-op-translator.yml`
- Wykorzystuje usługi Azure AI/OpenAI do tłumaczeń
- Obsługuje ponad 40 języków

## Wytyczne dotyczące wkładu

### Dla współtwórców treści

1. **Zforkuj repozytorium** i utwórz gałąź funkcji
2. **Wprowadź zmiany w treści lekcji**, jeśli dodajesz/aktualizujesz lekcje
3. **Nie modyfikuj plików tłumaczeń** - są generowane automatycznie
4. **Przetestuj swój kod** - upewnij się, że wszystkie komórki notatnika działają poprawnie
5. **Zweryfikuj linki i obrazy**, czy działają poprawnie
6. **Złóż pull request** z jasnym opisem

### Wytyczne dotyczące pull requestów

- **Format tytułu**: `[Sekcja] Krótki opis zmian`
  - Przykład: `[Regression] Popraw literówkę w lekcji 5`
  - Przykład: `[Quiz-App] Aktualizacja zależności`
- **Przed złożeniem**:
  - Upewnij się, że wszystkie komórki notatnika wykonują się bez błędów
  - Uruchom `npm run lint`, jeśli modyfikujesz quiz-app
  - Zweryfikuj formatowanie markdown
  - Przetestuj nowe przykłady kodu
- **PR musi zawierać**:
  - Opis zmian
  - Powód zmian
  - Zrzuty ekranu, jeśli zmiany dotyczą UI
- **Kodeks postępowania**: Przestrzegaj [Kodeksu postępowania Microsoft Open Source](CODE_OF_CONDUCT.md)
- **CLA**: Musisz podpisać Umowę Licencyjną Współtwórcy

## Struktura lekcji

Każda lekcja ma spójny schemat:

1. **Quiz przed wykładem** - Sprawdzenie wiedzy początkowej
2. **Treść lekcji** - Instrukcje i wyjaśnienia
3. **Demonstracje kodu** - Przykłady praktyczne w notatnikach
4. **Sprawdzanie wiedzy** - Weryfikacja zrozumienia na bieżąco
5. **Wyzwanie** - Samodzielne zastosowanie koncepcji
6. **Zadanie** - Rozszerzona praktyka
7. **Quiz po wykładzie** - Ocena wyników nauki

## Odniesienia do poleceń

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

## Dodatkowe zasoby

- **Kolekcja Microsoft Learn**: [Moduły ML dla początkujących](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplikacja quizowa**: [Quizy online](https://ff-quizzes.netlify.app/en/ml/)
- **Forum dyskusyjne**: [Dyskusje na GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Przewodniki wideo**: [Lista odtwarzania na YouTube](https://aka.ms/ml-beginners-videos)

## Kluczowe technologie

- **Python**: Główny język dla lekcji ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternatywna implementacja z użyciem tidyverse, tidymodels, caret
- **Jupyter**: Interaktywne notatniki dla lekcji w Pythonie
- **R Markdown**: Dokumenty dla lekcji w R
- **Vue.js 3**: Framework aplikacji quizowej
- **Flask**: Framework aplikacji webowych do wdrażania modeli ML
- **Docsify**: Generator strony dokumentacji
- **GitHub Actions**: CI/CD i automatyczne tłumaczenia

## Rozważania dotyczące bezpieczeństwa

- **Brak tajnych danych w kodzie**: Nigdy nie commituj kluczy API ani danych uwierzytelniających
- **Zależności**: Aktualizuj pakiety npm i pip
- **Dane wejściowe użytkownika**: Przykłady aplikacji webowych Flask zawierają podstawową walidację danych wejściowych
- **Dane wrażliwe**: Przykładowe zestawy danych są publiczne i niewrażliwe

## Rozwiązywanie problemów

### Notatniki Jupyter

- **Problemy z jądrem**: Zrestartuj jądro, jeśli komórki się zawieszają: Kernel → Restart
- **Błędy importu**: Upewnij się, że wszystkie wymagane pakiety są zainstalowane za pomocą pip
- **Problemy ze ścieżką**: Uruchamiaj notatniki z ich katalogu zawierającego

### Aplikacja quizowa

- **npm install nie działa**: Wyczyść cache npm: `npm cache clean --force`
- **Konflikty portów**: Zmień port za pomocą: `npm run serve -- --port 8081`
- **Błędy budowy**: Usuń `node_modules` i zainstaluj ponownie: `rm -rf node_modules && npm install`

### Lekcje w R

- **Brak pakietu**: Zainstaluj za pomocą: `install.packages("package-name")`
- **Renderowanie RMarkdown**: Upewnij się, że pakiet rmarkdown jest zainstalowany
- **Problemy z jądrem**: Może być konieczne zainstalowanie IRkernel dla Jupyter

## Uwagi dotyczące projektu

- To jest głównie **program nauczania**, a nie kod produkcyjny
- Skupienie na **zrozumieniu koncepcji ML** poprzez praktykę
- Przykłady kodu priorytetowo traktują **jasność nad optymalizacją**
- Większość lekcji jest **samodzielna** i może być ukończona niezależnie
- **Rozwiązania są dostępne**, ale uczniowie powinni najpierw spróbować ćwiczeń
- Repozytorium używa **Docsify** do dokumentacji webowej bez kroku budowy
- **Sketchnotes** zapewniają wizualne podsumowania koncepcji
- **Wsparcie wielojęzyczne** sprawia, że treści są dostępne globalnie

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego języku źródłowym powinien być uznawany za autorytatywne źródło. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.