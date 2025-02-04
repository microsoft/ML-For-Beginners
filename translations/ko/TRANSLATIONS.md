# 수업 번역으로 기여하기

이 커리큘럼의 수업 번역을 환영합니다!
## 가이드라인

각 수업 폴더와 수업 소개 폴더에는 번역된 마크다운 파일이 들어 있습니다.

> 참고: 코드 샘플 파일의 코드는 번역하지 마세요. 번역할 것은 README, 과제, 퀴즈뿐입니다. 감사합니다!

번역된 파일은 다음과 같은 명명 규칙을 따라야 합니다:

**README._[language]_.md**

여기서 _[language]_는 ISO 639-1 표준을 따르는 두 글자의 언어 약어입니다 (예: 스페인어는 `README.es.md`, 네덜란드어는 `README.nl.md`).

**assignment._[language]_.md**

Readme와 마찬가지로 과제도 번역해 주세요.

> 중요: 이 저장소의 텍스트를 번역할 때는 기계 번역을 사용하지 마세요. 커뮤니티를 통해 번역을 검증할 것이므로, 능숙한 언어에 대해서만 번역을 자원해 주세요.

**퀴즈**

1. 번역을 퀴즈 앱에 추가하려면 여기 파일을 추가하세요: https://github.com/microsoft/ML-For-Beginners/tree/main/quiz-app/src/assets/translations, 적절한 명명 규칙을 따릅니다 (en.json, fr.json). **'true'나 'false'라는 단어는 로컬라이즈하지 마세요. 감사합니다!**

2. 퀴즈 앱의 App.vue 파일에서 드롭다운에 언어 코드를 추가하세요.

3. 퀴즈 앱의 [translations index.js 파일](https://github.com/microsoft/ML-For-Beginners/blob/main/quiz-app/src/assets/translations/index.js)을 편집하여 언어를 추가하세요.

4. 마지막으로, 번역된 README.md 파일의 모든 퀴즈 링크를 직접 번역된 퀴즈로 가리키도록 수정하세요: https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1이 https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1?loc=id로 변경됩니다.

**감사합니다**

정말로 여러분의 노고에 감사드립니다!

**면책 조항**:
이 문서는 기계 기반 AI 번역 서비스를 사용하여 번역되었습니다. 정확성을 위해 노력하지만 자동 번역에는 오류나 부정확성이 있을 수 있습니다. 원본 문서를 해당 언어로 작성된 문서를 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 우리는 책임을 지지 않습니다.