<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-12-19T12:59:59+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "kn"
}
-->
# ನಿಮ್ಮ ML ಮಾದರಿಯನ್ನು ಬಳಸಲು ವೆಬ್ ಅಪ್ಲಿಕೇಶನ್ ನಿರ್ಮಿಸಿ

ಈ ಪಠ್ಯಕ್ರಮದ ವಿಭಾಗದಲ್ಲಿ, ನೀವು ಅನ್ವಯಿತ ML ವಿಷಯವನ್ನು ಪರಿಚಯಿಸಿಕೊಳ್ಳುತ್ತೀರಿ: ನಿಮ್ಮ Scikit-learn ಮಾದರಿಯನ್ನು ಫೈಲ್ ಆಗಿ ಉಳಿಸುವುದು, ಅದನ್ನು ವೆಬ್ ಅಪ್ಲಿಕೇಶನ್‌ನಲ್ಲಿ ಭವಿಷ್ಯವಾಣಿ ಮಾಡಲು ಬಳಸಬಹುದು. ಮಾದರಿ ಉಳಿಸಿದ ನಂತರ, ನೀವು ಅದನ್ನು Flask ನಲ್ಲಿ ನಿರ್ಮಿಸಲಾದ ವೆಬ್ ಅಪ್ಲಿಕೇಶನ್‌ನಲ್ಲಿ ಹೇಗೆ ಬಳಸುವುದು ಎಂದು ಕಲಿಯುತ್ತೀರಿ. ಮೊದಲು, ನೀವು UFO ದೃಶ್ಯಗಳ ಬಗ್ಗೆ ಇರುವ ಕೆಲವು ಡೇಟಾ ಬಳಸಿ ಮಾದರಿಯನ್ನು ರಚಿಸುವಿರಿ! ನಂತರ, ನೀವು ಸೆಕೆಂಡುಗಳ ಸಂಖ್ಯೆ, ಅಕ್ಷಾಂಶ ಮತ್ತು ರೇಖಾಂಶ ಮೌಲ್ಯವನ್ನು ನಮೂದಿಸಲು ಅನುಮತಿಸುವ ವೆಬ್ ಅಪ್ಲಿಕೇಶನ್ ಅನ್ನು ನಿರ್ಮಿಸುವಿರಿ, ಅದು ಯಾವ ದೇಶ UFO ನೋಡಿದ ಎಂದು ಭವಿಷ್ಯವಾಣಿ ಮಾಡುತ್ತದೆ.

![UFO Parking](../../../translated_images/kn/ufo.9e787f5161da9d4d.webp)

ಚಿತ್ರವನ್ನು <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ಮೈಕೆಲ್ ಹೆರೆನ್</a> ಅವರು <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ಅನ್ಸ್ಪ್ಲ್ಯಾಶ್</a> ನಲ್ಲಿ ತೆಗೆದಿದ್ದಾರೆ

## ಪಾಠಗಳು

1. [ವೆಬ್ ಅಪ್ಲಿಕೇಶನ್ ನಿರ್ಮಿಸಿ](1-Web-App/README.md)

## ಕ್ರೆಡಿಟ್‌ಗಳು

"ವೆಬ್ ಅಪ್ಲಿಕೇಶನ್ ನಿರ್ಮಿಸಿ" ಅನ್ನು ♥️ [ಜೆನ್ ಲೂಪರ್](https://twitter.com/jenlooper) ರವರು ಬರೆಯಲಾಗಿದೆ.

♥️ ಪ್ರಶ್ನೋತ್ತರಗಳನ್ನು ರೋಹನ್ ರಾಜ್ ರವರು ಬರೆಯಲಾಗಿದೆ.

ಡೇಟಾಸೆಟ್ ಅನ್ನು [ಕಾಗಲ್](https://www.kaggle.com/NUFORC/ufo-sightings) ನಿಂದ ಪಡೆದಿದೆ.

ವೆಬ್ ಅಪ್ಲಿಕೇಶನ್ ವಾಸ್ತುಶಿಲ್ಪವನ್ನು ಭಾಗಶಃ [ಈ ಲೇಖನ](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) ಮತ್ತು ಅಭಿನವ್ ಸಾಗರ್ ಅವರ [ಈ ರೆಪೊ](https://github.com/abhinavsagar/machine-learning-deployment) ಸೂಚಿಸಿದ್ದಾರೆ.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ಅಸ್ವೀಕರಣ**:  
ಈ ದಸ್ತಾವೇಜು AI ಅನುವಾದ ಸೇವೆ [Co-op Translator](https://github.com/Azure/co-op-translator) ಬಳಸಿ ಅನುವಾದಿಸಲಾಗಿದೆ. ನಾವು ನಿಖರತೆಯಿಗಾಗಿ ಪ್ರಯತ್ನಿಸುತ್ತಿದ್ದರೂ, ಸ್ವಯಂಚಾಲಿತ ಅನುವಾದಗಳಲ್ಲಿ ತಪ್ಪುಗಳು ಅಥವಾ ಅಸತ್ಯತೆಗಳು ಇರಬಹುದು ಎಂದು ದಯವಿಟ್ಟು ಗಮನಿಸಿ. ಮೂಲ ಭಾಷೆಯಲ್ಲಿರುವ ಮೂಲ ದಸ್ತಾವೇಜನ್ನು ಅಧಿಕೃತ ಮೂಲವೆಂದು ಪರಿಗಣಿಸಬೇಕು. ಮಹತ್ವದ ಮಾಹಿತಿಗಾಗಿ ವೃತ್ತಿಪರ ಮಾನವ ಅನುವಾದವನ್ನು ಶಿಫಾರಸು ಮಾಡಲಾಗುತ್ತದೆ. ಈ ಅನುವಾದ ಬಳಕೆಯಿಂದ ಉಂಟಾಗುವ ಯಾವುದೇ ತಪ್ಪು ಅರ್ಥಮಾಡಿಕೊಳ್ಳುವಿಕೆ ಅಥವಾ ತಪ್ಪು ವಿವರಣೆಗಳಿಗೆ ನಾವು ಹೊಣೆಗಾರರಾಗುವುದಿಲ್ಲ.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->