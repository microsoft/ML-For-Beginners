<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-12-19T14:41:03+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ml"
}
-->
# ഹോട്ടൽ റിവ്യൂകളുമായി അനുഭവം വിശകലനം

ഇപ്പോൾ നിങ്ങൾ ഡാറ്റാസെറ്റ് വിശദമായി പരിശോധിച്ചിരിക്കുന്നു, കോളങ്ങൾ ഫിൽട്ടർ ചെയ്ത് പിന്നീട് ഡാറ്റാസെറ്റിൽ NLP സാങ്കേതിക വിദ്യകൾ ഉപയോഗിച്ച് ഹോട്ടലുകളെക്കുറിച്ചുള്ള പുതിയ洞察ങ്ങൾ നേടാനുള്ള സമയം.

## [പ്രീ-ലെക്ചർ ക്വിസ്](https://ff-quizzes.netlify.app/en/ml/)

### ഫിൽട്ടറിംഗ് & അനുഭവം വിശകലന പ്രവർത്തനങ്ങൾ

നിങ്ങൾ ശ്രദ്ധിച്ചിരിക്കാം, ഡാറ്റാസെറ്റിൽ ചില പ്രശ്നങ്ങളുണ്ട്. ചില കോളങ്ങൾ ഉപകാരമില്ലാത്ത വിവരങ്ങൾ നിറഞ്ഞിരിക്കുന്നു, മറ്റുള്ളവ തെറ്റായതായി തോന്നുന്നു. ശരിയാണെങ്കിൽ, അവ എങ്ങനെ കണക്കാക്കിയതെന്ന് വ്യക്തമല്ല, നിങ്ങളുടെ സ്വന്തം കണക്കുകൾ ഉപയോഗിച്ച് സ്വതന്ത്രമായി സ്ഥിരീകരിക്കാൻ കഴിയില്ല.

## അഭ്യാസം: കുറച്ച് കൂടുതൽ ഡാറ്റ പ്രോസസ്സിംഗ്

ഡാറ്റ കുറച്ച് കൂടുതൽ ശുദ്ധമാക്കുക. പിന്നീട് ഉപകാരപ്രദമായ കോളങ്ങൾ ചേർക്കുക, മറ്റു കോളങ്ങളിലെ മൂല്യങ്ങൾ മാറ്റുക, ചില കോളങ്ങൾ പൂർണ്ണമായും ഒഴിവാക്കുക.

1. പ്രാഥമിക കോളം പ്രോസസ്സിംഗ്

   1. `lat` and `lng` ഒഴിവാക്കുക

   2. `Hotel_Address` മൂല്യങ്ങൾ താഴെപ്പറയുന്ന മൂല്യങ്ങളാൽ മാറ്റുക (അഡ്രസിൽ നഗരം, രാജ്യം ഒരുപോലെ ഉണ്ടെങ്കിൽ, അത് നഗരവും രാജ്യവും മാത്രം മാറ്റുക).

      ഡാറ്റാസെറ്റിലെ ഏകദേശം ഈ നഗരങ്ങളും രാജ്യങ്ങളും മാത്രമാണ്:

      ആംസ്റ്റർഡാം, നെതർലാൻഡ്‌സ്

      ബാഴ്‌സലോണ, സ്പെയിൻ

      ലണ്ടൻ, യുണൈറ്റഡ് കിംഗ്‌ഡം

      മിലാൻ, ഇറ്റലി

      പാരിസ്, ഫ്രാൻസ്

      വിയന്ന, ഓസ്ട്രിയ

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # എല്ലാ വിലാസങ്ങളും ചുരുക്കിയ, കൂടുതൽ ഉപയോഗപ്രദമായ രൂപത്തിലേക്ക് മാറ്റുക
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # value_counts() ന്റെ മൊത്തം സംഖ്യ റിവ്യൂകളുടെ മൊത്തം എണ്ണത്തോടൊപ്പം ചേർന്നിരിക്കണം
      print(df["Hotel_Address"].value_counts())
      ```

      ഇപ്പോൾ നിങ്ങൾക്ക് രാജ്യ തലത്തിലുള്ള ഡാറ്റ ചോദിക്കാം:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. ഹോട്ടൽ മെറ്റാ-റിവ്യൂ കോളങ്ങൾ പ്രോസസ്സ് ചെയ്യുക

  1. `Additional_Number_of_Scoring` ഒഴിവാക്കുക

  2. `Total_Number_of_Reviews` ആ ഹോട്ടലിനുള്ള ഡാറ്റാസെറ്റിൽ ഉള്ള റിവ്യൂകളുടെ മൊത്തം എണ്ണം കൊണ്ട് മാറ്റുക

  3. `Average_Score` നമ്മുടെ സ്വന്തം കണക്കാക്കിയ സ്കോറിൽ മാറ്റുക

  ```python
  # `Additional_Number_of_Scoring` ഒഴിവാക്കുക
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # `Total_Number_of_Reviews` ഉം `Average_Score` ഉം നമ്മുടെ സ്വന്തം കണക്കാക്കിയ മൂല്യങ്ങളാൽ മാറ്റി വയ്ക്കുക
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. റിവ്യൂ കോളങ്ങൾ പ്രോസസ്സ് ചെയ്യുക

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review` ഒഴിവാക്കുക

   2. `Reviewer_Score`, `Negative_Review`, and `Positive_Review` അതുപോലെ തന്നെ സൂക്ഷിക്കുക,
     
   3. ഇപ്പോൾ `Tags` സൂക്ഷിക്കുക

     - അടുത്ത സെക്ഷനിൽ ടാഗുകളിൽ കൂടുതൽ ഫിൽട്ടറിംഗ് പ്രവർത്തനങ്ങൾ നടത്തും, പിന്നീട് ടാഗുകൾ ഒഴിവാക്കും

4. റിവ്യൂവറിന്റെ കോളങ്ങൾ പ്രോസസ്സ് ചെയ്യുക

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` ഒഴിവാക്കുക
  
  2. `Reviewer_Nationality` സൂക്ഷിക്കുക

### ടാഗ് കോളങ്ങൾ

`Tag` കോളം പ്രശ്നകരമാണ്, കാരണം അത് ഒരു ലിസ്റ്റ് (ടെക്സ്റ്റ് രൂപത്തിൽ) കോളത്തിൽ സൂക്ഷിച്ചിരിക്കുന്നു. ദുർഭാഗ്യവശാൽ ഈ കോളത്തിലെ ഉപവിഭാഗങ്ങളുടെ ക്രമവും എണ്ണം എല്ലായ്പ്പോഴും ഒരുപോലെ അല്ല. മനുഷ്യന് ശരിയായ വാചകങ്ങൾ കണ്ടെത്താൻ ബുദ്ധിമുട്ടാണ്, കാരണം 515,000 വരികളുണ്ട്, 1427 ഹോട്ടലുകൾ ഉണ്ട്, ഓരോ ഹോട്ടലിനും റിവ്യൂവറിന് തിരഞ്ഞെടുക്കാനുള്ള വ്യത്യസ്ത ഓപ്ഷനുകൾ ഉണ്ട്. ഇവിടെ NLP പ്രഭാവം കാണിക്കുന്നു. നിങ്ങൾ ടെക്സ്റ്റ് സ്കാൻ ചെയ്ത് ഏറ്റവും സാധാരണമായ വാചകങ്ങൾ കണ്ടെത്തി എണ്ണാം.

ദുർഭാഗ്യവശാൽ, ഞങ്ങൾ ഏകവാക്കുകൾക്ക് അല്ല, ബഹുവാക്ക് വാചകങ്ങൾക്കാണ് (ഉദാ: *ബിസിനസ് ട്രിപ്പ്*) താൽപര്യം. അത്തരം വാചകങ്ങളുടെ ഫ്രീക്വൻസി ഡിസ്‌ട്രിബ്യൂഷൻ ആൽഗോരിതം 6762646 വാക്കുകളിൽ നടത്തുന്നത് വളരെ സമയം എടുക്കും, പക്ഷേ ഡാറ്റ നോക്കാതെ അത് അനിവാര്യമായ ചെലവാണെന്ന് തോന്നും. ഇവിടെ എക്സ്പ്ലോറേറ്ററി ഡാറ്റ അനാലിസിസ് സഹായിക്കുന്നു, കാരണം നിങ്ങൾ ടാഗുകളുടെ സാമ്പിൾ കണ്ടിട്ടുണ്ട്, ഉദാ: `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, നിങ്ങൾ ടാഗുകൾ കുറയ്ക്കാൻ കഴിയുമോ എന്ന് ചോദിക്കാൻ തുടങ്ങാം. ഭാഗ്യവശാൽ, കഴിയും - പക്ഷേ ആദ്യം ടാഗുകൾ കണ്ടെത്താൻ ചില ഘട്ടങ്ങൾ പാലിക്കണം.

### ടാഗുകൾ ഫിൽട്ടർ ചെയ്യൽ

ഡാറ്റാസെറ്റിന്റെ ലക്ഷ്യം അനുഭവം കൂട്ടിച്ചേർക്കലും മികച്ച ഹോട്ടൽ തിരഞ്ഞെടുക്കാൻ സഹായിക്കുന്ന കോളങ്ങൾ ചേർക്കലുമാണ് (സ്വന്തം ആവശ്യത്തിനോ ക്ലയന്റിനോ ഹോട്ടൽ ശുപാർശ ബോട്ട് നിർമ്മിക്കാൻ). ടാഗുകൾ ഫൈനൽ ഡാറ്റാസെറ്റിൽ ഉപകാരപ്രദമാണോ എന്ന് ചോദിക്കണം. ഇതാ ഒരു വ്യാഖ്യാനം (മറ്റു ആവശ്യങ്ങൾക്കായി വേറെ ടാഗുകൾ വേണമെങ്കിൽ അവ തിരഞ്ഞെടുക്കാം):

1. യാത്രയുടെ തരം പ്രസക്തമാണ്, അത് സൂക്ഷിക്കണം
2. അതിഥി ഗ്രൂപ്പിന്റെ തരം പ്രധാനമാണ്, അത് സൂക്ഷിക്കണം
3. അതിഥി താമസിച്ച മുറി, സ്യൂട്ട്, സ്റ്റുഡിയോ എന്നിവ പ്രസക്തമല്ല (എല്ലാ ഹോട്ടലുകളും അടിസ്ഥാനപരമായി ഒരുപോലെ മുറികൾ ഉണ്ട്)
4. റിവ്യൂ സമർപ്പിച്ച ഉപകരണം പ്രസക്തമല്ല
5. റിവ്യൂവറുടെ താമസിച്ച രാത്രികളുടെ എണ്ണം *പ്രസക്തമായിരിക്കാം* (നീണ്ട താമസങ്ങൾ ഹോട്ടൽ ഇഷ്ടപ്പെടുന്നതായി കാണിച്ചാൽ), പക്ഷേ അത് സംശയാസ്പദവും പ്രസക്തമല്ലാത്തതും ആണ്

സംഗ്രഹത്തിൽ, **2 തരത്തിലുള്ള ടാഗുകൾ സൂക്ഷിച്ച് മറ്റെല്ലാം ഒഴിവാക്കുക**.

ആദ്യം, ടാഗുകൾ നല്ല ഫോർമാറ്റിൽ വരുന്നതുവരെ എണ്ണാൻ ആഗ്രഹിക്കില്ല, അതിനാൽ സ്ക്വയർ ബ്രാക്കറ്റുകളും ഉദ്ധരണികളും നീക്കം ചെയ്യണം. ഇത് പല രീതികളിൽ ചെയ്യാം, പക്ഷേ വേഗതയുള്ളതായിരിക്കും തിരഞ്ഞെടുക്കുക, കാരണം വലിയ ഡാറ്റ പ്രോസസ്സ് ചെയ്യാൻ സമയം എടുക്കും. ഭാഗ്യവശാൽ, pandas ഇതിന് എളുപ്പം മാർഗ്ഗം നൽകുന്നു.

```Python
# തുറക്കുന്നും അടയ്ക്കുന്നും ബ്രാക്കറ്റുകൾ നീക്കം ചെയ്യുക
df.Tags = df.Tags.str.strip("[']")
# എല്ലാ ഉദ്ധരണികളും നീക്കം ചെയ്യുക
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

ഓരോ ടാഗും ഇങ്ങനെ മാറും: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`.

അടുത്തതായി ഒരു പ്രശ്നം കണ്ടെത്തുന്നു. ചില റിവ്യൂകൾക്ക് 5 കോളങ്ങൾ ഉണ്ട്, ചിലക്ക് 3, ചിലക്ക് 6. ഇത് ഡാറ്റാസെറ്റ് സൃഷ്ടിച്ച രീതിയുടെ ഫലമാണ്, പരിഹരിക്കാൻ ബുദ്ധിമുട്ടാണ്. ഓരോ വാചകത്തിന്റെയും ഫ്രീക്വൻസി എണ്ണണം, പക്ഷേ അവ വ്യത്യസ്ത ക്രമത്തിലാണ്, അതിനാൽ എണ്ണൽ തെറ്റായിരിക്കാം, ഹോട്ടലിന് അവകാശപ്പെട്ട ടാഗ് ലഭിക്കാതിരിക്കാം.

പകരം, വ്യത്യസ്ത ക്രമം നമ്മുടെ ഗുണം ചെയ്യും, കാരണം ഓരോ ടാഗും ബഹുവാക്ക് ആണ്, കൂടാതെ കോമയാൽ വേർതിരിച്ചിരിക്കുന്നു! ഏറ്റവും ലളിതമായ മാർഗ്ഗം 6 താൽക്കാലിക കോളങ്ങൾ സൃഷ്ടിച്ച് ഓരോ ടാഗും അതിന്റെ ക്രമാനുസൃത കോളത്തിൽ ഇടുക എന്നതാണ്. പിന്നീട് ആ 6 കോളങ്ങൾ ഒന്നായി ചേർത്ത് `value_counts()` മെത്തഡ് ഓടിക്കാം. അച്ചടിച്ച് നോക്കുമ്പോൾ 2428 വ്യത്യസ്ത ടാഗുകൾ ഉണ്ടെന്ന് കാണാം. ചെറിയ സാമ്പിൾ:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

`Submitted from a mobile device` പോലുള്ള ചില സാധാരണ ടാഗുകൾ ഞങ്ങൾക്ക് ഉപകാരമില്ല, അതിനാൽ അവ എണ്ണുന്നതിന് മുമ്പ് നീക്കം ചെയ്യുന്നത് ബുദ്ധിമാനായിരിക്കും, പക്ഷേ അത്ര വേഗത്തിൽ പ്രവർത്തിക്കുന്നതിനാൽ അവ അവിടെ വെച്ച് അവഗണിക്കാം.

### താമസ കാലാവധി ടാഗുകൾ നീക്കം ചെയ്യൽ

ഈ ടാഗുകൾ നീക്കം ചെയ്യുന്നത് ആദ്യ ഘട്ടമാണ്, ഇത് പരിഗണിക്കേണ്ട ടാഗുകളുടെ മൊത്തം എണ്ണം കുറയ്ക്കും. ഡാറ്റാസെറ്റിൽ നിന്ന് നീക്കം ചെയ്യുന്നത് അല്ല, റിവ്യൂ ഡാറ്റയിൽ എണ്ണലിൽ/സൂക്ഷിക്കലിൽ നിന്ന് മാത്രം നീക്കം ചെയ്യുക.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

വിവിധ മുറികൾ, സ്യൂട്ടുകൾ, സ്റ്റുഡിയോകൾ, അപാർട്ട്മെന്റുകൾ എന്നിവ വളരെ വ്യത്യസ്തമാണ്. അവ എല്ലാം ഏകദേശം ഒരുപോലെ അർത്ഥം വഹിക്കുന്നു, അതിനാൽ അവ പരിഗണനയിൽ നിന്ന് നീക്കം ചെയ്യുക.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

അവസാനമായി, ഇത് സന്തോഷകരമാണ് (കുറച്ച് പ്രോസസ്സിംഗ് മാത്രം ആവശ്യമായതിനാൽ), നിങ്ങൾക്ക് താഴെപ്പറയുന്ന *ഉപകാരപ്രദമായ* ടാഗുകൾ മാത്രം ബാക്കി ഉണ്ടാകും:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

`Travellers with friends` `Group` എന്നതിന്റെ സമാനമാണ് എന്ന് വാദിക്കാം, അതിനാൽ മുകളിൽ കാണുന്ന പോലെ രണ്ടും ചേർക്കുന്നത് ന്യായമാണ്. ശരിയായ ടാഗുകൾ കണ്ടെത്താനുള്ള കോഡ് [Tags നോട്ട്‌ബുക്ക്](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ൽ കാണാം.

അവസാന ഘട്ടം ഈ ടാഗുകൾക്ക് ഓരോന്നിനും പുതിയ കോളങ്ങൾ സൃഷ്ടിക്കുക. പിന്നീട്, ഓരോ റിവ്യൂ വരിയിലും, `Tag` കോളം പുതിയ കോളങ്ങളിൽ ഒന്നുമായി പൊരുത്തപ്പെടുന്നെങ്കിൽ 1 ചേർക്കുക, അല്ലെങ്കിൽ 0 ചേർക്കുക. ഇതിന്റെ ഫലം, എത്ര റിവ്യൂവറുകൾ ആ ഹോട്ടൽ തിരഞ്ഞെടുക്കുകയാണെന്ന് (ഉദാ: ബിസിനസ് vs ലെisure, അല്ലെങ്കിൽ ഒരു മൃഗം കൊണ്ടുപോകാൻ) എണ്ണമായിരിക്കും, ഇത് ഹോട്ടൽ ശുപാർശ ചെയ്യുമ്പോൾ ഉപകാരപ്രദമാണ്.

```python
# ടാഗുകൾ പുതിയ കോളങ്ങളായി പ്രോസസ് ചെയ്യുക
# Hotel_Reviews_Tags.py ഫയൽ, ഏറ്റവും പ്രധാനപ്പെട്ട ടാഗുകൾ തിരിച്ചറിയുന്നു
# വിനോദയാത്ര, ദമ്പതികൾ, ഒറ്റയാത്രക്കാരൻ, ബിസിനസ് യാത്ര, കൂട്ടം യാത്രക്കാർക്ക് കൂട്ടിച്ചേർത്തത്,
# ചെറുപ്പക്കാരുള്ള കുടുംബം, മുതിർന്ന കുട്ടികളുള്ള കുടുംബം, ഒരു മൃഗത്തോടൊപ്പം
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### ഫയൽ സേവ് ചെയ്യുക

അവസാനമായി, ഇപ്പോഴത്തെ ഡാറ്റാസെറ്റ് പുതിയ പേരിൽ സേവ് ചെയ്യുക.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# കണക്കാക്കിയ കോളങ്ങളോടുകൂടിയ പുതിയ ഡാറ്റ ഫയൽ സേവ് ചെയ്യുന്നു
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## അനുഭവം വിശകലന പ്രവർത്തനങ്ങൾ

ഈ അവസാന സെക്ഷനിൽ, റിവ്യൂ കോളങ്ങളിൽ അനുഭവം വിശകലനം പ്രയോഗിച്ച് ഫലങ്ങൾ ഡാറ്റാസെറ്റിൽ സേവ് ചെയ്യും.

## അഭ്യാസം: ഫിൽട്ടർ ചെയ്ത ഡാറ്റ ലോഡ് ചെയ്ത് സേവ് ചെയ്യുക

ഇപ്പോൾ നിങ്ങൾ മുമ്പത്തെ സെക്ഷനിൽ സേവ് ചെയ്ത ഫിൽട്ടർ ചെയ്ത ഡാറ്റാസെറ്റ് ലോഡ് ചെയ്യുകയാണ്, **അസൽ ഡാറ്റാസെറ്റ് അല്ല**.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# ഫിൽട്ടർ ചെയ്ത ഹോട്ടൽ റിവ്യൂകൾ CSV-യിൽ നിന്ന് ലോഡ് ചെയ്യുക
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# നിങ്ങളുടെ കോഡ് ഇവിടെ ചേർക്കും


# അവസാനം പുതിയ NLP ഡാറ്റ ചേർത്ത ഹോട്ടൽ റിവ്യൂകൾ സേവ് ചെയ്യാൻ മറക്കരുത്
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### സ്റ്റോപ്പ് വാക്കുകൾ നീക്കം ചെയ്യൽ

നീഗറ്റീവ്, പോസിറ്റീവ് റിവ്യൂ കോളങ്ങളിൽ അനുഭവം വിശകലനം നടത്താൻ പോകുമ്പോൾ, അത് വളരെ സമയം എടുക്കാം. ശക്തമായ ലാപ്‌ടോപ്പിൽ പരീക്ഷിച്ചപ്പോൾ 12 - 14 മിനിറ്റ് എടുത്തു, ഉപയോഗിച്ച അനുഭവം ലൈബ്രറിയുടെ അടിസ്ഥാനത്തിൽ വ്യത്യാസമുണ്ട്. ഇത് (സംബന്ധിച്ച) ദീർഘകാലമാണ്, അതിനാൽ വേഗത വർദ്ധിപ്പിക്കാൻ ശ്രമിക്കേണ്ടതാണ്.

സ്റ്റോപ്പ് വാക്കുകൾ, അഥവാ സാധാരണ ഇംഗ്ലീഷ് വാക്കുകൾ, ഒരു വാചകത്തിന്റെ അനുഭവം മാറ്റുന്നില്ല, അവ നീക്കം ചെയ്യുന്നത് ആദ്യ ഘട്ടമാണ്. അവ നീക്കം ചെയ്താൽ അനുഭവം വിശകലനം വേഗത്തിൽ നടക്കും, കൃത്യത കുറയില്ല (സ്റ്റോപ്പ് വാക്കുകൾ അനുഭവത്തെ ബാധിക്കാറില്ല, പക്ഷേ വിശകലനം മന്ദഗതിയാക്കുന്നു).

ഏറ്റവും നീണ്ട നെഗറ്റീവ് റിവ്യൂ 395 വാക്കുകൾ ആയിരുന്നു, സ്റ്റോപ്പ് വാക്കുകൾ നീക്കം ചെയ്ത ശേഷം 195 വാക്കുകൾ മാത്രമാണ്.

സ്റ്റോപ്പ് വാക്കുകൾ നീക്കം ചെയ്യൽ വേഗത്തിലുള്ള പ്രവർത്തനമാണ്, 2 റിവ്യൂ കോളങ്ങളിൽ 515,000 വരികളിൽ 3.3 സെക്കൻഡ് എടുത്തു. നിങ്ങളുടെ ഉപകരണത്തിന്റെ CPU വേഗം, RAM, SSD ഉണ്ട്/ഇല്ല, മറ്റ് ഘടകങ്ങൾ എന്നിവ അനുസരിച്ച് സമയം വ്യത്യാസപ്പെടാം. പ്രവർത്തനത്തിന്റെ സാന്ദ്രത കുറവായതിനാൽ, ഇത് അനുഭവം വിശകലന സമയം മെച്ചപ്പെടുത്തുകയാണെങ്കിൽ ചെയ്യുന്നത് ഉചിതമാണ്.

```python
from nltk.corpus import stopwords

# ഹോട്ടൽ റിവ്യൂകൾ CSV-യിൽ നിന്ന് ലോഡ് ചെയ്യുക
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# സ്റ്റോപ്പ് വാക്കുകൾ നീക്കം ചെയ്യുക - വളരെ വലുതായ ടെക്സ്റ്റിനായി ഇത് മന്ദഗതിയാകാം!
# റയാൻ ഹാൻ (Kaggle-ൽ ryanxjhan) വിവിധ സ്റ്റോപ്പ് വാക്ക് നീക്കം ചെയ്യൽ സമീപനങ്ങളുടെ പ്രകടനം അളക്കുന്ന മികച്ച ഒരു പോസ്റ്റ് ഉണ്ട്
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # റയാൻ ശുപാർശ ചെയ്യുന്ന സമീപനം ഉപയോഗിച്ച്
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# രണ്ട് കോളങ്ങളിലെയും സ്റ്റോപ്പ് വാക്കുകൾ നീക്കം ചെയ്യുക
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### അനുഭവം വിശകലനം നടത്തൽ

ഇപ്പോൾ നിങ്ങൾ നെഗറ്റീവ്, പോസിറ്റീവ് റിവ്യൂ കോളങ്ങളിൽ അനുഭവം വിശകലനം കണക്കാക്കി 2 പുതിയ കോളങ്ങളിൽ ഫലം സൂക്ഷിക്കണം. അനുഭവം പരിശോധന റിവ്യൂവറുടെ സ്കോറുമായി താരതമ്യം ചെയ്യുന്നതായിരിക്കും. ഉദാഹരണത്തിന്, നെഗറ്റീവ് റിവ്യൂ sentiment 1 (അത്യന്തം പോസിറ്റീവ്) ആണെന്ന് അനുഭവം വിശകലന യന്ത്രം കരുതിയാൽ, പോസിറ്റീവ് റിവ്യൂ sentiment 1 ആണെങ്കിൽ, എന്നാൽ റിവ്യൂവർ ഹോട്ടലിന് ഏറ്റവും താഴ്ന്ന സ്കോർ നൽകിയാൽ, റിവ്യൂ ടെക്സ്റ്റും സ്കോറും പൊരുത്തപ്പെടുന്നില്ല, അല്ലെങ്കിൽ sentiment analyser ശരിയായി തിരിച്ചറിയാൻ കഴിഞ്ഞില്ല. ചില sentiment സ്കോറുകൾ പൂർണ്ണമായും തെറ്റായിരിക്കാം, പലപ്പോഴും അത് വിശദീകരിക്കാവുന്നതാണ്, ഉദാ: റിവ്യൂ വളരെ സാര്കാസ്റ്റിക് ആയിരിക്കാം "Of course I LOVED sleeping in a room with no heating" എന്നുപറഞ്ഞാൽ sentiment analyser അത് പോസിറ്റീവ് sentiment ആണെന്ന് കരുതും, എന്നാൽ മനുഷ്യൻ വായിച്ചാൽ അത് സാര്കാസം ആണെന്ന് അറിയും.
NLTK വ്യത്യസ്തമായ സെന്റിമെന്റ് അനലൈസറുകൾ പഠിക്കാൻ നൽകുന്നു, നിങ്ങൾ അവ മാറ്റി സെന്റിമെന്റ് കൂടുതൽ കൃത്യമാണോ കുറവാണോ എന്ന് പരിശോധിക്കാം. ഇവിടെ VADER സെന്റിമെന്റ് അനാലിസിസ് ഉപയോഗിക്കുന്നു.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# വാഡർ സെന്റിമെന്റ് അനലൈസർ സൃഷ്ടിക്കുക (നിങ്ങൾക്ക് പരീക്ഷിക്കാൻ കഴിയുന്ന മറ്റ് NLTK ഉം ഉണ്ട്)
vader_sentiment = SentimentIntensityAnalyzer()
# ഹുട്ടോ, സി.ജെ. & ഗിൽബർട്ട്, ഇ.ഇ. (2014). VADER: സോഷ്യൽ മീഡിയ ടെക്സ്റ്റിന്റെ സെന്റിമെന്റ് വിശകലനത്തിനുള്ള ഒരു ലഘുവായ നിയമാധിഷ്ഠിത മോഡൽ. എട്ടാം അന്താരാഷ്ട്ര വെബ്ലോഗ്‌സ് ആൻഡ് സോഷ്യൽ മീഡിയ സമ്മേളനം (ICWSM-14). ആൻ ആർബർ, MI, ജൂൺ 2014.

# ഒരു റിവ്യൂവിന് 3 ഇൻപുട്ട് സാധ്യതകൾ ഉണ്ട്:
# അത് "നോ നെഗറ്റീവ്" ആകാം, അപ്പോൾ 0 മടക്കുക
# അത് "നോ പോസിറ്റീവ്" ആകാം, അപ്പോൾ 0 മടക്കുക
# അത് ഒരു റിവ്യൂ ആകാം, അപ്പോൾ സെന്റിമെന്റ് കണക്കാക്കുക
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

നിങ്ങളുടെ പ്രോഗ്രാമിൽ പിന്നീട് സെന്റിമെന്റ് കണക്കാക്കാൻ തയ്യാറായപ്പോൾ, ഓരോ റിവ്യൂവിനും ഇത് ഇങ്ങനെ പ്രയോഗിക്കാം:

```python
# ഒരു നെഗറ്റീവ് സെന്റിമെന്റ് കോളവും പോസിറ്റീവ് സെന്റിമെന്റ് കോളവും ചേർക്കുക
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

എന്റെ കമ്പ്യൂട്ടറിൽ ഇത് ഏകദേശം 120 സെക്കൻഡ് എടുക്കുന്നു, പക്ഷേ ഓരോ കമ്പ്യൂട്ടറിലും വ്യത്യാസമുണ്ടാകും. ഫലങ്ങൾ പ്രിന്റ് ചെയ്ത് സെന്റിമെന്റ് റിവ്യൂവിനോട് പൊരുത്തപ്പെടുന്നുണ്ടോ എന്ന് കാണാൻ:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

ചലഞ്ചിൽ ഉപയോഗിക്കുന്നതിന് മുമ്പ് ഫയൽ സംരക്ഷിക്കുക! പുതിയ എല്ലാ കോളങ്ങളെയും പുനഃക്രമീകരിക്കാൻ പരിഗണിക്കുക, ഇത് മനുഷ്യനായി പ്രവർത്തിക്കാൻ എളുപ്പമാക്കും (ഇത് ഒരു കോസ്മെറ്റിക് മാറ്റമാണ്).

```python
# കോളങ്ങൾ പുനഃക്രമീകരിക്കുക (ഇത് ദൃശ്യപരമാണ്, പക്ഷേ പിന്നീട് ഡാറ്റ എളുപ്പത്തിൽ പരിശോധിക്കാൻ)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

[ആനാലിസിസ് നോട്ട്‌ബുക്ക്](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) മുഴുവൻ കോഡ് പ്രവർത്തിപ്പിക്കണം (നിങ്ങൾ [ഫിൽട്ടറിംഗ് നോട്ട്‌ബുക്ക്](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) പ്രവർത്തിപ്പിച്ച് Hotel_Reviews_Filtered.csv ഫയൽ സൃഷ്ടിച്ച ശേഷം).

പരിശോധിക്കാൻ, ചുവടെയുള്ള ഘട്ടങ്ങളാണ്:

1. ഒറിജിനൽ ഡാറ്റാസെറ്റ് ഫയൽ **Hotel_Reviews.csv** മുമ്പത്തെ പാഠത്തിൽ [എക്സ്പ്ലോറർ നോട്ട്‌ബുക്ക്](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) ഉപയോഗിച്ച് പരിശോധിച്ചു
2. Hotel_Reviews.csv [ഫിൽട്ടറിംഗ് നോട്ട്‌ബുക്ക്](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ഉപയോഗിച്ച് ഫിൽട്ടർ ചെയ്ത് **Hotel_Reviews_Filtered.csv** സൃഷ്ടിച്ചു
3. Hotel_Reviews_Filtered.csv [സെന്റിമെന്റ് അനാലിസിസ് നോട്ട്‌ബുക്ക്](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ഉപയോഗിച്ച് പ്രോസസ് ചെയ്ത് **Hotel_Reviews_NLP.csv** സൃഷ്ടിച്ചു
4. താഴെ കൊടുത്ത NLP ചലഞ്ചിൽ Hotel_Reviews_NLP.csv ഉപയോഗിക്കുക

### നിഗമനം

നിങ്ങൾ ആരംഭിച്ചപ്പോൾ, കോളങ്ങളും ഡാറ്റയും ഉള്ള ഒരു ഡാറ്റാസെറ്റ് ഉണ്ടായിരുന്നു, പക്ഷേ എല്ലാം പരിശോധിക്കാനോ ഉപയോഗിക്കാനോ കഴിയുന്നില്ലായിരുന്നു. നിങ്ങൾ ഡാറ്റ പരിശോധിച്ചു, ആവശ്യമില്ലാത്തത് ഫിൽട്ടർ ചെയ്തു, ടാഗുകൾ ഉപയോഗപ്രദമായ ഒന്നായി മാറ്റി, നിങ്ങളുടെ സ്വന്തം ശരാശരികൾ കണക്കാക്കി, ചില സെന്റിമെന്റ് കോളങ്ങൾ ചേർത്ത്, സ്വാഭാവിക വാചക പ്രോസസ്സിംഗിനെക്കുറിച്ച് ചില രസകരമായ കാര്യങ്ങൾ പഠിച്ചു.

## [പാഠാനന്തര ക്വിസ്](https://ff-quizzes.netlify.app/en/ml/)

## ചലഞ്ച്

ഇപ്പോൾ നിങ്ങളുടെ ഡാറ്റാസെറ്റ് സെന്റിമെന്റിനായി വിശകലനം ചെയ്തതിനുശേഷം, ഈ പാഠ്യപദ്ധതിയിൽ നിങ്ങൾ പഠിച്ച തന്ത്രങ്ങൾ (ക്ലസ്റ്ററിംഗ്, ആകാം?) ഉപയോഗിച്ച് സെന്റിമെന്റിനുള്ള പാറ്റേണുകൾ കണ്ടെത്താൻ ശ്രമിക്കുക.

## അവലോകനവും സ്വയം പഠനവും

[ഈ ലേൺ മോഡ്യൂൾ](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) എടുത്ത് കൂടുതൽ പഠിക്കുകയും ടെക്സ്റ്റിലെ സെന്റിമെന്റ് പരിശോധിക്കാൻ വ്യത്യസ്ത ഉപകരണങ്ങൾ ഉപയോഗിക്കുകയും ചെയ്യുക.

## അസൈൻമെന്റ്

[മറ്റൊരു ഡാറ്റാസെറ്റ് പരീക്ഷിക്കുക](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**അസൂയാ**:  
ഈ രേഖ AI വിവർത്തന സേവനം [Co-op Translator](https://github.com/Azure/co-op-translator) ഉപയോഗിച്ച് വിവർത്തനം ചെയ്തതാണ്. നാം കൃത്യതയ്ക്ക് ശ്രമിച്ചെങ്കിലും, സ്വയം പ്രവർത്തിക്കുന്ന വിവർത്തനങ്ങളിൽ പിശകുകൾ അല്ലെങ്കിൽ തെറ്റുകൾ ഉണ്ടാകാമെന്ന് ദയവായി ശ്രദ്ധിക്കുക. അതിന്റെ മാതൃഭാഷയിലുള്ള യഥാർത്ഥ രേഖ അധികാരപരമായ ഉറവിടമായി കണക്കാക്കണം. നിർണായക വിവരങ്ങൾക്ക്, പ്രൊഫഷണൽ മനുഷ്യ വിവർത്തനം ശുപാർശ ചെയ്യപ്പെടുന്നു. ഈ വിവർത്തനം ഉപയോഗിക്കുന്നതിൽ നിന്നുണ്ടാകുന്ന ഏതെങ്കിലും തെറ്റിദ്ധാരണകൾക്കോ തെറ്റായ വ്യാഖ്യാനങ്ങൾക്കോ ഞങ്ങൾ ഉത്തരവാദികളല്ല.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->