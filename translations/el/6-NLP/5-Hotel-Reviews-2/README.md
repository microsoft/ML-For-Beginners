<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T01:45:22+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "el"
}
-->
# Ανάλυση συναισθημάτων με κριτικές ξενοδοχείων

Τώρα που έχετε εξερευνήσει το σύνολο δεδομένων λεπτομερώς, είναι ώρα να φιλτράρετε τις στήλες και να χρησιμοποιήσετε τεχνικές NLP στο σύνολο δεδομένων για να αποκτήσετε νέες πληροφορίες σχετικά με τα ξενοδοχεία.

## [Προ-διάλεξη κουίζ](https://ff-quizzes.netlify.app/en/ml/)

### Ενέργειες Φιλτραρίσματος & Ανάλυσης Συναισθημάτων

Όπως πιθανώς έχετε παρατηρήσει, το σύνολο δεδομένων έχει κάποια προβλήματα. Ορισμένες στήλες είναι γεμάτες με άχρηστες πληροφορίες, ενώ άλλες φαίνονται λανθασμένες. Ακόμα κι αν είναι σωστές, δεν είναι σαφές πώς υπολογίστηκαν, και οι απαντήσεις δεν μπορούν να επαληθευτούν ανεξάρτητα μέσω των δικών σας υπολογισμών.

## Άσκηση: Λίγη περισσότερη επεξεργασία δεδομένων

Καθαρίστε τα δεδομένα λίγο περισσότερο. Προσθέστε στήλες που θα είναι χρήσιμες αργότερα, αλλάξτε τις τιμές σε άλλες στήλες και διαγράψτε ορισμένες στήλες εντελώς.

1. Αρχική επεξεργασία στηλών

   1. Διαγράψτε τις `lat` και `lng`

   2. Αντικαταστήστε τις τιμές της `Hotel_Address` με τις εξής τιμές (αν η διεύθυνση περιέχει το όνομα της πόλης και της χώρας, αλλάξτε την ώστε να περιέχει μόνο την πόλη και τη χώρα).

      Αυτές είναι οι μοναδικές πόλεις και χώρες στο σύνολο δεδομένων:

      Άμστερνταμ, Ολλανδία

      Βαρκελώνη, Ισπανία

      Λονδίνο, Ηνωμένο Βασίλειο

      Μιλάνο, Ιταλία

      Παρίσι, Γαλλία

      Βιέννη, Αυστρία 

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
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Τώρα μπορείτε να κάνετε ερωτήματα σε επίπεδο χώρας:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Άμστερνταμ, Ολλανδία   |    105     |
      | Βαρκελώνη, Ισπανία     |    211     |
      | Λονδίνο, Ηνωμένο Βασίλειο |    400     |
      | Μιλάνο, Ιταλία         |    162     |
      | Παρίσι, Γαλλία         |    458     |
      | Βιέννη, Αυστρία        |    158     |

2. Επεξεργασία στηλών Μετα-κριτικής Ξενοδοχείου

  1. Διαγράψτε την `Additional_Number_of_Scoring`

  1. Αντικαταστήστε την `Total_Number_of_Reviews` με τον συνολικό αριθμό κριτικών για το συγκεκριμένο ξενοδοχείο που βρίσκονται πραγματικά στο σύνολο δεδομένων 

  1. Αντικαταστήστε την `Average_Score` με τη δική μας υπολογισμένη βαθμολογία

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Επεξεργασία στηλών κριτικής

   1. Διαγράψτε τις `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` και `days_since_review`

   2. Κρατήστε τις `Reviewer_Score`, `Negative_Review` και `Positive_Review` όπως είναι,
     
   3. Κρατήστε τις `Tags` προς το παρόν

     - Θα κάνουμε κάποιες επιπλέον ενέργειες φιλτραρίσματος στις ετικέτες στην επόμενη ενότητα και μετά οι ετικέτες θα διαγραφούν

4. Επεξεργασία στηλών κριτικών χρηστών

  1. Διαγράψτε την `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Κρατήστε την `Reviewer_Nationality`

### Στήλη Ετικετών

Η στήλη `Tag` είναι προβληματική καθώς είναι μια λίστα (σε μορφή κειμένου) αποθηκευμένη στη στήλη. Δυστυχώς, η σειρά και ο αριθμός των υποτμημάτων σε αυτή τη στήλη δεν είναι πάντα τα ίδια. Είναι δύσκολο για έναν άνθρωπο να εντοπίσει τις σωστές φράσεις που πρέπει να ενδιαφερθεί, επειδή υπάρχουν 515.000 γραμμές και 1427 ξενοδοχεία, και κάθε ένα έχει ελαφρώς διαφορετικές επιλογές που θα μπορούσε να επιλέξει ένας κριτικός. Εδώ είναι που το NLP διαπρέπει. Μπορείτε να σαρώσετε το κείμενο και να βρείτε τις πιο κοινές φράσεις και να τις μετρήσετε.

Δυστυχώς, δεν μας ενδιαφέρουν οι μεμονωμένες λέξεις, αλλά οι φράσεις πολλών λέξεων (π.χ. *Επαγγελματικό ταξίδι*). Η εκτέλεση ενός αλγορίθμου κατανομής συχνότητας φράσεων πολλών λέξεων σε τόσα δεδομένα (6762646 λέξεις) θα μπορούσε να πάρει εξαιρετικά πολύ χρόνο, αλλά χωρίς να κοιτάξετε τα δεδομένα, φαίνεται ότι είναι απαραίτητη δαπάνη. Εδώ είναι που η εξερευνητική ανάλυση δεδομένων είναι χρήσιμη, επειδή έχετε δει ένα δείγμα των ετικετών όπως `[' Επαγγελματικό ταξίδι  ', ' Μοναχικός ταξιδιώτης ', ' Μονόκλινο δωμάτιο ', ' Έμεινε 5 νύχτες ', ' Υποβλήθηκε από κινητή συσκευή ']`, μπορείτε να αρχίσετε να ρωτάτε αν είναι δυνατόν να μειώσετε σημαντικά την επεξεργασία που πρέπει να κάνετε. Ευτυχώς, είναι - αλλά πρώτα πρέπει να ακολουθήσετε μερικά βήματα για να διαπιστώσετε τις ετικέτες ενδιαφέροντος.

### Φιλτράρισμα ετικετών

Θυμηθείτε ότι ο στόχος του συνόλου δεδομένων είναι να προσθέσετε συναισθήματα και στήλες που θα σας βοηθήσουν να επιλέξετε το καλύτερο ξενοδοχείο (για εσάς ή ίσως για έναν πελάτη που σας αναθέτει να δημιουργήσετε ένα bot σύστασης ξενοδοχείου). Πρέπει να αναρωτηθείτε αν οι ετικέτες είναι χρήσιμες ή όχι στο τελικό σύνολο δεδομένων. Εδώ είναι μια ερμηνεία (αν χρειαζόσασταν το σύνολο δεδομένων για άλλους λόγους, διαφορετικές ετικέτες μπορεί να παραμείνουν/αφαιρεθούν από την επιλογή):

1. Ο τύπος ταξιδιού είναι σχετικός και πρέπει να παραμείνει
2. Ο τύπος της ομάδας επισκεπτών είναι σημαντικός και πρέπει να παραμείνει
3. Ο τύπος δωματίου, σουίτας ή στούντιο που έμεινε ο επισκέπτης είναι άσχετος (όλα τα ξενοδοχεία έχουν βασικά τα ίδια δωμάτια)
4. Η συσκευή από την οποία υποβλήθηκε η κριτική είναι άσχετη
5. Ο αριθμός των νυχτών που έμεινε ο κριτικός *θα μπορούσε* να είναι σχετικός αν αποδίδατε μεγαλύτερες διαμονές με το να τους αρέσει περισσότερο το ξενοδοχείο, αλλά είναι αμφίβολο και πιθανώς άσχετο

Συνοπτικά, **κρατήστε 2 είδη ετικετών και αφαιρέστε τις υπόλοιπες**.

Πρώτα, δεν θέλετε να μετρήσετε τις ετικέτες μέχρι να είναι σε καλύτερη μορφή, οπότε αυτό σημαίνει να αφαιρέσετε τα τετράγωνα αγκύλες και τα εισαγωγικά. Μπορείτε να το κάνετε αυτό με διάφορους τρόπους, αλλά θέλετε τον πιο γρήγορο καθώς θα μπορούσε να πάρει πολύ χρόνο να επεξεργαστείτε πολλά δεδομένα. Ευτυχώς, το pandas έχει έναν εύκολο τρόπο να κάνετε κάθε ένα από αυτά τα βήματα.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Κάθε ετικέτα γίνεται κάτι σαν: `Επαγγελματικό ταξίδι, Μοναχικός ταξιδιώτης, Μονόκλινο δωμάτιο, Έμεινε 5 νύχτες, Υποβλήθηκε από κινητή συσκευή`. 

Στη συνέχεια βρίσκουμε ένα πρόβλημα. Ορισμένες κριτικές ή γραμμές έχουν 5 στήλες, άλλες 3, άλλες 6. Αυτό είναι αποτέλεσμα του πώς δημιουργήθηκε το σύνολο δεδομένων και δύσκολο να διορθωθεί. Θέλετε να πάρετε μια καταμέτρηση συχνότητας κάθε φράσης, αλλά είναι σε διαφορετική σειρά σε κάθε κριτική, οπότε η καταμέτρηση μπορεί να είναι λανθασμένη και ένα ξενοδοχείο μπορεί να μην πάρει μια ετικέτα που του άξιζε.

Αντί να το διορθώσετε, θα χρησιμοποιήσετε τη διαφορετική σειρά προς όφελός μας, επειδή κάθε ετικέτα είναι πολλών λέξεων αλλά επίσης χωρίζεται με κόμμα! Ο απλούστερος τρόπος να το κάνετε αυτό είναι να δημιουργήσετε 6 προσωρινές στήλες με κάθε ετικέτα να εισάγεται στη στήλη που αντιστοιχεί στη σειρά της ετικέτας. Στη συνέχεια μπορείτε να συγχωνεύσετε τις 6 στήλες σε μία μεγάλη στήλη και να εκτελέσετε τη μέθοδο `value_counts()` στη resulting στήλη. Εκτυπώνοντας αυτό, θα δείτε ότι υπήρχαν 2428 μοναδικές ετικέτες. Εδώ είναι ένα μικρό δείγμα:

| Ετικέτα                        | Καταμέτρηση |
| ------------------------------ | ----------- |
| Ταξίδι αναψυχής               | 417778      |
| Υποβλήθηκε από κινητή συσκευή | 307640      |
| Ζευγάρι                       | 252294      |
| Έμεινε 1 νύχτα                | 193645      |
| Έμεινε 2 νύχτες               | 133937      |
| Μοναχικός ταξιδιώτης          | 108545      |
| Έμεινε 3 νύχτες               | 95821       |
| Επαγγελματικό ταξίδι          | 82939       |
| Ομάδα                         | 65392       |
| Οικογένεια με μικρά παιδιά    | 61015       |
| Έμεινε 4 νύχτες               | 47817       |
| Δίκλινο δωμάτιο               | 35207       |
| Standard Δίκλινο δωμάτιο      | 32248       |
| Superior Δίκλινο δωμάτιο      | 31393       |
| Οικογένεια με μεγαλύτερα παιδιά | 26349       |
| Deluxe Δίκλινο δωμάτιο        | 24823       |
| Δίκλινο ή Twin δωμάτιο        | 22393       |
| Έμεινε 5 νύχτες               | 20845       |
| Standard Δίκλινο ή Twin δωμάτιο | 17483       |
| Classic Δίκλινο δωμάτιο       | 16989       |
| Superior Δίκλινο ή Twin δωμάτιο | 13570       |
| 2 δωμάτια                     | 12393       |

Ορισμένες από τις κοινές ετικέτες όπως `Υποβλήθηκε από κινητή συσκευή` δεν μας είναι χρήσιμες, οπότε ίσως είναι έξυπνο να τις αφαιρέσουμε πριν μετρήσουμε την εμφάνιση φράσεων, αλλά είναι τόσο γρήγορη λειτουργία που μπορείτε να τις αφήσετε και να τις αγνοήσετε.

### Αφαίρεση ετικετών διάρκειας διαμονής

Η αφαίρεση αυτών των ετικετών είναι το πρώτο βήμα, μειώνει τον συνολικό αριθμό ετικετών που πρέπει να ληφθούν υπόψη ελαφρώς. Σημειώστε ότι δεν τις αφαιρείτε από το σύνολο δεδομένων, απλώς επιλέγετε να τις αφαιρέσετε από την εξέταση ως τιμές για καταμέτρηση/διατήρηση στο σύνολο δεδομένων κριτικών.

| Διάρκεια διαμονής | Καταμέτρηση |
| ----------------- | ----------- |
| Έμεινε 1 νύχτα   | 193645      |
| Έμεινε 2 νύχτες  | 133937      |
| Έμεινε 3 νύχτες  | 95821       |
| Έμεινε 4 νύχτες  | 47817       |
| Έμεινε 5 νύχτες  | 20845       |
| Έμεινε 6 νύχτες  | 9776        |
| Έμεινε 7 νύχτες  | 7399        |
| Έμεινε 8 νύχτες  | 2502        |
| Έμεινε 9 νύχτες  | 1293        |
| ...              | ...         |

Υπάρχει μια τεράστια ποικιλία δωματίων, σουιτών, στούντιο, διαμερισμάτων και ούτω καθεξής. Όλα σημαίνουν περίπου το ίδιο πράγμα και δεν είναι σχετικά για εσάς, οπότε αφαιρέστε τα από την εξέταση.

| Τύπος δωματίου               | Καταμέτρηση |
| ---------------------------- | ----------- |
| Δίκλινο δωμάτιο             | 35207       |
| Standard Δίκλινο δωμάτιο    | 32248       |
| Superior Δίκλινο δωμάτιο    | 31393       |
| Deluxe Δίκλινο δωμάτιο      | 24823       |
| Δίκλινο ή Twin δωμάτιο      | 22393       |
| Standard Δίκλινο ή Twin δωμάτιο | 17483       |
| Classic Δίκλινο δωμάτιο     | 16989       |
| Superior Δίκλινο ή Twin δωμάτιο | 13570       |

Τέλος, και αυτό είναι ευχάριστο (επειδή δεν χρειάστηκε πολλή επεξεργασία), θα μείνετε με τις εξής *χρήσιμες* ετικέτες:

| Ετικέτα                                      | Καταμέτρηση |
| -------------------------------------------- | ----------- |
| Ταξίδι αναψυχής                              | 417778      |
| Ζευγάρι                                      | 252294      |
| Μοναχικός ταξιδιώτης                         | 108545      |
| Επαγγελματικό ταξίδι                         | 82939       |
| Ομάδα (συνδυασμένο με Ταξιδιώτες με φίλους) | 67535       |
| Οικογένεια με μικρά παιδιά                  | 61015       |
| Οικογένεια με μεγαλύτερα παιδιά             | 26349       |
| Με κατοικίδιο                               | 1405        |

Θα μπορούσατε να υποστηρίξετε ότι το `Ταξιδιώτες με φίλους` είναι το ίδιο με το `Ομάδα` λίγο πολύ, και θα ήταν δίκαιο να τα συνδυάσετε όπως παραπάνω. Ο κώδικας για την αναγνώριση των σωστών ετικετών βρίσκεται στο [notebook Ετικετών](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Το τελικό βήμα είναι να δημιουργήσετε νέες στήλες για κάθε μία από αυτές τις ετικέτες. Στη συνέχεια, για κάθε γραμμή κριτικής, αν η στήλη `Tag` ταιριάζει με μία από τις νέες στήλες, προσθέστε 1, αν όχι, προσθέστε 0. Το τελικό αποτέλεσμα θα είναι μια καταμέτρηση του πόσοι κριτικοί επέλεξαν αυτό το ξενοδοχείο (συνολικά) για, π.χ., επαγγελματικό ταξίδι έναντι ταξιδιού αναψυχής ή για να φέρουν κατοικίδιο, και αυτές είναι χρήσιμες πληροφορίες όταν προτείνετε ένα ξενοδοχείο.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Αποθήκευση του αρχείου σας

Τέλος, αποθηκεύστε το σύνολο δεδομένων όπως είναι τώρα με ένα νέο όνομα.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Ενέργειες Ανάλυσης Συναισθημάτων

Σε αυτή την τελική ενότητα, θα εφαρμόσετε ανάλυση συναισθημάτων στις στήλες κριτικής και θα αποθηκεύσετε τα αποτελέσματα σε ένα σύνολο δεδομένων.

## Άσκηση: Φόρτωση και αποθήκευση των φιλτραρισμένων δεδομένων

Σημειώστε ότι τώρα φορτώνετε το φιλτραρισμένο σύνολο δεδομένων που αποθηκεύτηκε στην προηγούμενη ενότητα, **όχι** το αρχικό σύνολο δεδομένων.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Αφαίρεση λέξεων-σταματημάτων

Αν εκτελούσατε ανάλυση συναισθημάτων στις στήλες αρνητικής και θετικής κριτικής, θα μπορούσε να πάρει πολύ χρόνο. Δοκιμασμένο σε ένα ισχυρό laptop με γρήγορη CPU
Το NLTK παρέχει διάφορους αναλυτές συναισθημάτων για να μάθετε, και μπορείτε να τους αντικαταστήσετε και να δείτε αν η ανάλυση συναισθημάτων είναι πιο ή λιγότερο ακριβής. Εδώ χρησιμοποιείται η ανάλυση συναισθημάτων VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Ένα Οικονομικό Μοντέλο Βασισμένο σε Κανόνες για Ανάλυση Συναισθημάτων Κειμένου Κοινωνικών Μέσων. Όγδοο Διεθνές Συνέδριο για Weblogs και Κοινωνικά Μέσα (ICWSM-14). Ann Arbor, MI, Ιούνιος 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Αργότερα στο πρόγραμμά σας, όταν είστε έτοιμοι να υπολογίσετε τα συναισθήματα, μπορείτε να το εφαρμόσετε σε κάθε κριτική ως εξής:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Αυτό διαρκεί περίπου 120 δευτερόλεπτα στον υπολογιστή μου, αλλά θα διαφέρει σε κάθε υπολογιστή. Εάν θέλετε να εκτυπώσετε τα αποτελέσματα και να δείτε αν τα συναισθήματα ταιριάζουν με την κριτική:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Το τελευταίο πράγμα που πρέπει να κάνετε με το αρχείο πριν το χρησιμοποιήσετε στην πρόκληση, είναι να το αποθηκεύσετε! Θα πρέπει επίσης να σκεφτείτε να αναδιατάξετε όλες τις νέες στήλες σας ώστε να είναι εύκολες στη χρήση (για έναν άνθρωπο, είναι μια αισθητική αλλαγή).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Θα πρέπει να εκτελέσετε ολόκληρο τον κώδικα για [το σημειωματάριο ανάλυσης](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (αφού έχετε εκτελέσει [το σημειωματάριο φιλτραρίσματος](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) για να δημιουργήσετε το αρχείο Hotel_Reviews_Filtered.csv).

Για να ανακεφαλαιώσουμε, τα βήματα είναι:

1. Το αρχικό αρχείο δεδομένων **Hotel_Reviews.csv** εξετάζεται στο προηγούμενο μάθημα με [το σημειωματάριο εξερεύνησης](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Το Hotel_Reviews.csv φιλτράρεται από [το σημειωματάριο φιλτραρίσματος](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) και προκύπτει το **Hotel_Reviews_Filtered.csv**
3. Το Hotel_Reviews_Filtered.csv επεξεργάζεται από [το σημειωματάριο ανάλυσης συναισθημάτων](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) και προκύπτει το **Hotel_Reviews_NLP.csv**
4. Χρησιμοποιήστε το Hotel_Reviews_NLP.csv στην πρόκληση NLP παρακάτω

### Συμπέρασμα

Όταν ξεκινήσατε, είχατε ένα σύνολο δεδομένων με στήλες και δεδομένα, αλλά δεν μπορούσαν όλα να επαληθευτούν ή να χρησιμοποιηθούν. Εξερευνήσατε τα δεδομένα, φιλτράρατε ό,τι δεν χρειαζόταν, μετατρέψατε ετικέτες σε κάτι χρήσιμο, υπολογίσατε τους δικούς σας μέσους όρους, προσθέσατε κάποιες στήλες συναισθημάτων και, ελπίζουμε, μάθατε ενδιαφέροντα πράγματα για την επεξεργασία φυσικού κειμένου.

## [Κουίζ μετά το μάθημα](https://ff-quizzes.netlify.app/en/ml/)

## Πρόκληση

Τώρα που έχετε αναλύσει το σύνολο δεδομένων σας για συναισθήματα, δείτε αν μπορείτε να χρησιμοποιήσετε στρατηγικές που έχετε μάθει σε αυτό το πρόγραμμα σπουδών (ίσως clustering;) για να προσδιορίσετε μοτίβα γύρω από τα συναισθήματα.

## Ανασκόπηση & Αυτομελέτη

Πάρτε [αυτό το Learn module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) για να μάθετε περισσότερα και να χρησιμοποιήσετε διαφορετικά εργαλεία για να εξερευνήσετε τα συναισθήματα στο κείμενο.

## Εργασία 

[Δοκιμάστε ένα διαφορετικό σύνολο δεδομένων](assignment.md)

---

**Αποποίηση ευθύνης**:  
Αυτό το έγγραφο έχει μεταφραστεί χρησιμοποιώντας την υπηρεσία αυτόματης μετάφρασης [Co-op Translator](https://github.com/Azure/co-op-translator). Παρόλο που καταβάλλουμε προσπάθειες για ακρίβεια, παρακαλούμε να έχετε υπόψη ότι οι αυτοματοποιημένες μεταφράσεις ενδέχεται να περιέχουν σφάλματα ή ανακρίβειες. Το πρωτότυπο έγγραφο στη μητρική του γλώσσα θα πρέπει να θεωρείται η αυθεντική πηγή. Για κρίσιμες πληροφορίες, συνιστάται επαγγελματική ανθρώπινη μετάφραση. Δεν φέρουμε ευθύνη για τυχόν παρεξηγήσεις ή εσφαλμένες ερμηνείες που προκύπτουν από τη χρήση αυτής της μετάφρασης.