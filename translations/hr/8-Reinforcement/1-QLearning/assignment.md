<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T13:42:18+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "hr"
}
-->
# Realističniji svijet

U našoj situaciji, Peter se mogao kretati gotovo bez umaranja ili osjećaja gladi. U realističnijem svijetu, morao bi se povremeno odmoriti i nahraniti. Učinimo naš svijet realističnijim implementirajući sljedeća pravila:

1. Kretanjem s jednog mjesta na drugo, Peter gubi **energiju** i dobiva **umor**.
2. Peter može dobiti više energije jedući jabuke.
3. Peter se može riješiti umora odmarajući se ispod stabla ili na travi (tj. hodanjem do polja na ploči koje ima stablo ili travu - zeleno polje).
4. Peter mora pronaći i ubiti vuka.
5. Da bi ubio vuka, Peter mora imati određene razine energije i umora, inače gubi bitku.

## Upute

Koristite originalni [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) kao početnu točku za svoje rješenje.

Modificirajte funkciju nagrade prema pravilima igre, pokrenite algoritam za učenje pojačanjem kako biste naučili najbolju strategiju za pobjedu u igri, i usporedite rezultate nasumičnog hodanja s vašim algoritmom u smislu broja pobijeđenih i izgubljenih igara.

> **Note**: U vašem novom svijetu, stanje je složenije i, uz poziciju čovjeka, uključuje i razine umora i energije. Možete odabrati prikazati stanje kao tuple (Ploča, energija, umor), ili definirati klasu za stanje (možete je također izvesti iz `Board`), ili čak modificirati originalnu klasu `Board` unutar [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

U svom rješenju, molimo vas da zadržite kod odgovoran za strategiju nasumičnog hodanja i usporedite rezultate svog algoritma s nasumičnim hodanjem na kraju.

> **Note**: Možda ćete morati prilagoditi hiperparametre kako bi sve funkcioniralo, posebno broj epoha. Budući da je uspjeh u igri (borba s vukom) rijedak događaj, možete očekivati znatno duže vrijeme treniranja.

## Rubrika

| Kriterij | Izvrsno                                                                                                                                                                                                 | Zadovoljavajuće                                                                                                                                                                       | Potrebno poboljšanje                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Predstavljen je notebook s definicijom novih pravila svijeta, Q-Learning algoritmom i nekim tekstualnim objašnjenjima. Q-Learning značajno poboljšava rezultate u usporedbi s nasumičnim hodanjem.       | Predstavljen je notebook, Q-Learning je implementiran i poboljšava rezultate u usporedbi s nasumičnim hodanjem, ali ne značajno; ili je notebook loše dokumentiran, a kod nije dobro strukturiran. | Napravljeni su neki pokušaji redefiniranja pravila svijeta, ali Q-Learning algoritam ne funkcionira ili funkcija nagrade nije potpuno definirana. |

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden koristeći AI uslugu za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.