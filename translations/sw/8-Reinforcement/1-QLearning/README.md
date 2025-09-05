<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T16:37:04+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "sw"
}
-->
# Utangulizi wa Kujifunza kwa Kuimarisha na Q-Learning

![Muhtasari wa kujifunza kwa kuimarisha katika mashine ya kujifunza kwenye sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote na [Tomomi Imura](https://www.twitter.com/girlie_mac)

Kujifunza kwa kuimarisha kunahusisha dhana tatu muhimu: wakala, hali fulani, na seti ya vitendo kwa kila hali. Kwa kutekeleza kitendo katika hali maalum, wakala hupewa tuzo. Fikiria tena mchezo wa kompyuta wa Super Mario. Wewe ni Mario, uko katika kiwango cha mchezo, umesimama karibu na ukingo wa mwamba. Juu yako kuna sarafu. Wewe ukiwa Mario, katika kiwango cha mchezo, katika nafasi maalum ... hiyo ndiyo hali yako. Kusonga hatua moja kulia (kitendo) kutakupeleka kwenye ukingo, na hiyo itakupa alama ya chini ya nambari. Hata hivyo, kubonyeza kitufe cha kuruka kutakufanya upate alama na utaendelea kuishi. Hilo ni jambo chanya na linapaswa kukupa alama chanya ya nambari.

Kwa kutumia kujifunza kwa kuimarisha na simulator (mchezo), unaweza kujifunza jinsi ya kucheza mchezo ili kuongeza tuzo ambayo ni kuishi na kupata alama nyingi iwezekanavyo.

[![Utangulizi wa Kujifunza kwa Kuimarisha](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Bonyeza picha hapo juu kusikiliza Dmitry akijadili Kujifunza kwa Kuimarisha

## [Jaribio la awali la somo](https://ff-quizzes.netlify.app/en/ml/)

## Mahitaji na Usanidi

Katika somo hili, tutajaribu baadhi ya msimbo kwa kutumia Python. Unapaswa kuwa na uwezo wa kuendesha msimbo wa Jupyter Notebook kutoka somo hili, ama kwenye kompyuta yako au mahali fulani mtandaoni.

Unaweza kufungua [notebook ya somo](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) na kupitia somo hili ili kujifunza.

> **Note:** Ikiwa unafungua msimbo huu kutoka mtandaoni, unahitaji pia kupata faili [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), ambayo inatumika katika msimbo wa notebook. Ongeza faili hiyo kwenye saraka sawa na notebook.

## Utangulizi

Katika somo hili, tutachunguza ulimwengu wa **[Peter na Mbwa Mwitu](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, uliochochewa na hadithi ya muziki ya mtunzi wa Kirusi, [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Tutatumia **Kujifunza kwa Kuimarisha** kumruhusu Peter kuchunguza mazingira yake, kukusanya matufaha matamu na kuepuka kukutana na mbwa mwitu.

**Kujifunza kwa Kuimarisha** (RL) ni mbinu ya kujifunza inayoturuhusu kujifunza tabia bora ya **wakala** katika **mazingira** fulani kwa kufanya majaribio mengi. Wakala katika mazingira haya anapaswa kuwa na **lengo**, lililofafanuliwa na **kazi ya tuzo**.

## Mazingira

Kwa urahisi, hebu tuchukulie ulimwengu wa Peter kuwa ubao wa mraba wa ukubwa `width` x `height`, kama hivi:

![Mazingira ya Peter](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Kila seli katika ubao huu inaweza kuwa:

* **ardhi**, ambapo Peter na viumbe wengine wanaweza kutembea.
* **maji**, ambapo huwezi kutembea.
* **mti** au **nyasi**, mahali ambapo unaweza kupumzika.
* **tufaha**, ambalo linawakilisha kitu ambacho Peter angefurahia kukipata ili kujilisha.
* **mbwa mwitu**, ambaye ni hatari na anapaswa kuepukwa.

Kuna moduli ya Python tofauti, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), ambayo ina msimbo wa kufanya kazi na mazingira haya. Kwa sababu msimbo huu si muhimu kwa kuelewa dhana zetu, tutaleta moduli hiyo na kuitumia kuunda ubao wa mfano (msimbo wa block 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Msimbo huu unapaswa kuchapisha picha ya mazingira yanayofanana na ile hapo juu.

## Vitendo na sera

Katika mfano wetu, lengo la Peter litakuwa kupata tufaha, huku akiepuka mbwa mwitu na vikwazo vingine. Ili kufanya hivyo, anaweza kimsingi kutembea hadi apate tufaha.

Kwa hivyo, katika nafasi yoyote, anaweza kuchagua kati ya mojawapo ya vitendo vifuatavyo: juu, chini, kushoto na kulia.

Tutafafanua vitendo hivyo kama kamusi, na kuviunganisha na jozi za mabadiliko ya kuratibu yanayolingana. Kwa mfano, kusonga kulia (`R`) kungefanana na jozi `(1,0)`. (msimbo wa block 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Kwa muhtasari, mkakati na lengo la hali hii ni kama ifuatavyo:

- **Mkakati**, wa wakala wetu (Peter) unafafanuliwa na kinachoitwa **sera**. Sera ni kazi inayorejesha kitendo katika hali yoyote iliyotolewa. Katika kesi yetu, hali ya tatizo inawakilishwa na ubao, ikiwa ni pamoja na nafasi ya sasa ya mchezaji.

- **Lengo**, la kujifunza kwa kuimarisha ni hatimaye kujifunza sera nzuri ambayo itaturuhusu kutatua tatizo kwa ufanisi. Hata hivyo, kama msingi, hebu tuchukulie sera rahisi inayoitwa **kutembea bila mpangilio**.

## Kutembea bila mpangilio

Hebu kwanza tutatue tatizo letu kwa kutekeleza mkakati wa kutembea bila mpangilio. Kwa kutembea bila mpangilio, tutachagua kitendo kinachofuata bila mpangilio kutoka kwa vitendo vinavyoruhusiwa, hadi tufikie tufaha (msimbo wa block 3).

1. Tekeleza kutembea bila mpangilio kwa msimbo ulio hapa chini:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    Simu ya `walk` inapaswa kurejesha urefu wa njia inayolingana, ambayo inaweza kutofautiana kutoka kwa mzunguko mmoja hadi mwingine. 

1. Endesha jaribio la kutembea mara kadhaa (sema, 100), na uchapishe takwimu zinazotokana (msimbo wa block 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    Kumbuka kuwa urefu wa wastani wa njia ni karibu hatua 30-40, ambayo ni nyingi, ikizingatiwa kwamba umbali wa wastani hadi tufaha lililo karibu ni karibu hatua 5-6.

    Unaweza pia kuona jinsi Peter anavyosonga wakati wa kutembea bila mpangilio:

    ![Kutembea bila mpangilio kwa Peter](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Kazi ya tuzo

Ili kufanya sera yetu iwe na akili zaidi, tunahitaji kuelewa ni hatua zipi ni "bora" kuliko nyingine. Ili kufanya hivyo, tunahitaji kufafanua lengo letu.

Lengo linaweza kufafanuliwa kwa maneno ya **kazi ya tuzo**, ambayo itarejesha thamani fulani ya alama kwa kila hali. Kadri nambari inavyokuwa juu, ndivyo kazi ya tuzo inavyokuwa bora. (msimbo wa block 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

Jambo la kuvutia kuhusu kazi za tuzo ni kwamba katika hali nyingi, *tunapewa tuzo kubwa mwishoni mwa mchezo tu*. Hii inamaanisha kuwa algoriti yetu inapaswa kwa namna fulani kukumbuka hatua "nzuri" zinazopelekea tuzo chanya mwishoni, na kuongeza umuhimu wake. Vivyo hivyo, hatua zote zinazopelekea matokeo mabaya zinapaswa kupunguzwa.

## Q-Learning

Algoriti ambayo tutajadili hapa inaitwa **Q-Learning**. Katika algoriti hii, sera inafafanuliwa na kazi (au muundo wa data) inayoitwa **Q-Table**. Inarekodi "ubora" wa kila kitendo katika hali fulani.

Inaitwa Q-Table kwa sababu mara nyingi ni rahisi kuiwakilisha kama jedwali, au safu ya vipimo vingi. Kwa kuwa ubao wetu una vipimo `width` x `height`, tunaweza kuwakilisha Q-Table kwa kutumia safu ya numpy yenye umbo `width` x `height` x `len(actions)`: (msimbo wa block 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Kumbuka kuwa tunaanzisha maadili yote ya Q-Table na thamani sawa, katika kesi yetu - 0.25. Hii inafanana na sera ya "kutembea bila mpangilio", kwa sababu hatua zote katika kila hali ni sawa. Tunaweza kupitisha Q-Table kwa kazi ya `plot` ili kuonyesha jedwali kwenye ubao: `m.plot(Q)`.

![Mazingira ya Peter](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Katikati ya kila seli kuna "mshale" unaoonyesha mwelekeo unaopendelewa wa harakati. Kwa kuwa mwelekeo wote ni sawa, nukta inaonyeshwa.

Sasa tunahitaji kuendesha simulizi, kuchunguza mazingira yetu, na kujifunza usambazaji bora wa maadili ya Q-Table, ambayo yataturuhusu kupata njia ya kufikia tufaha haraka zaidi.

## Kiini cha Q-Learning: Mlinganyo wa Bellman

Mara tu tunapoanza kusonga, kila kitendo kitakuwa na tuzo inayolingana, yaani tunaweza kinadharia kuchagua kitendo kinachofuata kulingana na tuzo ya haraka zaidi. Hata hivyo, katika hali nyingi, hatua hiyo haitafanikisha lengo letu la kufikia tufaha, na hivyo hatuwezi kuamua mara moja ni mwelekeo gani ni bora.

> Kumbuka kuwa si matokeo ya haraka yanayojalisha, bali ni matokeo ya mwisho, ambayo tutapata mwishoni mwa simulizi.

Ili kuzingatia tuzo hii ya kuchelewa, tunahitaji kutumia kanuni za **[programu ya nguvu](https://en.wikipedia.org/wiki/Dynamic_programming)**, ambayo inaturuhusu kufikiria tatizo letu kwa njia ya kurudia.

Tuseme sasa tuko katika hali *s*, na tunataka kusonga hadi hali inayofuata *s'*. Kwa kufanya hivyo, tutapokea tuzo ya haraka *r(s,a)*, iliyofafanuliwa na kazi ya tuzo, pamoja na tuzo fulani ya baadaye. Ikiwa tunadhani kwamba Q-Table yetu inaonyesha kwa usahihi "uvutaji" wa kila kitendo, basi katika hali *s'* tutachagua kitendo *a* kinacholingana na thamani ya juu zaidi ya *Q(s',a')*. Kwa hivyo, tuzo bora zaidi ya baadaye ambayo tunaweza kupata katika hali *s* itafafanuliwa kama `max`

## Kukagua sera

Kwa kuwa Q-Table inaorodhesha "mvuto" wa kila kitendo katika kila hali, ni rahisi kuitumia kufafanua urambazaji bora katika ulimwengu wetu. Katika hali rahisi, tunaweza kuchagua kitendo kinacholingana na thamani ya juu zaidi ya Q-Table: (code block 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Ukijaribu msimbo hapo juu mara kadhaa, unaweza kugundua kuwa wakati mwingine "inakwamia", na unahitaji kubonyeza kitufe cha STOP kwenye daftari ili kuisimamisha. Hii hutokea kwa sababu kunaweza kuwa na hali ambapo hali mbili "zinaelekeza" kwa kila moja kwa mujibu wa thamani bora ya Q-Value, ambapo wakala huishia kusonga kati ya hali hizo bila kikomo.

## 🚀Changamoto

> **Kazi ya 1:** Badilisha kazi ya `walk` ili kupunguza urefu wa njia kwa idadi fulani ya hatua (sema, 100), na angalia msimbo hapo juu ukirudisha thamani hii mara kwa mara.

> **Kazi ya 2:** Badilisha kazi ya `walk` ili isirudi kwenye maeneo ambayo tayari imekuwa hapo awali. Hii itazuia `walk` kurudia, hata hivyo, wakala bado anaweza kujikuta "amekwama" mahali ambapo hawezi kutoroka.

## Urambazaji

Sera bora ya urambazaji itakuwa ile tuliyotumia wakati wa mafunzo, ambayo inachanganya unyonyaji na uchunguzi. Katika sera hii, tutachagua kila kitendo kwa uwezekano fulani, kulingana na thamani katika Q-Table. Mkakati huu bado unaweza kusababisha wakala kurudi kwenye nafasi ambayo tayari imechunguza, lakini, kama unavyoona kutoka kwa msimbo hapa chini, husababisha njia fupi sana kwa wastani kuelekea eneo linalotakiwa (kumbuka kuwa `print_statistics` inaendesha simulizi mara 100): (code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Baada ya kuendesha msimbo huu, unapaswa kupata urefu wa njia ya wastani ndogo zaidi kuliko hapo awali, katika kiwango cha 3-6.

## Kuchunguza mchakato wa kujifunza

Kama tulivyotaja, mchakato wa kujifunza ni usawa kati ya uchunguzi na unyonyaji wa maarifa yaliyopatikana kuhusu muundo wa nafasi ya tatizo. Tumeona kuwa matokeo ya kujifunza (uwezo wa kusaidia wakala kupata njia fupi kuelekea lengo) yameboreshwa, lakini pia ni ya kuvutia kuangalia jinsi urefu wa njia ya wastani unavyobadilika wakati wa mchakato wa kujifunza:

## Muhtasari wa mafunzo:

- **Urefu wa njia ya wastani huongezeka**. Tunachokiona hapa ni kwamba mwanzoni, urefu wa njia ya wastani huongezeka. Hii labda ni kwa sababu tunapokuwa hatujui chochote kuhusu mazingira, tuna uwezekano wa kukwama katika hali mbaya, maji au mbwa mwitu. Tunapojifunza zaidi na kuanza kutumia maarifa haya, tunaweza kuchunguza mazingira kwa muda mrefu, lakini bado hatujui vizuri mahali ambapo matunda yapo.

- **Urefu wa njia hupungua, tunapojifunza zaidi**. Mara tu tunapojifunza vya kutosha, inakuwa rahisi kwa wakala kufikia lengo, na urefu wa njia huanza kupungua. Hata hivyo, bado tunafungua uchunguzi, kwa hivyo mara nyingi tunatoka kwenye njia bora, na kuchunguza chaguo mpya, na kufanya njia kuwa ndefu zaidi kuliko ilivyo bora.

- **Urefu huongezeka ghafla**. Tunachokiona pia kwenye grafu hii ni kwamba wakati fulani, urefu uliongezeka ghafla. Hii inaonyesha asili ya mchakato wa nasibu, na kwamba tunaweza wakati fulani "kuharibu" coefficients za Q-Table kwa kuandika upya na thamani mpya. Hii inapaswa kupunguzwa kwa kupunguza kiwango cha kujifunza (kwa mfano, kuelekea mwisho wa mafunzo, tunarekebisha thamani za Q-Table kwa kiasi kidogo).

Kwa ujumla, ni muhimu kukumbuka kuwa mafanikio na ubora wa mchakato wa kujifunza hutegemea sana vigezo, kama kiwango cha kujifunza, kupungua kwa kiwango cha kujifunza, na sababu ya punguzo. Hizi mara nyingi huitwa **vigezo vya juu**, ili kuzitofautisha na **vigezo**, ambavyo tunaboresha wakati wa mafunzo (kwa mfano, coefficients za Q-Table). Mchakato wa kutafuta thamani bora za vigezo vya juu huitwa **ubunifu wa vigezo vya juu**, na unastahili mada tofauti.

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Kazi 
[Dunia Halisi Zaidi](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.