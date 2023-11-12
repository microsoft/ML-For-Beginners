# Natural Language Toolkit: Sun Tsu-Bot
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Sam Huston 2007
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Tsu bot responds to all queries with a Sun Tsu sayings

Quoted from Sun Tsu's The Art of War
Translated by LIONEL GILES, M.A. 1910
Hosted by the Gutenberg Project
https://www.gutenberg.org/
"""

from nltk.chat.util import Chat, reflections

pairs = (
    (r"quit", ("Good-bye.", "Plan well", "May victory be your future")),
    (
        r"[^\?]*\?",
        (
            "Please consider whether you can answer your own question.",
            "Ask me no questions!",
        ),
    ),
    (
        r"[0-9]+(.*)",
        (
            "It is the rule in war, if our forces are ten to the enemy's one, to surround him; if five to one, to attack him; if twice as numerous, to divide our army into two.",
            "There are five essentials for victory",
        ),
    ),
    (
        r"[A-Ca-c](.*)",
        (
            "The art of war is of vital importance to the State.",
            "All warfare is based on deception.",
            "If your opponent is secure at all points, be prepared for him. If he is in superior strength, evade him.",
            "If the campaign is protracted, the resources of the State will not be equal to the strain.",
            "Attack him where he is unprepared, appear where you are not expected.",
            "There is no instance of a country having benefited from prolonged warfare.",
        ),
    ),
    (
        r"[D-Fd-f](.*)",
        (
            "The skillful soldier does not raise a second levy, neither are his supply-wagons loaded more than twice.",
            "Bring war material with you from home, but forage on the enemy.",
            "In war, then, let your great object be victory, not lengthy campaigns.",
            "To fight and conquer in all your battles is not supreme excellence; supreme excellence consists in breaking the enemy's resistance without fighting.",
        ),
    ),
    (
        r"[G-Ig-i](.*)",
        (
            "Heaven signifies night and day, cold and heat, times and seasons.",
            "It is the rule in war, if our forces are ten to the enemy's one, to surround him; if five to one, to attack him; if twice as numerous, to divide our army into two.",
            "The good fighters of old first put themselves beyond the possibility of defeat, and then waited for an opportunity of defeating the enemy.",
            "One may know how to conquer without being able to do it.",
        ),
    ),
    (
        r"[J-Lj-l](.*)",
        (
            "There are three ways in which a ruler can bring misfortune upon his army.",
            "By commanding the army to advance or to retreat, being ignorant of the fact that it cannot obey. This is called hobbling the army.",
            "By attempting to govern an army in the same way as he administers a kingdom, being ignorant of the conditions which obtain in an army. This causes restlessness in the soldier's minds.",
            "By employing the officers of his army without discrimination, through ignorance of the military principle of adaptation to circumstances. This shakes the confidence of the soldiers.",
            "There are five essentials for victory",
            "He will win who knows when to fight and when not to fight.",
            "He will win who knows how to handle both superior and inferior forces.",
            "He will win whose army is animated by the same spirit throughout all its ranks.",
            "He will win who, prepared himself, waits to take the enemy unprepared.",
            "He will win who has military capacity and is not interfered with by the sovereign.",
        ),
    ),
    (
        r"[M-Om-o](.*)",
        (
            "If you know the enemy and know yourself, you need not fear the result of a hundred battles.",
            "If you know yourself but not the enemy, for every victory gained you will also suffer a defeat.",
            "If you know neither the enemy nor yourself, you will succumb in every battle.",
            "The control of a large force is the same principle as the control of a few men: it is merely a question of dividing up their numbers.",
        ),
    ),
    (
        r"[P-Rp-r](.*)",
        (
            "Security against defeat implies defensive tactics; ability to defeat the enemy means taking the offensive.",
            "Standing on the defensive indicates insufficient strength; attacking, a superabundance of strength.",
            "He wins his battles by making no mistakes. Making no mistakes is what establishes the certainty of victory, for it means conquering an enemy that is already defeated.",
            "A victorious army opposed to a routed one, is as a pound's weight placed in the scale against a single grain.",
            "The onrush of a conquering force is like the bursting of pent-up waters into a chasm a thousand fathoms deep.",
        ),
    ),
    (
        r"[S-Us-u](.*)",
        (
            "What the ancients called a clever fighter is one who not only wins, but excels in winning with ease.",
            "Hence his victories bring him neither reputation for wisdom nor credit for courage.",
            "Hence the skillful fighter puts himself into a position which makes defeat impossible, and does not miss the moment for defeating the enemy.",
            "In war the victorious strategist only seeks battle after the victory has been won, whereas he who is destined to defeat first fights and afterwards looks for victory.",
            "There are not more than five musical notes, yet the combinations of these five give rise to more melodies than can ever be heard.",
            "Appear at points which the enemy must hasten to defend; march swiftly to places where you are not expected.",
        ),
    ),
    (
        r"[V-Zv-z](.*)",
        (
            "It is a matter of life and death, a road either to safety or to ruin.",
            "Hold out baits to entice the enemy. Feign disorder, and crush him.",
            "All men can see the tactics whereby I conquer, but what none can see is the strategy out of which victory is evolved.",
            "Do not repeat the tactics which have gained you one victory, but let your methods be regulated by the infinite variety of circumstances.",
            "So in war, the way is to avoid what is strong and to strike at what is weak.",
            "Just as water retains no constant shape, so in warfare there are no constant conditions.",
        ),
    ),
    (r"(.*)", ("Your statement insults me.", "")),
)

suntsu_chatbot = Chat(pairs, reflections)


def suntsu_chat():
    print("Talk to the program by typing in plain English, using normal upper-")
    print('and lower-case letters and punctuation.  Enter "quit" when done.')
    print("=" * 72)
    print("You seek enlightenment?")

    suntsu_chatbot.converse()


def demo():
    suntsu_chat()


if __name__ == "__main__":
    demo()
