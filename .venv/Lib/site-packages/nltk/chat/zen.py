# Natural Language Toolkit: Zen Chatbot
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Amy Holland <amyrh@csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Zen Chatbot talks in gems of Zen wisdom.

This is a sample conversation with Zen Chatbot:
ZC:    Welcome, my child.
me:    Good afternoon.
ZC:    Ask the question you have come to ask.
me:    How can I achieve enlightenment?
ZC:    How do you suppose?
me:    Through meditation.
ZC:    Form is emptiness, and emptiness form.
me:    How can I empty my mind of worldly troubles?
ZC:    Will an answer to that really help in your search for enlightenment?
me:    Yes.
ZC:    It is better to be right than to be certain.
me:    I seek truth and wisdom.
ZC:    The search for truth is a long journey.
me:    Are you sure?
ZC:    Maybe sure, maybe not sure.


The chatbot structure is based on that of chat.eliza. Thus, it uses
a translation table to convert from question to response
i.e. "I am" --> "you are"

Of course, since Zen Chatbot does not understand the meaning of any words,
responses are very limited. Zen Chatbot will usually answer very vaguely, or
respond to a question by asking a different question, in much the same way
as Eliza.
"""

from nltk.chat.util import Chat, reflections

# responses are matched top to bottom, so non-specific matches occur later
# for each match, a list of possible responses is provided
responses = (
    # Zen Chatbot opens with the line "Welcome, my child." The usual
    # response will be a greeting problem: 'good' matches "good morning",
    # "good day" etc, but also "good grief!"  and other sentences starting
    # with the word 'good' that may not be a greeting
    (
        r"(hello(.*))|(good [a-zA-Z]+)",
        (
            "The path to enlightenment is often difficult to see.",
            "Greetings. I sense your mind is troubled. Tell me of your troubles.",
            "Ask the question you have come to ask.",
            "Hello. Do you seek englightenment?",
        ),
    ),
    # "I need" and "I want" can be followed by a thing (eg 'help')
    # or an action (eg 'to see you')
    #
    # This is a problem with this style of response -
    # person:    "I need you"
    # chatbot:    "me can be achieved by hard work and dedication of the mind"
    # i.e. 'you' is not really a thing that can be mapped this way, so this
    # interpretation only makes sense for some inputs
    #
    (
        r"i need (.*)",
        (
            "%1 can be achieved by hard work and dedication of the mind.",
            "%1 is not a need, but a desire of the mind. Clear your mind of such concerns.",
            "Focus your mind on%1, and you will find what you need.",
        ),
    ),
    (
        r"i want (.*)",
        (
            "Desires of the heart will distract you from the path to enlightenment.",
            "Will%1 help you attain enlightenment?",
            "Is%1 a desire of the mind, or of the heart?",
        ),
    ),
    # why questions are separated into three types:
    # "why..I"     e.g. "why am I here?" "Why do I like cake?"
    # "why..you"    e.g. "why are you here?" "Why won't you tell me?"
    # "why..."    e.g. "Why is the sky blue?"
    # problems:
    #     person:  "Why can't you tell me?"
    #     chatbot: "Are you sure I tell you?"
    # - this style works for positives (e.g. "why do you like cake?")
    #   but does not work for negatives (e.g. "why don't you like cake?")
    (r"why (.*) i (.*)\?", ("You%1%2?", "Perhaps you only think you%1%2")),
    (r"why (.*) you(.*)\?", ("Why%1 you%2?", "%2 I%1", "Are you sure I%2?")),
    (r"why (.*)\?", ("I cannot tell you why%1.", "Why do you think %1?")),
    # e.g. "are you listening?", "are you a duck"
    (
        r"are you (.*)\?",
        ("Maybe%1, maybe not%1.", "Whether I am%1 or not is God's business."),
    ),
    # e.g. "am I a duck?", "am I going to die?"
    (
        r"am i (.*)\?",
        ("Perhaps%1, perhaps not%1.", "Whether you are%1 or not is not for me to say."),
    ),
    # what questions, e.g. "what time is it?"
    # problems:
    #     person:  "What do you want?"
    #    chatbot: "Seek truth, not what do me want."
    (r"what (.*)\?", ("Seek truth, not what%1.", "What%1 should not concern you.")),
    # how questions, e.g. "how do you do?"
    (
        r"how (.*)\?",
        (
            "How do you suppose?",
            "Will an answer to that really help in your search for enlightenment?",
            "Ask yourself not how, but why.",
        ),
    ),
    # can questions, e.g. "can you run?", "can you come over here please?"
    (
        r"can you (.*)\?",
        (
            "I probably can, but I may not.",
            "Maybe I can%1, and maybe I cannot.",
            "I can do all, and I can do nothing.",
        ),
    ),
    # can questions, e.g. "can I have some cake?", "can I know truth?"
    (
        r"can i (.*)\?",
        (
            "You can%1 if you believe you can%1, and have a pure spirit.",
            "Seek truth and you will know if you can%1.",
        ),
    ),
    # e.g. "It is raining" - implies the speaker is certain of a fact
    (
        r"it is (.*)",
        (
            "How can you be certain that%1, when you do not even know yourself?",
            "Whether it is%1 or not does not change the way the world is.",
        ),
    ),
    # e.g. "is there a doctor in the house?"
    (
        r"is there (.*)\?",
        ("There is%1 if you believe there is.", "It is possible that there is%1."),
    ),
    # e.g. "is it possible?", "is this true?"
    (r"is(.*)\?", ("%1 is not relevant.", "Does this matter?")),
    # non-specific question
    (
        r"(.*)\?",
        (
            "Do you think %1?",
            "You seek the truth. Does the truth seek you?",
            "If you intentionally pursue the answers to your questions, the answers become hard to see.",
            "The answer to your question cannot be told. It must be experienced.",
        ),
    ),
    # expression of hate of form "I hate you" or "Kelly hates cheese"
    (
        r"(.*) (hate[s]?)|(dislike[s]?)|(don\'t like)(.*)",
        (
            "Perhaps it is not about hating %2, but about hate from within.",
            "Weeds only grow when we dislike them",
            "Hate is a very strong emotion.",
        ),
    ),
    # statement containing the word 'truth'
    (
        r"(.*) truth(.*)",
        (
            "Seek truth, and truth will seek you.",
            "Remember, it is not the spoon which bends - only yourself.",
            "The search for truth is a long journey.",
        ),
    ),
    # desire to do an action
    # e.g. "I want to go shopping"
    (
        r"i want to (.*)",
        ("You may %1 if your heart truly desires to.", "You may have to %1."),
    ),
    # desire for an object
    # e.g. "I want a pony"
    (
        r"i want (.*)",
        (
            "Does your heart truly desire %1?",
            "Is this a desire of the heart, or of the mind?",
        ),
    ),
    # e.g. "I can't wait" or "I can't do this"
    (
        r"i can\'t (.*)",
        (
            "What we can and can't do is a limitation of the mind.",
            "There are limitations of the body, and limitations of the mind.",
            "Have you tried to%1 with a clear mind?",
        ),
    ),
    # "I think.." indicates uncertainty. e.g. "I think so."
    # problem: exceptions...
    # e.g. "I think, therefore I am"
    (
        r"i think (.*)",
        (
            "Uncertainty in an uncertain world.",
            "Indeed, how can we be certain of anything in such uncertain times.",
            "Are you not, in fact, certain that%1?",
        ),
    ),
    # "I feel...emotions/sick/light-headed..."
    (
        r"i feel (.*)",
        (
            "Your body and your emotions are both symptoms of your mind."
            "What do you believe is the root of such feelings?",
            "Feeling%1 can be a sign of your state-of-mind.",
        ),
    ),
    # exclaimation mark indicating emotion
    # e.g. "Wow!" or "No!"
    (
        r"(.*)!",
        (
            "I sense that you are feeling emotional today.",
            "You need to calm your emotions.",
        ),
    ),
    # because [statement]
    # e.g. "because I said so"
    (
        r"because (.*)",
        (
            "Does knowning the reasons behind things help you to understand"
            " the things themselves?",
            "If%1, what else must be true?",
        ),
    ),
    # yes or no - raise an issue of certainty/correctness
    (
        r"(yes)|(no)",
        (
            "Is there certainty in an uncertain world?",
            "It is better to be right than to be certain.",
        ),
    ),
    # sentence containing word 'love'
    (
        r"(.*)love(.*)",
        (
            "Think of the trees: they let the birds perch and fly with no intention to call them when they come, and no longing for their return when they fly away. Let your heart be like the trees.",
            "Free love!",
        ),
    ),
    # sentence containing word 'understand' - r
    (
        r"(.*)understand(.*)",
        (
            "If you understand, things are just as they are;"
            " if you do not understand, things are just as they are.",
            "Imagination is more important than knowledge.",
        ),
    ),
    # 'I', 'me', 'my' - person is talking about themself.
    # this breaks down when words contain these - eg 'Thyme', 'Irish'
    (
        r"(.*)(me )|( me)|(my)|(mine)|(i)(.*)",
        (
            "'I', 'me', 'my'... these are selfish expressions.",
            "Have you ever considered that you might be a selfish person?",
            "Try to consider others, not just yourself.",
            "Think not just of yourself, but of others.",
        ),
    ),
    # 'you' starting a sentence
    # e.g. "you stink!"
    (
        r"you (.*)",
        ("My path is not of concern to you.", "I am but one, and you but one more."),
    ),
    # say goodbye with some extra Zen wisdom.
    (
        r"exit",
        (
            "Farewell. The obstacle is the path.",
            "Farewell. Life is a journey, not a destination.",
            "Good bye. We are cups, constantly and quietly being filled."
            "\nThe trick is knowning how to tip ourselves over and let the beautiful stuff out.",
        ),
    ),
    # fall through case -
    # when stumped, respond with generic zen wisdom
    #
    (
        r"(.*)",
        (
            "When you're enlightened, every word is wisdom.",
            "Random talk is useless.",
            "The reverse side also has a reverse side.",
            "Form is emptiness, and emptiness is form.",
            "I pour out a cup of water. Is the cup empty?",
        ),
    ),
)

zen_chatbot = Chat(responses, reflections)


def zen_chat():
    print("*" * 75)
    print("Zen Chatbot!".center(75))
    print("*" * 75)
    print('"Look beyond mere words and letters - look into your mind"'.center(75))
    print("* Talk your way to truth with Zen Chatbot.")
    print("* Type 'quit' when you have had enough.")
    print("*" * 75)
    print("Welcome, my child.")

    zen_chatbot.converse()


def demo():
    zen_chat()


if __name__ == "__main__":
    demo()
