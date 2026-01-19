from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
extracctor = ConllExtractor();


def mani():
    print("Hello, I am Paskal, the friendly robot")
    print("You can end the conversation at any time typing  'bye'")
    print("After typing each sentence presthe Enter for the sentiment analysis")
    print("How are you feeling today?")

    while True:
        user_input = input("You:")
        if user_input.lower() in ('bye', "exit", "quit"):
            break

        user_blob = TextBlob(user_input, np_extractor=extracctor)
        polarity = user_blob.polarity
        noun_phrases = user_blob.noun_phrases

        if polarity <= -0.5:
            response = "Oh Dear, that sounds really bad!"
        elif polarity <= 0:
            response = "Hmm, that's not great."
        elif polarity < 0.5:
            response = "Thats sound positive! "
        else:
            response = "Yay! That sounds awesome!"
        if noun_phrases:
            np = noun_phrases[0]
            try:
                plural_np = np.pluralize()
            except Exception:
                plural_np = np
            response += f"Can you tell me more about {plural_np}?"
        else:
            response += "Can you tell me more about that?"
        print(response)
    print("Goodbye! It was nice talking to you.")
mani()
