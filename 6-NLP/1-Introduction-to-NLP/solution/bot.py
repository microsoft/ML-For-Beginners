import random
import string
# New Update chatbot code with intent detection and state management
INTENTS = {
   "greeting": ["hello", "hi", "hey"],
    "question": ["why", "what", "how", "is", "are", "?"],
    "opinion": ["i think", "i believe", "in my opinion"],
    "goodbye": ["bye", "exit", "quit"]
}
responses ={
     "greeting": [
        "Hello. How can I assist you today?"
    ],
    "question": [
        "That is a thoughtful question. Could you clarify further?",
        "Let us explore that in more detail."
    ],
    "opinion": [
        "That is an interesting perspective. What led you to that conclusion?"
    ],
    "unknown": [
        "I am not entirely sure I understand. Could you rephrase?",
        "Could you provide more detail so I can respond appropriately?"
    ]
}
conversation_state = {
    "last_intent": None,
    "last_topic": None
}
def normalize(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))
def detect_intent(text):
    for intent, patterns in INTENTS.items():
        for pattern in patterns:
            if pattern in text:
                return intent
    return "unknown"

def extract_topic(text):
    keywords = ["coding", "programming", "money", "career", "finance"]
    for word in keywords:
        if word in text:
            return word
    return None

def generate_response(intent, topic):
    base_response = random.choice(responses.get(intent, responses["unknown"]))
    if topic:
        return f"{base_response} You mentioned {topic}, Would you like to discuss it further?"
    return base_response
print("Hello, I am Paskal Chatbot, the conversational AI.")
print("Type 'bye' to end the conversation.")
while True:
    user_input = input("> ")
    normalized_input = normalize(user_input)
    if normalized_input == "bye":
        print("It was nice talking to you, goodbye!")
        break
    intent = detect_intent(normalized_input)
    topic = extract_topic(normalized_input)
    conversation_state["last_intent"] = intent
    if topic:
        conversation_state["last_topic"] = topic
    response = generate_response(intent, topic)
    print(response)

    # This list contains the random responses (you can add your own or translate them into your own language too)
# random_responses = ["That is quite interesting, please tell me more.",
#                     "I see. Do go on.",
#                     "Why do you say that?",
#                     "Funny weather we've been having, isn't it?",
#                     "Let's change the subject.",
#                     "Did you catch the game last night?"]

# print("Hello, I am Marvin, the simple robot.")
# print("You can end this conversation at any time by typing 'bye'")
# print("After typing each answer, press 'enter'")
# print("How are you today?")

# while True:
#     # wait for the user to enter some text
#     user_input = input("> ")
#     if user_input.lower() == "bye":
#         # if they typed in 'bye' (or even BYE, ByE, byE etc.), break out of the loop
#         break
#     else:
#         response = random.choices(random_responses)[0]
#     print(response)

# print("It was nice talking to you, goodbye!")
