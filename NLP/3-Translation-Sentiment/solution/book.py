from textblob import TextBlob

# You should download the book text, clean it, and import it here
with open("pride.txt", encoding="utf8") as f:
    file_contents = f.read()

book_pride = TextBlob(file_contents)
positive_sentiment_sentences = []
negative_sentiment_sentences = []

for sentence in book_pride.sentences:
    if sentence.sentiment.polarity == 1:
        positive_sentiment_sentences.append(sentence)
    if sentence.sentiment.polarity == -1:
        negative_sentiment_sentences.append(sentence)

print("The " + str(len(positive_sentiment_sentences)) + " most positive sentences:")
for sentence in positive_sentiment_sentences:
    print("+ " + str(sentence.replace("\n", "").replace("      ", " ")))

    print("The " + str(len(negative_sentiment_sentences)) + " most negative sentences:")
for sentence in negative_sentiment_sentences:
    print("- " + str(sentence.replace("\n", "").replace("      ", " ")))
