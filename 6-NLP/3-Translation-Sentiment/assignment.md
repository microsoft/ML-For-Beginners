# Poetic license

## Instructions

Use [Emily Dickinson’s poems from PoetryDB](https://poetrydb.org/author/Emily%20Dickinson) to explore how well a sentiment score captures the tone of a poem. The response contains each poem’s title and lines. Save the linked JSON as `dickinson.json` next to your notebook and load it with Python’s standard library:

```python
import json

with open("dickinson.json", encoding="utf-8") as file:
    poems = json.load(file)

print(poems[0]["title"])
print("\n".join(poems[0]["lines"]))
```

1. Choose three poems. Before scoring them, read each one and record whether you find its tone positive, negative, mixed, or neutral. Quote a line that supports your reading.
2. Join each poem’s lines into a string and use the lesson’s `TextBlob(text).sentiment` to get its polarity and subjectivity. Record the two scores alongside the title and your interpretation.
3. Compare your readings with the scores. Which words might explain a score? Where does the poem’s meaning depend on context the score misses?

If you have access to Azure sentiment analysis, you can also submit the same poems and compare its predicted labels with your readings and TextBlob’s polarity.

## Rubric

| Criteria | Exemplary                                                                  | Adequate                                                | Needs Improvement        |
| -------- | -------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------ |
|          | A notebook is presented with a solid analysis of an author's sample output | The notebook is incomplete or does not perform analysis | No notebook is presented |
