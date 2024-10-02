from transformers import pipeline

# Initialize the zero-shot-classification pipeline for intent recognition
intent_classifier = pipeline("zero-shot-classification")

# The sentence you want to analyze
sentence = "Repeat the question"

# Define possible intents
intents = ["repeat question", "clarify", "ask another question", "move on", "other"]

# Analyze the intent
result = intent_classifier(sentence, candidate_labels=intents)

# Print the results
print(result)
