"""
STUDENT NAME:
STUDENT ID:
"""

from transformers import pipeline

# Load your chosen models here
def load_emotion_model():
    # Research and return a pre-trained emotion detection pipeline
    pass

def load_summarization_model():
    # Research and return a pre-trained summarization pipeline
    pass

# Process journal entries and return emotion predictions
def detect_emotions(text_entries, emotion_model):
    # Implement logic to process entries and extract emotions
    pass

# Generate summaries for journal entries
def summarize_entries(entries, summarizer):
    # Implement logic to summarize each entry
    pass

if __name__ == '__main__':
    # Load models
    emotion_model = load_emotion_model()
    summarizer = load_summarization_model()

    # Example input
    journal_entries = [
        "Felt anxious about my exam, but happy after completing it.",
        "It rained all day and I stayed inside feeling calm.",
    ]

    # Apply pipelines
    emotions = detect_emotions(journal_entries, emotion_model)
    summaries = summarize_entries(journal_entries, summarizer)

    # Output results
    print("Emotion Predictions:", emotions)
    print("Summaries:", summaries)