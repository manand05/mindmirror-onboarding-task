"""
STUDENT NAME: Maher
STUDENT ID:
"""

from transformers import pipeline

# Load your chosen models here
def load_emotion_model():

    emotion_classifier = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student", top_k=None)
    return emotion_classifier

def load_summarization_model():

    text_summarizer = pipeline("summarization", model="google/flan-t5-base")
    return text_summarizer

# Process journal entries and return emotion predictions
def detect_emotions(text_entries, emotion_model):
    results = []

    for entry in text_entries:
        analysis = emotion_model(entry)

        # Initialize a list to store emotions for this entry
        filtered_emotions = []

        # Loop through the list of predictions
        for prediction in analysis[0]:
            confidence_score = prediction["score"]
            emotion_label = prediction["label"]

            # Only keep emotions with a confidence score higher than 0.3
            if confidence_score > 0.1:
                filtered_emotions.append(emotion_label.lower())

        # Append the results for this entry
        results.append({"journal_entry": entry, "emotions": filtered_emotions})

    return results



# Generate summaries for journal entries
def summarize_entries(entries, summarizer):
    results = []

    for entry in entries:
        input_length = len(entry.split())
        max_summary_length = max(15, int(input_length * 0.6))

        # generate summary
        summary = summarizer("Summarize: " + entry, max_length=max_summary_length, min_length=int(max_summary_length * 0.7),
            do_sample=True,
            temperature=0.7
        )

        results.append({"journal_entry": entry, "summary": summary[0]['summary_text']})

    return results

if __name__ == '__main__':
    # Load models
    emotion_model = load_emotion_model()
    summarizer = load_summarization_model()

    # Example input
    journal_entries = [
        "Felt anxious about my exam, but happy after completing it.",
        "It rained all day and I stayed inside feeling calm.",
        "Woke up feeling tired, but a morning walk lifted my mood.",
        "Had a productive day at work, though a bit stressed by deadlines.",
        "Met up with an old friend—felt nostalgic and happy reminiscing.",
        "Struggled to focus today, feeling frustrated and overwhelmed.",
        "Enjoyed a quiet evening reading, felt peaceful and content.",
        "Got some unexpected good news, which made my day brighter.",
        "Felt lonely tonight, but a phone call with family helped.",
        "Went to the gym despite feeling unmotivated—felt accomplished afterward."
    ]

    # Apply pipelines
    emotions = detect_emotions(journal_entries, emotion_model)
    summaries = summarize_entries(journal_entries, summarizer)

    # Output results
    print("Emotion Predictions:", emotions)
    print("Summaries:", summaries)