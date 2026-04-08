# src/extractive_summarizer.py

from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def summarize_extractive(text, num_sentences=3):
    """
    Performs extractive text summarization by selecting the most important sentences.
    This is a simplified extractive summarizer using a sentence scoring approach.

    Args:
        text (str): The input text to summarize.
        num_sentences (int): The number of sentences to extract for the summary.

    Returns:
        str: The generated extractive summary.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return ""

    # A very simple scoring: sentences with more keywords or at the beginning are more important
    # In a real scenario, this would involve TF-IDF, TextRank, or other NLP techniques.
    keywords = ["AI", "intelligence", "machines", "humans", "learning", "problem solving", "technologies", "risks", "responsibly", "ethically"]
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        score = sum(1 for word in keywords if word.lower() in sentence.lower())
        # Boost score for sentences appearing earlier
        score += (len(sentences) - i) / len(sentences) * 2 # Give more weight to earlier sentences
        sentence_scores.append((score, sentence))

    # Sort sentences by score in descending order
    sentence_scores.sort(key=lambda x: x[0], reverse=True)

    # Select the top N sentences and maintain original order
    selected_sentences = sorted(sentence_scores[:num_sentences], key=lambda x: sentences.index(x[1]))
    return ' '.join([s[1] for s in selected_sentences])

if __name__ == '__main__':
    sample_text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence "
        "displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": "
        "any device that perceives its environment and takes actions that maximize its chance of successfully "
        "achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines "
        "that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "
        ""problem solving".

        Capabilities of AI include: successful understanding of human speech, competing at the highest level in "
        "strategic game systems (such as chess and Go), autonomous cars, intelligent routing in content delivery "
        "networks, and military simulations. AI is one of the most important technologies of the 21st century. "
        "It has the potential to solve many of the world\'s most pressing problems, from climate change to disease. "
        "However, it also poses significant risks, such as job displacement and the potential for autonomous weapons. "
        "It is therefore crucial to develop AI responsibly and ethically."
    )
    print("Original Text:
", sample_text)
    print("
Extractive Summary (3 sentences):
", summarize_extractive(sample_text, num_sentences=3))
