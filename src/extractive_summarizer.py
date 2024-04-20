import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import heapq
import logging

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ExtractiveSummarizer:
    """
    A class for performing extractive text summarization. This method works by
    identifying the most important sentences in the original text and combining
    them to form a summary. It uses a frequency-based approach to score sentences.
    """
    def __init__(self, language=\'english\'):
        """
        Initializes the summarizer with a specified language for stopwords.
        
        Args:
            language (str): The language for stopwords. Defaults to \'english\'.
        """
        self.stop_words = set(stopwords.words(language))
        logger.info(f"ExtractiveSummarizer initialized with language: {language}")

    def _create_frequency_table(self, text):
        """
        Creates a frequency table of words in the text, excluding stopwords.
        
        Args:
            text (str): The input text.
            
        Returns:
            dict: A dictionary where keys are words and values are their frequencies.
        """
        words = word_tokenize(text)
        frequency_table = defaultdict(int)
        for word in words:
            word = word.lower()
            if word not in self.stop_words and word.isalpha(): # Only consider alphabetic words
                frequency_table[word] += 1
        return frequency_table

    def _score_sentences(self, sentences, frequency_table):
        """
        Scores each sentence based on the frequency of its words.
        
        Args:
            sentences (list): A list of sentences.
            frequency_table (dict): The word frequency table.
            
        Returns:
            dict: A dictionary where keys are sentences and values are their scores.
        """
        sentence_scores = defaultdict(int)
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in frequency_table:
                    sentence_scores[sentence] += frequency_table[word]
        return sentence_scores

    def summarize(self, text, num_sentences=5):
        """
        Generates an extractive summary for the given text.
        
        Args:
            text (str): The input text to summarize.
            num_sentences (int): The desired number of sentences in the summary.
            
        Returns:
            str: The generated extractive summary.
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text provided. Returning empty string.")
            return ""

        logger.info(f"Generating extractive summary for text of length: {len(text)} characters.")
        
        # 1. Tokenize text into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return ""

        # 2. Create word frequency table
        frequency_table = self._create_frequency_table(text)

        # 3. Score sentences
        sentence_scores = self._score_sentences(sentences, frequency_table)

        # 4. Get the top N sentences based on score
        # Using heapq.nlargest to get the top sentences efficiently
        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

        # 5. Join the selected sentences to form the summary
        summary = \' \'.join(summary_sentences)
        logger.info(f"Extractive summary generated with {len(summary_sentences)} sentences.")
        return summary

if __name__ == "__main__":
    # Example usage
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence
    displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize its chance of successfully
    achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines
    that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem-solving".
    
    AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube,
    Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo),
    generative AI (e.g., ChatGPT, Midjourney), and competing at the highest level in strategic game systems (such as chess and Go).
    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition
    of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from
    the definition of AI, having become a routine technology.
    """
    
    print("Initializing extractive summarizer...")
    summarizer = ExtractiveSummarizer()
    
    print("\nOriginal Text:")
    print(sample_text.strip())
    
    print("\nGenerating Summary (3 sentences)...")
    summary = summarizer.summarize(sample_text, num_sentences=3)
    
    print("\nSummary:")
    print(summary)
