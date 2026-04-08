# src/abstractive_summarizer.py

from transformers import pipeline

def summarize_abstractive(text, model_name='facebook/bart-large-cnn', max_length=150, min_length=50):
    """
    Performs abstractive text summarization using a pre-trained HuggingFace model.

    Args:
        text (str): The input text to summarize.
        model_name (str): The name of the pre-trained model to use.
        max_length (int): The maximum length of the generated summary.
        min_length (int): The minimum length of the generated summary.

    Returns:
        str: The generated abstractive summary.
    """
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

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
        "It has the potential to solve many of the world's most pressing problems, from climate change to disease. "
        "However, it also poses significant risks, such as job displacement and the potential for autonomous weapons. "
        "It is therefore crucial to develop AI responsibly and ethically."
    )
    print("Original Text:
", sample_text)
    print("
Abstractive Summary:
", summarize_abstractive(sample_text))
