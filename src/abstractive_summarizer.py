import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AbstractiveSummarizer:
    """
    A class for performing abstractive text summarization using the BART model
    from Hugging Face Transformers. Abstractive summarization generates new
    sentences that capture the essence of the original text, rather than just
    extracting existing sentences.
    """
    def __init__(self, model_name='facebook/bart-large-cnn', device=None):
        """
        Initializes the summarizer with a pre-trained BART model.
        
        Args:
            model_name (str): The name of the pre-trained model to use.
                              Defaults to 'facebook/bart-large-cnn'.
            device (str): The device to run the model on ('cpu' or 'cuda').
                          If None, it automatically selects 'cuda' if available.
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing AbstractiveSummarizer with model: {self.model_name} on device: {self.device}")
        
        try:
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise

    def summarize(self, text, max_length=130, min_length=30, num_beams=4, length_penalty=2.0, early_stopping=True):
        """
        Generates an abstractive summary for the given text.
        
        Args:
            text (str): The input text to summarize.
            max_length (int): The maximum length of the generated summary.
            min_length (int): The minimum length of the generated summary.
            num_beams (int): Number of beams for beam search.
            length_penalty (float): Exponential penalty to the length.
            early_stopping (bool): Whether to stop the beam search when at least
                                   `num_beams` sentences are finished per batch.
                                   
        Returns:
            str: The generated summary.
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text provided. Returning empty string.")
            return ""

        logger.info(f"Generating summary for text of length: {len(text)} characters.")
        
        try:
            # Tokenize the input text
            inputs = self.tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate summary ids
            summary_ids = self.model.generate(
                inputs['input_ids'],
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                early_stopping=early_stopping
            )

            # Decode the summary ids to text
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            logger.info("Summary generated successfully.")
            return summary
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return "Error generating summary."

    def batch_summarize(self, texts, **kwargs):
        """
        Generates summaries for a list of texts.
        
        Args:
            texts (list of str): A list of input texts to summarize.
            **kwargs: Additional arguments to pass to the `summarize` method.
            
        Returns:
            list of str: A list of generated summaries.
        """
        logger.info(f"Processing batch of {len(texts)} texts.")
        summaries = []
        for i, text in enumerate(texts):
            logger.debug(f"Summarizing text {i+1}/{len(texts)}")
            summary = self.summarize(text, **kwargs)
            summaries.append(summary)
        return summaries

if __name__ == "__main__":
    # Example usage
    sample_text = """
    The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. 
    As the largest telescope in space, it is equipped with high-resolution and highly sensitive instruments, 
    allowing it to view objects too old, distant, or faint for the Hubble Space Telescope. This will enable 
    investigations across many fields of astronomy and cosmology, such as observation of the first stars and 
    the formation of the first galaxies, and detailed atmospheric characterization of potentially habitable exoplanets.
    The telescope was launched on 25 December 2021 on an Ariane 5 rocket from Kourou, French Guiana, and arrived 
    at the Sun–Earth L2 Lagrange point in January 2022. The first image from JWST was released to the public via 
    a press conference on 11 July 2022.
    """
    
    print("Initializing summarizer...")
    summarizer = AbstractiveSummarizer()
    
    print("\nOriginal Text:")
    print(sample_text.strip())
    
    print("\nGenerating Summary...")
    summary = summarizer.summarize(sample_text, max_length=50, min_length=10)
    
    print("\nSummary:")
    print(summary)
