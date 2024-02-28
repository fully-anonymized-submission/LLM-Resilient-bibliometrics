import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class LanguageDetector:

    """Detect the language of a text with a LLM"""

    language_model_name = "papluca/xlm-roberta-base-language-detection"

    def __init__(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.language_model_name
        )
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def get_language(self, text: str) -> list[dict]:
        """Get the language of a text

        Parameters:
            text (int): text to analyze

        Return:
            language_score list[{'label': str, 'score': int}]: list of dict with the language and the score
        """

        # If the input is too long can return an error
        try:
            return self.classifier(text[:1300])
        except Exception:
            try:
                return self.classifier(text[:514])
            except Exception:
                return [{"label": "err"}]