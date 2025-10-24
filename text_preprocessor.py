"""
Text preprocessing module for FactSnap-V
Handles sentence segmentation and text cleaning
"""

import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from config import MAX_SENTENCE_LENGTH

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass


class TextPreprocessor:
    """
    Handles text preprocessing including sentence segmentation
    """
    
    def __init__(self, use_spacy=True):
        """
        Initialize the text preprocessor
        
        Args:
            use_spacy (bool): Whether to use spaCy for sentence segmentation
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            self._load_spacy_model()
    
    def _load_spacy_model(self):
        """
        Load spaCy model for sentence segmentation
        """
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Using NLTK for sentence segmentation.")
            self.use_spacy = False
    
    def clean_text(self, text):
        """
        Clean text by removing extra whitespace and fixing common issues
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common transcription issues
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after sentence end
        
        # Capitalize first letter of sentences
        text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text.strip()
    
    def segment_sentences_spacy(self, text):
        """
        Segment text into sentences using spaCy
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        if not self.nlp:
            raise Exception("spaCy model not loaded")
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s]
        
        return sentences
    
    def segment_sentences_nltk(self, text):
        """
        Segment text into sentences using NLTK
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        sentences = sent_tokenize(text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def segment_sentences(self, text):
        """
        Segment text into sentences using the preferred method
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        if self.use_spacy and self.nlp:
            return self.segment_sentences_spacy(text)
        else:
            return self.segment_sentences_nltk(text)
    
    def filter_sentences(self, sentences):
        """
        Filter sentences based on length and content
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            list: Filtered list of sentences
        """
        filtered = []
        
        for sentence in sentences:
            # Skip very short sentences (likely noise)
            if len(sentence.split()) < 3:
                continue
                
            # Skip very long sentences
            if len(sentence) > MAX_SENTENCE_LENGTH:
                # Try to split long sentences at commas or semicolons
                sub_sentences = re.split(r'[,;]', sentence)
                for sub_sent in sub_sentences:
                    sub_sent = sub_sent.strip()
                    if len(sub_sent.split()) >= 3 and len(sub_sent) <= MAX_SENTENCE_LENGTH:
                        filtered.append(sub_sent)
                continue
            
            filtered.append(sentence)
        
        return filtered
    
    def preprocess_transcript(self, transcript):
        """
        Complete preprocessing pipeline for transcript
        
        Args:
            transcript (str): Raw transcript text
            
        Returns:
            list: List of processed sentences
        """
        # Clean the text
        cleaned_text = self.clean_text(transcript)
        
        # Segment into sentences
        sentences = self.segment_sentences(cleaned_text)
        
        # Filter sentences
        filtered_sentences = self.filter_sentences(sentences)
        
        return filtered_sentences
    
    def extract_claims(self, sentences):
        """
        Extract factual claims from sentences with improved filtering
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            list: List of sentences that contain potential factual claims
        """
        claims = []
        
        # Strong indicators of verifiable claims
        strong_indicators = [
            'according to', 'research shows', 'studies indicate', 'data suggests',
            'statistics show', 'evidence indicates', 'report states', 'survey found',
            'study found', 'experts say', 'scientists claim', 'research reveals',
            'data shows', 'analysis shows', 'findings suggest'
        ]
        
        # Factual statement patterns
        factual_patterns = [
            r'\b\d+(\.\d+)?\s*(percent|%|million|billion|trillion|thousand)\b',
            r'\bin\s+(19|20)\d{2}\b',  # Years
            r'\b(covid|coronavirus|vaccine|climate|temperature|population)\b',
            r'\b(celsius|fahrenheit|degrees)\b',
            r'\b(miles|kilometers|feet|meters)\b',
            r'\b(dollars|euros|pounds|yen)\b'
        ]
        
        # Topic keywords that are often fact-checked
        verifiable_topics = [
            'covid', 'coronavirus', 'vaccine', 'climate', 'election', 'poll',
            'economy', 'unemployment', 'inflation', 'gdp', 'temperature',
            'population', 'crime', 'immigration', 'health', 'medical',
            'scientific', 'study', 'research'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            # Score based on strong indicators
            if any(indicator in sentence_lower for indicator in strong_indicators):
                score += 3
            
            # Score based on factual patterns
            pattern_matches = sum(1 for pattern in factual_patterns if re.search(pattern, sentence_lower))
            score += pattern_matches * 2
            
            # Score based on verifiable topics
            topic_matches = sum(1 for topic in verifiable_topics if topic in sentence_lower)
            score += topic_matches
            
            # Additional scoring for definitive statements
            if re.search(r'\b(is|are|was|were)\s+\w+\s+(than|at|in|of|by)\b', sentence_lower):
                score += 1
            
            # Filter out questions and opinions
            if ('?' in sentence or 
                any(word in sentence_lower for word in ['i think', 'i believe', 'maybe', 'perhaps', 'might'])):
                score -= 2
            
            # Minimum length requirement
            if len(sentence.split()) < 5:
                score -= 1
            
            # If score is high enough, consider it a claim
            if score >= 2:
                claims.append(sentence)
        
        # Sort by likelihood of being verifiable (longer, more specific claims first)
        claims.sort(key=lambda x: (-len(x), -x.lower().count('according to')))
        
        # Limit to top 10 most promising claims to avoid API quota issues
        return claims[:10]


def test_text_preprocessor():
    """
    Test function for the TextPreprocessor class
    """
    # Sample transcript text
    sample_text = """
    Hello everyone.    This is a test transcript.   According to recent studies, 
    artificial intelligence is growing rapidly. The market size was 120 billion dollars in 2022.
    This is exciting news for technology companies.    What do you think about this?
    """
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test cleaning
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Cleaned text: {cleaned}")
    
    # Test sentence segmentation
    sentences = preprocessor.segment_sentences(cleaned)
    print(f"Sentences: {sentences}")
    
    # Test complete preprocessing
    processed = preprocessor.preprocess_transcript(sample_text)
    print(f"Processed sentences: {processed}")
    
    # Test claim extraction
    claims = preprocessor.extract_claims(processed)
    print(f"Claims: {claims}")


if __name__ == "__main__":
    test_text_preprocessor()
