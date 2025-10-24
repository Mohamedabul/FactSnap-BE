"""
Bias detection module for FactSnap-V
Uses Hugging Face transformers for local bias classification
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import warnings
from config import BIAS_MODEL, BATCH_SIZE, BIAS_LEVELS

# Suppress warnings
warnings.filterwarnings("ignore")


class BiasDetector:
    """
    Handles bias detection using Hugging Face transformer models
    """
    
    def __init__(self, model_name=BIAS_MODEL):
        """
        Initialize the bias detection model
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the bias detection model and tokenizer
        """
        try:
            print(f"Loading bias detection model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            print("Bias detection model loaded successfully!")
            
        except Exception as e:
            raise Exception(f"Error loading bias detection model: {str(e)}")
    
    def _map_bias_score_to_level(self, score):
        """
        Map bias score to bias level
        
        Args:
            score (float): Bias score from model
            
        Returns:
            str: Bias level (Low, Medium, High)
        """
        if score < 0.3:
            return "Low"
        elif score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def predict_bias(self, text):
        """
        Predict bias for a single text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing bias predictions
        """
        if not text or not text.strip():
            return {"bias_level": "Low", "bias_score": 0.0, "confidence": 0.0, "all_scores": []}
        
        try:
            # Get predictions
            results = self.pipeline(text.strip())
            
            # Extract the results (results is a list of lists)
            scores = results[0] if isinstance(results[0], list) else results
            
            # Find toxic/bias score
            bias_score = 0.0
            for score_item in scores:
                if score_item['label'].upper() in ['TOXIC', 'BIAS', 'BIASED', '1']:
                    bias_score = score_item['score']
                    break
            
            # If no explicit bias label, use the highest score that's not neutral
            if bias_score == 0.0:
                non_neutral_scores = [s for s in scores if s['label'].upper() not in ['NEUTRAL', 'NON-TOXIC', 'CLEAN', '0']]
                if non_neutral_scores:
                    bias_score = max(non_neutral_scores, key=lambda x: x['score'])['score']
            
            # Map to bias level
            bias_level = self._map_bias_score_to_level(bias_score)
            
            # Format results
            bias_result = {
                "bias_level": bias_level,
                "bias_score": bias_score,
                "confidence": max(scores, key=lambda x: x['score'])['score'],
                "all_scores": [{
                    "label": score['label'],
                    "score": score['score']
                } for score in scores]
            }
            
            return bias_result
            
        except Exception as e:
            print(f"Error predicting bias: {str(e)}")
            return {"bias_level": "Low", "bias_score": 0.0, "confidence": 0.0, "all_scores": []}
    
    def predict_bias_batch(self, texts):
        """
        Predict bias for multiple texts in batch
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of bias predictions
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            try:
                # Get predictions for batch
                batch_results = self.pipeline(batch)
                
                # Process results
                for j, text_results in enumerate(batch_results):
                    # Handle different result formats
                    scores = text_results if isinstance(text_results, list) else text_results
                    
                    # Find bias score
                    bias_score = 0.0
                    for score_item in scores:
                        if score_item['label'].upper() in ['TOXIC', 'BIAS', 'BIASED', '1']:
                            bias_score = score_item['score']
                            break
                    
                    # If no explicit bias label, use the highest score that's not neutral
                    if bias_score == 0.0:
                        non_neutral_scores = [s for s in scores if s['label'].upper() not in ['NEUTRAL', 'NON-TOXIC', 'CLEAN', '0']]
                        if non_neutral_scores:
                            bias_score = max(non_neutral_scores, key=lambda x: x['score'])['score']
                    
                    # Map to bias level
                    bias_level = self._map_bias_score_to_level(bias_score)
                    
                    bias_result = {
                        "text": batch[j],
                        "bias_level": bias_level,
                        "bias_score": bias_score,
                        "confidence": max(scores, key=lambda x: x['score'])['score'],
                        "all_scores": [{
                            "label": score['label'],
                            "score": score['score']
                        } for score in scores]
                    }
                    
                    results.append(bias_result)
                    
            except Exception as e:
                print(f"Error processing batch {i//BATCH_SIZE + 1}: {str(e)}")
                # Add default results for failed batch
                for text in batch:
                    results.append({
                        "text": text,
                        "bias_level": "Low",
                        "bias_score": 0.0,
                        "confidence": 0.0,
                        "all_scores": []
                    })
        
        return results
    
    def analyze_sentences(self, sentences):
        """
        Analyze bias for a list of sentences
        
        Args:
            sentences (list): List of sentences to analyze
            
        Returns:
            list: List of dictionaries with sentence and bias analysis
        """
        print(f"Analyzing bias for {len(sentences)} sentences...")
        
        # Filter out empty sentences
        valid_sentences = [s for s in sentences if s and s.strip()]
        
        if not valid_sentences:
            return []
        
        # Get bias predictions
        results = self.predict_bias_batch(valid_sentences)
        
        print("Bias analysis completed!")
        return results
    
    def get_bias_summary(self, bias_results):
        """
        Get summary statistics of bias levels
        
        Args:
            bias_results (list): List of bias analysis results
            
        Returns:
            dict: Summary statistics
        """
        if not bias_results:
            return {}
        
        # Count bias levels
        bias_counts = {}
        total_bias_score = 0
        
        for result in bias_results:
            bias_level = result['bias_level']
            bias_score = result['bias_score']
            
            if bias_level not in bias_counts:
                bias_counts[bias_level] = {'count': 0, 'total_score': 0}
            
            bias_counts[bias_level]['count'] += 1
            bias_counts[bias_level]['total_score'] += bias_score
            total_bias_score += bias_score
        
        # Calculate percentages and average scores
        summary = {
            'total_sentences': len(bias_results),
            'average_bias_score': total_bias_score / len(bias_results),
            'bias_distribution': {}
        }
        
        for bias_level, data in bias_counts.items():
            summary['bias_distribution'][bias_level] = {
                'count': data['count'],
                'percentage': (data['count'] / len(bias_results)) * 100,
                'average_score': data['total_score'] / data['count']
            }
        
        return summary
    
    def get_high_bias_sentences(self, bias_results, threshold=0.7):
        """
        Get sentences with high bias
        
        Args:
            bias_results (list): List of bias analysis results
            threshold (float): Minimum bias score threshold
            
        Returns:
            list: List of high bias sentences
        """
        high_bias = []
        
        for result in bias_results:
            if result['bias_score'] >= threshold or result['bias_level'] == 'High':
                high_bias.append(result)
        
        return high_bias


def test_bias_detector():
    """
    Test function for the BiasDetector class
    """
    # Sample sentences
    sample_sentences = [
        "This is a neutral statement about technology.",
        "I think that's a reasonable approach to the problem.",
        "The research shows promising results.",
        "This policy is absolutely terrible and wrong.",
        "People from that group are always causing problems."
    ]
    
    try:
        # Initialize bias detector
        detector = BiasDetector()
        
        # Test single prediction
        bias = detector.predict_bias(sample_sentences[0])
        print(f"Single prediction: {bias}")
        
        # Test batch prediction
        results = detector.analyze_sentences(sample_sentences)
        print(f"Batch results: {len(results)} sentences analyzed")
        
        for result in results:
            print(f"'{result['text']}' -> {result['bias_level']} (score: {result['bias_score']:.3f})")
        
        # Test summary
        summary = detector.get_bias_summary(results)
        print(f"Summary: {summary}")
        
        # Test high bias sentences
        high_bias = detector.get_high_bias_sentences(results)
        print(f"High bias sentences: {len(high_bias)}")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")


if __name__ == "__main__":
    test_bias_detector()
