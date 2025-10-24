"""
Emotion detection module for FactSnap-V
Uses Hugging Face transformers for local emotion classification
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import warnings
from config import EMOTION_MODEL, BATCH_SIZE

# Suppress warnings
warnings.filterwarnings("ignore")


class EmotionDetector:
    """
    Handles emotion detection using Hugging Face transformer models
    """
    
    def __init__(self, model_name=EMOTION_MODEL):
        """
        Initialize the emotion detection model
        
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
        Load the emotion detection model and tokenizer
        """
        try:
            print(f"Loading emotion detection model: {self.model_name}")
            
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
            
            print("Emotion detection model loaded successfully!")
            
        except Exception as e:
            raise Exception(f"Error loading emotion detection model: {str(e)}")
    
    def predict_emotion(self, text):
        """
        Predict emotion for a single text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing emotion predictions
        """
        if not text or not text.strip():
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
        
        try:
            # Get predictions
            results = self.pipeline(text.strip())
            
            # Extract the results (results is a list of lists)
            scores = results[0] if isinstance(results[0], list) else results
            
            # Find the emotion with highest confidence
            best_emotion = max(scores, key=lambda x: x['score'])
            
            # Format results
            emotion_result = {
                "emotion": best_emotion['label'].lower(),
                "confidence": best_emotion['score'],
                "all_scores": [{
                    "emotion": score['label'].lower(),
                    "confidence": score['score']
                } for score in scores]
            }
            
            return emotion_result
            
        except Exception as e:
            print(f"Error predicting emotion: {str(e)}")
            return {"emotion": "neutral", "confidence": 0.0, "all_scores": []}
    
    def predict_emotions_batch(self, texts):
        """
        Predict emotions for multiple texts in batch
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of emotion predictions
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
                    
                    # Find best emotion
                    best_emotion = max(scores, key=lambda x: x['score'])
                    
                    emotion_result = {
                        "text": batch[j],
                        "emotion": best_emotion['label'].lower(),
                        "confidence": best_emotion['score'],
                        "all_scores": [{
                            "emotion": score['label'].lower(),
                            "confidence": score['score']
                        } for score in scores]
                    }
                    
                    results.append(emotion_result)
                    
            except Exception as e:
                print(f"Error processing batch {i//BATCH_SIZE + 1}: {str(e)}")
                # Add default results for failed batch
                for text in batch:
                    results.append({
                        "text": text,
                        "emotion": "neutral",
                        "confidence": 0.0,
                        "all_scores": []
                    })
        
        return results
    
    def analyze_sentences(self, sentences):
        """
        Analyze emotions for a list of sentences
        
        Args:
            sentences (list): List of sentences to analyze
            
        Returns:
            list: List of dictionaries with sentence and emotion analysis
        """
        print(f"Analyzing emotions for {len(sentences)} sentences...")
        
        # Filter out empty sentences
        valid_sentences = [s for s in sentences if s and s.strip()]
        
        if not valid_sentences:
            return []
        
        # Get emotion predictions
        results = self.predict_emotions_batch(valid_sentences)
        
        print("Emotion analysis completed!")
        return results
    
    def get_emotion_summary(self, emotion_results):
        """
        Get summary statistics of emotions
        
        Args:
            emotion_results (list): List of emotion analysis results
            
        Returns:
            dict: Summary statistics
        """
        if not emotion_results:
            return {}
        
        # Count emotions
        emotion_counts = {}
        total_confidence = 0
        
        for result in emotion_results:
            emotion = result['emotion']
            confidence = result['confidence']
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = {'count': 0, 'total_confidence': 0}
            
            emotion_counts[emotion]['count'] += 1
            emotion_counts[emotion]['total_confidence'] += confidence
            total_confidence += confidence
        
        # Calculate percentages and average confidence
        summary = {
            'total_sentences': len(emotion_results),
            'average_confidence': total_confidence / len(emotion_results),
            'emotion_distribution': {}
        }
        
        for emotion, data in emotion_counts.items():
            summary['emotion_distribution'][emotion] = {
                'count': data['count'],
                'percentage': (data['count'] / len(emotion_results)) * 100,
                'average_confidence': data['total_confidence'] / data['count']
            }
        
        return summary


def test_emotion_detector():
    """
    Test function for the EmotionDetector class
    """
    # Sample sentences
    sample_sentences = [
        "I am so happy about this great news!",
        "This is really disappointing and frustrating.",
        "I'm not sure how to feel about this situation.",
        "The weather is nice today.",
        "I'm scared about what might happen next."
    ]
    
    try:
        # Initialize emotion detector
        detector = EmotionDetector()
        
        # Test single prediction
        emotion = detector.predict_emotion(sample_sentences[0])
        print(f"Single prediction: {emotion}")
        
        # Test batch prediction
        results = detector.analyze_sentences(sample_sentences)
        print(f"Batch results: {len(results)} sentences analyzed")
        
        for result in results:
            print(f"'{result['text']}' -> {result['emotion']} ({result['confidence']:.3f})")
        
        # Test summary
        summary = detector.get_emotion_summary(results)
        print(f"Summary: {summary}")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")


if __name__ == "__main__":
    test_emotion_detector()
