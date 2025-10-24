"""
Main application module for FactSnap-V
Orchestrates the complete analysis pipeline
"""

import os
import time
import pandas as pd
from datetime import datetime

# Import all modules
from input_processor import InputProcessor
from speech_to_text import SpeechToText
from text_preprocessor import TextPreprocessor
from emotion_detector import EmotionDetector
from bias_detector import BiasDetector
from langgraph_fact_verifier import LangGraphFactVerifier


class FactSnapV:
    """
    Main FactSnap-V application class
    Orchestrates the complete analysis pipeline
    """
    
    def __init__(self):
        """
        Initialize all components
        """
        self.input_processor = None
        self.speech_to_text = None
        self.text_preprocessor = None
        self.emotion_detector = None
        self.bias_detector = None
        self.fact_verifier = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """
        Initialize all analysis components
        """
        print("Initializing FactSnap-V components...")
        
        try:
            # Initialize input processor
            self.input_processor = InputProcessor()
            print("✓ Input processor initialized")
            
            # Initialize speech-to-text
            self.speech_to_text = SpeechToText()
            print("✓ Speech-to-text initialized")
            
            # Initialize text preprocessor
            self.text_preprocessor = TextPreprocessor()
            print("✓ Text preprocessor initialized")
            
            # Initialize emotion detector
            self.emotion_detector = EmotionDetector()
            print("✓ Emotion detector initialized")
            
            # Initialize bias detector
            self.bias_detector = BiasDetector()
            print("✓ Bias detector initialized")
            
            # Initialize LangGraph fact verifier
            from config import GEMINI_API_KEY
            self.fact_verifier = LangGraphFactVerifier(GEMINI_API_KEY)
            print("✓ LangGraph fact verifier initialized")
            
            print("All components initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            raise
    
    def analyze_file(self, file_path, extract_claims=True):
        """
        Analyze a single audio or video file
        
        Args:
            file_path (str): Path to the input file
            extract_claims (bool): Whether to extract and fact-check claims
            
        Returns:
            dict: Complete analysis results
        """
        print(f"Starting analysis of: {file_path}")
        start_time = time.time()
        
        try:
            # Step 1: Process input file
            print("\n--- Step 1: Processing input file ---")
            audio_path = self.input_processor.process_input_file(file_path)
            print(f"Audio extracted to: {audio_path}")
            
            # Step 2: Speech-to-text
            print("\n--- Step 2: Speech-to-text conversion ---")
            transcript = self.speech_to_text.get_clean_transcript(audio_path)
            print(f"Transcript length: {len(transcript)} characters")
            print(f"Transcript preview: {transcript[:200]}...")
            
            # Step 3: Text preprocessing
            print("\n--- Step 3: Text preprocessing ---")
            sentences = self.text_preprocessor.preprocess_transcript(transcript)
            print(f"Extracted {len(sentences)} sentences")
            
            # Step 4: Emotion analysis
            print("\n--- Step 4: Emotion analysis ---")
            emotion_results = self.emotion_detector.analyze_sentences(sentences)
            emotion_summary = self.emotion_detector.get_emotion_summary(emotion_results)
            
            # Step 5: Bias analysis
            print("\n--- Step 5: Bias analysis ---")
            bias_results = self.bias_detector.analyze_sentences(sentences)
            bias_summary = self.bias_detector.get_bias_summary(bias_results)
            
            # Step 6: Fact verification (optional)
            fact_results = []
            fact_summary = {}
            
            if extract_claims:
                print("\n--- Step 6: LangGraph Fact verification ---")
                # Use LangGraph to verify each sentence directly
                fact_results = self.fact_verifier.verify_sentences_batch(sentences)
                fact_summary = self.fact_verifier.get_verification_summary(fact_results)
            
            # Combine results
            analysis_time = time.time() - start_time
            
            results = {
                'file_info': {
                    'file_path': file_path,
                    'audio_path': audio_path,
                    'analysis_time': analysis_time,
                    'timestamp': datetime.now().isoformat()
                },
                'transcript': {
                    'text': transcript,
                    'sentences': sentences,
                    'sentence_count': len(sentences),
                    'character_count': len(transcript),
                    'word_count': len(transcript.split())
                },
                'emotion_analysis': {
                    'results': emotion_results,
                    'summary': emotion_summary
                },
                'bias_analysis': {
                    'results': bias_results,
                    'summary': bias_summary
                },
                'fact_verification': {
                    'results': fact_results,
                    'summary': fact_summary,
                    'claims_extracted': len(fact_results) if fact_results else 0
                }
            }
            
            print(f"\n✓ Analysis completed in {analysis_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise
        
        finally:
            # Cleanup temporary files
            if self.input_processor:
                self.input_processor.cleanup()
    
    def create_detailed_dataframe(self, results):
        """
        Create a detailed DataFrame with all analysis results
        
        Args:
            results (dict): Analysis results from analyze_file()
            
        Returns:
            pd.DataFrame: Detailed results DataFrame
        """
        sentences = results['transcript']['sentences']
        emotion_results = results['emotion_analysis']['results']
        bias_results = results['bias_analysis']['results']
        
        # Create base DataFrame
        data = []
        
        for i, sentence in enumerate(sentences):
            row = {
                'sentence_id': i + 1,
                'text': sentence,
                'character_count': len(sentence),
                'word_count': len(sentence.split())
            }
            
            # Add emotion data
            if i < len(emotion_results):
                emotion = emotion_results[i]
                row.update({
                    'emotion': emotion['emotion'],
                    'emotion_confidence': emotion['confidence']
                })
            else:
                row.update({
                    'emotion': 'unknown',
                    'emotion_confidence': 0.0
                })
            
            # Add bias data
            if i < len(bias_results):
                bias = bias_results[i]
                row.update({
                    'bias_level': bias['bias_level'],
                    'bias_score': bias['bias_score'],
                    'bias_confidence': bias['confidence']
                })
            else:
                row.update({
                    'bias_level': 'Low',
                    'bias_score': 0.0,
                    'bias_confidence': 0.0
                })
            
            # Add fact verification data (if available)
            fact_result = None
            if results['fact_verification']['results'] and i < len(results['fact_verification']['results']):
                fact_result = results['fact_verification']['results'][i]
            
            if fact_result:
                row.update({
                    'fact_status': fact_result['overall_status'],
                    'fact_confidence': fact_result['confidence'],
                    'fact_claims_count': fact_result['claims_count'],
                    'fact_summary': fact_result['summary']
                })
            else:
                row.update({
                    'fact_status': 'Not Checked',
                    'fact_confidence': 0.0,
                    'fact_claims_count': 0,
                    'fact_summary': 'N/A'
                })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_results(self, results, output_dir="output", filename_prefix="factsnap"):
        """
        Export results in multiple formats
        
        Args:
            results (dict): Analysis results
            output_dir (str): Output directory
            filename_prefix (str): Prefix for output files
            
        Returns:
            dict: Paths to exported files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        exported_files = {}
        
        try:
            # Export detailed DataFrame as CSV
            df = self.create_detailed_dataframe(results)
            csv_path = os.path.join(output_dir, f"{base_filename}_detailed.csv")
            df.to_csv(csv_path, index=False)
            exported_files['csv'] = csv_path
            
            # Export summary as JSON
            import json
            summary_data = {
                'file_info': results['file_info'],
                'summary': {
                    'transcript_summary': {
                        'sentence_count': results['transcript']['sentence_count'],
                        'character_count': results['transcript']['character_count'],
                        'word_count': results['transcript']['word_count']
                    },
                    'emotion_summary': results['emotion_analysis']['summary'],
                    'bias_summary': results['bias_analysis']['summary'],
                    'fact_summary': results['fact_verification']['summary']
                }
            }
            
            json_path = os.path.join(output_dir, f"{base_filename}_summary.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            exported_files['json'] = json_path
            
            # Export transcript as text file
            txt_path = os.path.join(output_dir, f"{base_filename}_transcript.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"FactSnap-V Analysis Results\n")
                f.write(f"Generated: {results['file_info']['timestamp']}\n")
                f.write(f"Source: {results['file_info']['file_path']}\n")
                f.write(f"Analysis Time: {results['file_info']['analysis_time']:.2f} seconds\n\n")
                f.write("TRANSCRIPT:\n")
                f.write("=" * 50 + "\n")
                f.write(results['transcript']['text'])
            exported_files['txt'] = txt_path
            
            print(f"\nResults exported to:")
            for file_type, file_path in exported_files.items():
                print(f"  {file_type.upper()}: {file_path}")
            
            return exported_files
            
        except Exception as e:
            print(f"Error exporting results: {str(e)}")
            return {}


def main():
    """
    Main function for command-line usage
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_or_video_file>")
        print("Example: python main.py sample.mp3")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    try:
        # Initialize FactSnap-V
        app = FactSnapV()
        
        # Analyze file
        results = app.analyze_file(file_path)
        
        # Export results
        exported_files = app.export_results(results)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        # Emotion summary
        emotion_summary = results['emotion_analysis']['summary']
        if emotion_summary:
            print(f"\nEMOTION ANALYSIS:")
            print(f"  Total sentences: {emotion_summary['total_sentences']}")
            print(f"  Average confidence: {emotion_summary['average_confidence']:.3f}")
            for emotion, data in emotion_summary['emotion_distribution'].items():
                print(f"  {emotion.capitalize()}: {data['count']} ({data['percentage']:.1f}%)")
        
        # Bias summary
        bias_summary = results['bias_analysis']['summary']
        if bias_summary:
            print(f"\nBIAS ANALYSIS:")
            print(f"  Total sentences: {bias_summary['total_sentences']}")
            print(f"  Average bias score: {bias_summary['average_bias_score']:.3f}")
            for level, data in bias_summary['bias_distribution'].items():
                print(f"  {level} bias: {data['count']} ({data['percentage']:.1f}%)")
        
        # Fact verification summary
        fact_summary = results['fact_verification']['summary']
        if fact_summary:
            print(f"\nFACT VERIFICATION:")
            print(f"  Total claims: {fact_summary['total_claims']}")
            print(f"  Source coverage: {fact_summary['source_coverage']:.1f}%")
            for status, data in fact_summary['verification_distribution'].items():
                print(f"  {status}: {data['count']} ({data['percentage']:.1f}%)")
        
        print(f"\nAnalysis completed! Check the output files for detailed results.")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
