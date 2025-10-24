"""
LangGraph-based fact verification module for FactSnap-V
Uses Gemini API with LangGraph agents for comprehensive fact-checking
"""

import json
import time
from typing import List, Dict, Any, TypedDict
from datetime import datetime

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor  # Not needed for this implementation


class FactCheckState(TypedDict):
    """State for the fact-checking workflow"""
    sentence: str
    claim_extracted: bool
    claims: List[str]
    verification_results: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    error: str


class LangGraphFactVerifier:
    """
    LangGraph-based fact verifier using Gemini API
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the LangGraph fact verifier
        
        Args:
            api_key (str): Gemini API key
        """
        self.api_key = api_key
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize LangChain Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow for fact-checking
        
        Returns:
            StateGraph: The fact-checking workflow
        """
        workflow = StateGraph(FactCheckState)
        
        # Add nodes
        workflow.add_node("extract_claims", self._extract_claims_node)
        workflow.add_node("verify_claims", self._verify_claims_node)
        workflow.add_node("synthesize_result", self._synthesize_result_node)
        
        # Add edges
        workflow.add_edge("extract_claims", "verify_claims")
        workflow.add_edge("verify_claims", "synthesize_result")
        workflow.add_edge("synthesize_result", END)
        
        # Set entry point
        workflow.set_entry_point("extract_claims")
        
        return workflow.compile()
    
    def _extract_claims_node(self, state: FactCheckState) -> FactCheckState:
        """
        Extract factual claims from a sentence
        
        Args:
            state (FactCheckState): Current state
            
        Returns:
            FactCheckState: Updated state with extracted claims
        """
        try:
            sentence = state["sentence"]
            
            # System prompt for claim extraction
            system_prompt = """You are an expert fact-checker. Your task is to extract specific, verifiable factual claims from the given sentence.

Rules:
1. Only extract claims that can be fact-checked (not opinions, emotions, or subjective statements)
2. Break down complex sentences into individual claims
3. Ignore filler words, personal opinions, and emotional expressions
4. Focus on statements about facts, statistics, events, people, places, dates, etc.
5. Return claims as a JSON list of strings
6. If no factual claims exist, return an empty list

Examples:
Input: "I think the population of Tokyo is over 13 million people and it's the capital of Japan."
Output: ["The population of Tokyo is over 13 million people", "Tokyo is the capital of Japan"]

Input: "This is amazing and I love it so much!"
Output: []

Input: "The COVID-19 vaccine was developed in 2020 and has been proven effective."
Output: ["The COVID-19 vaccine was developed in 2020", "The COVID-19 vaccine has been proven effective"]"""
            
            human_prompt = f"Extract factual claims from this sentence: '{sentence}'"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                response_text = response.content.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()
                
                claims = json.loads(response_text)
                if not isinstance(claims, list):
                    claims = []
            except json.JSONDecodeError:
                # Fallback: try to extract claims manually
                claims = []
                if "no factual claims" not in response.content.lower():
                    # Simple fallback - treat the whole sentence as a claim if it seems factual
                    if any(word in sentence.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'can', 'million', 'percent', '%', 'year', 'date']):
                        claims = [sentence.strip()]
            
            state["claims"] = claims
            state["claim_extracted"] = True
            
        except Exception as e:
            state["error"] = f"Error extracting claims: {str(e)}"
            state["claims"] = []
            state["claim_extracted"] = False
        
        return state
    
    def _verify_claims_node(self, state: FactCheckState) -> FactCheckState:
        """
        Verify each extracted claim
        
        Args:
            state (FactCheckState): Current state
            
        Returns:
            FactCheckState: Updated state with verification results
        """
        try:
            claims = state["claims"]
            verification_results = []
            
            if not claims:
                state["verification_results"] = []
                return state
            
            # System prompt for fact verification
            system_prompt = """You are an expert fact-checker with access to comprehensive knowledge. Your task is to verify factual claims and provide detailed analysis.

For each claim, provide:
1. Verification status: "TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIABLE", or "NEEDS_CONTEXT"
2. Confidence level: 0.0 to 1.0
3. Explanation: Detailed reasoning for your assessment
4. Sources: Mention general knowledge areas or types of sources that support/contradict the claim
5. Context: Any important context or nuances

Return your response as JSON with this structure:
{
    "verification_status": "TRUE/FALSE/PARTIALLY_TRUE/UNVERIFIABLE/NEEDS_CONTEXT",
    "confidence": 0.0-1.0,
    "explanation": "Detailed explanation",
    "sources": "General source types or knowledge areas",
    "context": "Important context or caveats"
}

Be thorough but concise. If you're uncertain, use "UNVERIFIABLE" and explain why."""
            
            for claim in claims:
                try:
                    human_prompt = f"Fact-check this claim: '{claim}'"
                    
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ]
                    
                    response = self.llm.invoke(messages)
                    
                    # Parse the response
                    try:
                        response_text = response.content.strip()
                        if response_text.startswith('```json'):
                            response_text = response_text[7:-3].strip()
                        elif response_text.startswith('```'):
                            response_text = response_text[3:-3].strip()
                        
                        result = json.loads(response_text)
                        
                        # Ensure all required fields exist
                        verification_result = {
                            "claim": claim,
                            "verification_status": result.get("verification_status", "UNVERIFIABLE"),
                            "confidence": float(result.get("confidence", 0.0)),
                            "explanation": result.get("explanation", "No explanation provided"),
                            "sources": result.get("sources", "General knowledge"),
                            "context": result.get("context", ""),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                    except json.JSONDecodeError:
                        # Fallback parsing
                        verification_result = {
                            "claim": claim,
                            "verification_status": "UNVERIFIABLE",
                            "confidence": 0.0,
                            "explanation": f"Could not parse verification result: {response.content[:200]}",
                            "sources": "Unknown",
                            "context": "",
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    verification_results.append(verification_result)
                    
                    # Add delay to respect rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    verification_results.append({
                        "claim": claim,
                        "verification_status": "UNVERIFIABLE",
                        "confidence": 0.0,
                        "explanation": f"Error during verification: {str(e)}",
                        "sources": "Error",
                        "context": "",
                        "timestamp": datetime.now().isoformat()
                    })
            
            state["verification_results"] = verification_results
            
        except Exception as e:
            state["error"] = f"Error verifying claims: {str(e)}"
            state["verification_results"] = []
        
        return state
    
    def _synthesize_result_node(self, state: FactCheckState) -> FactCheckState:
        """
        Synthesize final result from all verifications
        
        Args:
            state (FactCheckState): Current state
            
        Returns:
            FactCheckState: Updated state with final result
        """
        try:
            sentence = state["sentence"]
            claims = state["claims"]
            verification_results = state["verification_results"]
            
            if not verification_results:
                # No claims to verify
                final_result = {
                    "sentence": sentence,
                    "overall_status": "NO_CLAIMS",
                    "confidence": 1.0,
                    "summary": "No factual claims found in this sentence",
                    "claims_count": 0,
                    "verified_claims": [],
                    "analysis_timestamp": datetime.now().isoformat()
                }
            else:
                # Calculate overall status
                statuses = [result["verification_status"] for result in verification_results]
                confidences = [result["confidence"] for result in verification_results]
                
                # Determine overall status
                if all(status == "TRUE" for status in statuses):
                    overall_status = "TRUE"
                elif all(status == "FALSE" for status in statuses):
                    overall_status = "FALSE"
                elif any(status == "FALSE" for status in statuses):
                    overall_status = "MIXED_FALSE"
                elif any(status == "PARTIALLY_TRUE" for status in statuses):
                    overall_status = "MIXED_PARTIAL"
                elif all(status == "UNVERIFIABLE" for status in statuses):
                    overall_status = "UNVERIFIABLE"
                else:
                    overall_status = "MIXED"
                
                # Calculate average confidence
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Create summary
                true_count = sum(1 for s in statuses if s == "TRUE")
                false_count = sum(1 for s in statuses if s == "FALSE")
                partial_count = sum(1 for s in statuses if s == "PARTIALLY_TRUE")
                unverifiable_count = sum(1 for s in statuses if s == "UNVERIFIABLE")
                
                summary_parts = []
                if true_count > 0:
                    summary_parts.append(f"{true_count} true")
                if false_count > 0:
                    summary_parts.append(f"{false_count} false")
                if partial_count > 0:
                    summary_parts.append(f"{partial_count} partially true")
                if unverifiable_count > 0:
                    summary_parts.append(f"{unverifiable_count} unverifiable")
                
                summary = f"Found {len(claims)} claims: " + ", ".join(summary_parts)
                
                final_result = {
                    "sentence": sentence,
                    "overall_status": overall_status,
                    "confidence": avg_confidence,
                    "summary": summary,
                    "claims_count": len(claims),
                    "verified_claims": verification_results,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            
            state["final_result"] = final_result
            
        except Exception as e:
            state["error"] = f"Error synthesizing result: {str(e)}"
            state["final_result"] = {
                "sentence": sentence,
                "overall_status": "ERROR",
                "confidence": 0.0,
                "summary": f"Error during analysis: {str(e)}",
                "claims_count": 0,
                "verified_claims": [],
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        return state
    
    def verify_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Verify a single sentence using the LangGraph workflow
        
        Args:
            sentence (str): Sentence to fact-check
            
        Returns:
            Dict[str, Any]: Verification result
        """
        if not sentence or not sentence.strip():
            return {
                "sentence": sentence,
                "overall_status": "NO_CONTENT",
                "confidence": 1.0,
                "summary": "Empty or invalid sentence",
                "claims_count": 0,
                "verified_claims": [],
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        # Initialize state
        initial_state = {
            "sentence": sentence.strip(),
            "claim_extracted": False,
            "claims": [],
            "verification_results": [],
            "final_result": {},
            "error": ""
        }
        
        try:
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            if result.get("error"):
                return {
                    "sentence": sentence,
                    "overall_status": "ERROR",
                    "confidence": 0.0,
                    "summary": f"Error: {result['error']}",
                    "claims_count": 0,
                    "verified_claims": [],
                    "analysis_timestamp": datetime.now().isoformat()
                }
            
            return result["final_result"]
            
        except Exception as e:
            return {
                "sentence": sentence,
                "overall_status": "ERROR",
                "confidence": 0.0,
                "summary": f"Workflow error: {str(e)}",
                "claims_count": 0,
                "verified_claims": [],
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def verify_sentences_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Verify multiple sentences
        
        Args:
            sentences (List[str]): List of sentences to fact-check
            
        Returns:
            List[Dict[str, Any]]: List of verification results
        """
        print(f"Verifying {len(sentences)} sentences with LangGraph...")
        
        results = []
        
        for i, sentence in enumerate(sentences):
            print(f"Verifying sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            result = self.verify_sentence(sentence)
            results.append(result)
            
            # Add delay between sentences to respect rate limits
            if i < len(sentences) - 1:
                time.sleep(1.0)  # Longer delay for Gemini API
        
        print("LangGraph fact verification completed!")
        return results
    
    def get_verification_summary(self, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of verification results
        
        Args:
            verification_results (List[Dict[str, Any]]): List of verification results
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not verification_results:
            return {}
        
        # Count overall statuses
        status_counts = {}
        total_confidence = 0
        total_claims = 0
        sentences_with_claims = 0
        
        for result in verification_results:
            status = result['overall_status']
            confidence = result['confidence']
            claims_count = result['claims_count']
            
            if status not in status_counts:
                status_counts[status] = 0
            
            status_counts[status] += 1
            total_confidence += confidence
            total_claims += claims_count
            
            if claims_count > 0:
                sentences_with_claims += 1
        
        # Calculate detailed claim statistics
        claim_status_counts = {}
        for result in verification_results:
            for claim_result in result.get('verified_claims', []):
                claim_status = claim_result['verification_status']
                if claim_status not in claim_status_counts:
                    claim_status_counts[claim_status] = 0
                claim_status_counts[claim_status] += 1
        
        # Calculate percentages
        summary = {
            'total_sentences': len(verification_results),
            'sentences_with_claims': sentences_with_claims,
            'total_claims': total_claims,
            'average_confidence': total_confidence / len(verification_results),
            'claims_per_sentence': total_claims / len(verification_results),
            'sentence_status_distribution': {},
            'claim_status_distribution': {}
        }
        
        # Sentence-level distribution
        for status, count in status_counts.items():
            summary['sentence_status_distribution'][status] = {
                'count': count,
                'percentage': (count / len(verification_results)) * 100
            }
        
        # Claim-level distribution
        if total_claims > 0:
            for status, count in claim_status_counts.items():
                summary['claim_status_distribution'][status] = {
                    'count': count,
                    'percentage': (count / total_claims) * 100
                }
        
        return summary


def test_langgraph_fact_verifier():
    """
    Test function for the LangGraphFactVerifier class
    """
    # Test with the provided API key
    api_key = "AIzaSyDkqJWfhHbtifqj3PjKZdXVL3JLowSj2cM"
    
    # Sample sentences
    sample_sentences = [
        "The COVID-19 vaccine is safe and effective.",
        "Climate change is causing global temperatures to rise.",
        "The Earth is flat and the moon landing was fake.",
        "Water boils at 100 degrees Celsius at sea level.",
        "I think this movie is really amazing and entertaining!"
    ]
    
    try:
        # Initialize fact verifier
        verifier = LangGraphFactVerifier(api_key)
        
        # Test single verification
        print("Testing single sentence verification...")
        result = verifier.verify_sentence(sample_sentences[0])
        print(f"Single verification result: {json.dumps(result, indent=2)}")
        
        # Test batch verification (only first 2 sentences to save API quota)
        print("\nTesting batch verification...")
        results = verifier.verify_sentences_batch(sample_sentences[:2])
        
        for result in results:
            print(f"'{result['sentence'][:50]}...' -> {result['overall_status']} (confidence: {result['confidence']:.2f})")
        
        # Test summary
        print("\nTesting summary generation...")
        summary = verifier.get_verification_summary(results)
        print(f"Summary: {json.dumps(summary, indent=2)}")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")


if __name__ == "__main__":
    test_langgraph_fact_verifier()