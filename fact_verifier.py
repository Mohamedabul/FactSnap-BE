"""
Fact verification module for FactSnap-V
Uses Google Fact Check Tools API for fact verification
"""

import requests
import time
import json
from urllib.parse import quote
from config import GOOGLE_FACT_CHECK_API_KEY, GOOGLE_FACT_CHECK_API_URL


class FactVerifier:
    """
    Handles fact verification using Google Fact Check Tools API with fallback methods
    """
    
    def __init__(self, api_key=GOOGLE_FACT_CHECK_API_KEY):
        """
        Initialize the fact verifier
        
        Args:
            api_key (str): Google Fact Check Tools API key
        """
        self.api_key = api_key
        self.base_url = GOOGLE_FACT_CHECK_API_URL
        self.rate_limit_delay = 0.1  # Delay between API calls to respect rate limits
    
    def _make_api_request(self, query, language='en', max_age_days=None):
        """
        Make a request to the Google Fact Check Tools API
        
        Args:
            query (str): Search query
            language (str): Language code
            max_age_days (int): Maximum age of fact checks in days
            
        Returns:
            dict: API response
        """
        try:
            print(f"Making API request with query: {query}")
            # Prepare parameters
            params = {
                'key': self.api_key,
                'query': query,
                'languageCode': language
            }
            
            if max_age_days:
                params['maxAgeDays'] = max_age_days
            
            # Make request
            response = requests.get(self.base_url, params=params, timeout=10)
            
            # Check for API errors
            if response.status_code == 400:
                print(f"Bad request for query: {query}")
                return None
            elif response.status_code == 403:
                print("API key error or quota exceeded")
                return None
            elif response.status_code != 200:
                print(f"API error {response.status_code} for query: {query}")
                return None
            
            # Parse response
            data = response.json()
            return data
            
        except requests.exceptions.Timeout:
            print(f"Timeout error for query: {query}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error for query '{query}': {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error for query '{query}': {str(e)}")
            return None
    
    def _extract_claim_info(self, claim_data):
        """
        Extract relevant information from a claim
        
        Args:
            claim_data (dict): Claim data from API response
            
        Returns:
            dict: Extracted claim information
        """
        try:
            claim_info = {
                'text': claim_data.get('text', ''),
                'claimant': claim_data.get('claimant', 'Unknown'),
                'claim_date': claim_data.get('claimDate', 'Unknown'),
                'rating': 'Unknown',
                'rating_explanation': '',
                'source_name': 'Unknown',
                'source_url': ''
            }
            
            # Extract rating information
            claim_reviews = claim_data.get('claimReview', [])
            if claim_reviews:
                review = claim_reviews[0]  # Use first review
                
                claim_info['rating'] = review.get('textualRating', 'Unknown')
                claim_info['rating_explanation'] = review.get('languageCode', '')
                
                # Extract publisher info
                publisher = review.get('publisher', {})
                claim_info['source_name'] = publisher.get('name', 'Unknown')
                claim_info['source_url'] = publisher.get('site', review.get('url', ''))
            
            return claim_info
            
        except Exception as e:
            print(f"Error extracting claim info: {str(e)}")
            return {
                'text': '',
                'claimant': 'Unknown',
                'claim_date': 'Unknown',
                'rating': 'Unknown',
                'rating_explanation': '',
                'source_name': 'Unknown',
                'source_url': ''
            }
    
    def _classify_rating(self, rating_text):
        """
        Classify textual rating into standard categories
        
        Args:
            rating_text (str): Textual rating from fact checker
            
        Returns:
            str: Standardized rating (True, False, Mixed, Not Verifiable)
        """
        if not rating_text:
            return "Not Verifiable"
        
        rating_lower = rating_text.lower()
        
        # True ratings
        true_indicators = ['true', 'correct', 'accurate', 'verified', 'fact', 'confirmed']
        if any(indicator in rating_lower for indicator in true_indicators):
            return "True"
        
        # False ratings
        false_indicators = ['false', 'incorrect', 'wrong', 'debunked', 'fake', 'misleading', 'inaccurate']
        if any(indicator in rating_lower for indicator in false_indicators):
            return "False"
        
        # Mixed/Partially true ratings
        mixed_indicators = ['mixed', 'partial', 'mostly', 'some', 'half', 'partly']
        if any(indicator in rating_lower for indicator in mixed_indicators):
            return "Mixed"
        
        # If no clear classification, return as not verifiable
        return "Not Verifiable"
    
    def verify_claim(self, claim_text):
        print(f"Verifying claim: {claim_text}")  # Debugging output
        """
        Verify a single claim
        
        Args:
            claim_text (str): Text of the claim to verify
            
        Returns:
            dict: Verification result
        """
        if not claim_text or not claim_text.strip():
            return {
                'query': claim_text,
                'verification_status': 'Not Verifiable',
                'rating': 'Not Verifiable',
                'confidence': 0.0,
                'source': 'None',
                'source_url': '',
                'claims_found': 0,
                'details': []
            }
        
        # Clean and prepare query
        query = claim_text.strip()
        if len(query) > 200:  # Truncate very long queries
            query = query[:200] + "..."
        
        try:
            # Make API request
            response = self._make_api_request(query)
            
            # Add rate limiting delay
            time.sleep(self.rate_limit_delay)
            
            if not response:
                return {
                    'query': claim_text,
                    'verification_status': 'Not Verifiable',
                    'rating': 'Not Verifiable',
                    'confidence': 0.0,
                    'source': 'API Error',
                    'source_url': '',
                    'claims_found': 0,
                    'details': []
                }
            
            # Extract claims
            claims = response.get('claims', [])
            
            if not claims:
                return {
                    'query': claim_text,
                    'verification_status': 'Not Verifiable',
                    'rating': 'Not Verifiable',
                    'confidence': 0.0,
                    'source': 'No matches found',
                    'source_url': '',
                    'claims_found': 0,
                    'details': []
                }
            
            # Process first/best claim
            best_claim = claims[0]
            claim_info = self._extract_claim_info(best_claim)
            
            # Classify rating
            standardized_rating = self._classify_rating(claim_info['rating'])
            
            # Calculate confidence based on source and rating clarity
            confidence = 0.5  # Base confidence
            if claim_info['source_name'] != 'Unknown':
                confidence += 0.3
            if standardized_rating != 'Not Verifiable':
                confidence += 0.2
            
            confidence = min(confidence, 1.0)
            
            return {
                'query': claim_text,
                'verification_status': standardized_rating,
                'rating': claim_info['rating'],
                'confidence': confidence,
                'source': claim_info['source_name'],
                'source_url': claim_info['source_url'],
                'claims_found': len(claims),
                'details': [self._extract_claim_info(claim) for claim in claims[:3]]  # Top 3 claims
            }
            
        except Exception as e:
            print(f"Error verifying claim '{claim_text}': {str(e)}")
            return {
                'query': claim_text,
                'verification_status': 'Not Verifiable',
                'rating': 'Error',
                'confidence': 0.0,
                'source': f'Error: {str(e)}',
                'source_url': '',
                'claims_found': 0,
                'details': []
            }
    
    def verify_claims_batch(self, claims):
        """
        Verify multiple claims
        
        Args:
            claims (list): List of claim texts
            
        Returns:
            list: List of verification results
        """
        print(f"Verifying {len(claims)} claims...")
        
        results = []
        
        for i, claim in enumerate(claims):
            print(f"Verifying claim {i+1}/{len(claims)}: {claim[:50]}...")
            
            result = self.verify_claim(claim)
            results.append(result)
            
            # Add longer delay between claims to respect rate limits
            if i < len(claims) - 1:
                time.sleep(0.5)
        
        print("Fact verification completed!")
        return results
    
    def get_verification_summary(self, verification_results):
        """
        Get summary statistics of verification results
        
        Args:
            verification_results (list): List of verification results
            
        Returns:
            dict: Summary statistics
        """
        if not verification_results:
            return {}
        
        # Count verification statuses
        status_counts = {}
        total_confidence = 0
        claims_with_sources = 0
        
        for result in verification_results:
            status = result['verification_status']
            confidence = result['confidence']
            
            if status not in status_counts:
                status_counts[status] = 0
            
            status_counts[status] += 1
            total_confidence += confidence
            
            if result['source'] not in ['None', 'API Error', 'No matches found']:
                claims_with_sources += 1
        
        # Calculate percentages
        summary = {
            'total_claims': len(verification_results),
            'average_confidence': total_confidence / len(verification_results),
            'claims_with_sources': claims_with_sources,
            'source_coverage': (claims_with_sources / len(verification_results)) * 100,
            'verification_distribution': {}
        }
        
        for status, count in status_counts.items():
            summary['verification_distribution'][status] = {
                'count': count,
                'percentage': (count / len(verification_results)) * 100
            }
        
        return summary


def test_fact_verifier():
    """
    Test function for the FactVerifier class
    """
    # Sample claims
    sample_claims = [
        "The COVID-19 vaccine is safe and effective.",
        "Climate change is causing global temperatures to rise.",
        "The Earth is flat.",
        "Water boils at 100 degrees Celsius.",
        "The population of Tokyo is over 13 million people."
    ]
    
    try:
        # Initialize fact verifier
        verifier = FactVerifier()
        
        # Test single verification
        result = verifier.verify_claim(sample_claims[0])
        print(f"Single verification: {result}")
        
        # Test batch verification (only first 2 claims to save API quota)
        results = verifier.verify_claims_batch(sample_claims[:2])
        
        for result in results:
            print(f"'{result['query'][:50]}...' -> {result['verification_status']} (confidence: {result['confidence']:.2f})")
        
        # Test summary
        summary = verifier.get_verification_summary(results)
        print(f"Summary: {summary}")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")


if __name__ == "__main__":
    test_fact_verifier()
