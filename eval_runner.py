import argparse
import json
import logging
import os
from eval import EvaluationMetrics
from recommendation_engine import SHLRecommendationEngine
from dotenv import load_dotenv

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "shl_assessments")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def load_ground_truth(file_path='ground_truth.json'):
    """Load ground truth data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping of queries to their relevant assessments
    ground_truth = {}
    for entry in data.get('queries', []):
        query = entry.get('query', '')
        relevant_assessments = entry.get('relevant_assessments', [])
        ground_truth[query] = relevant_assessments
    
    return ground_truth

def extract_ground_truth_from_file(file_path):
    """Extract ground truth from a JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping of queries to their relevant assessments
    ground_truth = {}
    for entry in data.get('queries', []):
        query = entry.get('query', '')
        relevant_assessments = entry.get('relevant_assessments', [])
        ground_truth[query] = relevant_assessments
        logger.info(f"Found query with {len(relevant_assessments)} relevant assessments")
    
    return ground_truth

def save_ground_truth(ground_truth, file_path='ground_truth.json'):
    """Save ground truth data to JSON file"""
    data = {'queries': []}
    for query, relevant_assessments in ground_truth.items():
        data['queries'].append({
            'query': query,
            'relevant_assessments': relevant_assessments
        })
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Ground truth saved to {file_path}")

def run_evaluation(engine, ground_truth, k_values=[1, 3, 5, 10]):
    """Run evaluation for the SHL recommendation engine"""
    # Get test queries from ground truth
    test_queries = list(ground_truth.keys())
    
    # Evaluate
    results = engine.evaluate(test_queries, ground_truth, k_values)
    
    # Log and return results
    logger.info("Evaluation results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return results



if __name__ == "__main__":
    # Initialize the recommendation engine
    # Replace these with your actual Milvus connection details and API key
    engine = SHLRecommendationEngine(
            collection_name=COLLECTION_NAME,
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            llm_api_key=GEMINI_API_KEY,
            llm_model=GEMINI_MODEL
        )
    
    # Load ground truth from eval_check.json
    ground_truth = extract_ground_truth_from_file('eval_check.json')
    logger.info(f"Loaded {len(ground_truth)} queries from eval_check.json")
    
    # Save ground truth for future use (optional)
    save_ground_truth(ground_truth)
    
    # Run evaluation
    k_values = [1, 3, 5, 10]
    results = run_evaluation(engine, ground_truth, k_values)
    
    # Save evaluation results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation completed and results saved to evaluation_results.json")