import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """
    Class for evaluating recommendation systems using standard metrics
    """
    
    def precision_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate precision@k for a single query
        
        Parameters:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            float: Precision@k
        """
        # Make sure we only consider the first k recommendations
        recommended_k = recommended_items[:k]
        
        # Count number of relevant items in the top-k recommendations
        if not recommended_k:
            return 0.0
            
        # Count matches
        matches = sum(1 for item in recommended_k if item in relevant_items)
        
        # Calculate precision@k
        return matches / min(k, len(recommended_k))
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate recall@k for a single query
        
        Parameters:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            float: Recall@k
        """
        # Make sure we only consider the first k recommendations
        recommended_k = recommended_items[:k]
        
        # Handle edge cases
        if not relevant_items:
            return 1.0  # By convention, if no relevant items, recall is 1.0
            
        # Count matches
        matches = sum(1 for item in recommended_k if item in relevant_items)
        
        # Calculate recall@k
        return matches / len(relevant_items)
    
    def average_precision(self, recommended_items, relevant_items):
        """
        Calculate average precision for a single query
        
        Parameters:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            
        Returns:
            float: Average precision
        """
        if not relevant_items:
            return 0.0
            
        # Track running sum of precision@k values for relevant items
        ap_sum = 0.0
        num_hits = 0
        
        # Calculate precision@k for each position where a relevant item was found
        for k, item in enumerate(recommended_items, 1):
            if item in relevant_items:
                num_hits += 1
                # Precision up to this point
                precision_at_position = num_hits / k
                ap_sum += precision_at_position
        
        # Normalize by total number of relevant items
        return ap_sum / len(relevant_items) if len(relevant_items) > 0 else 0.0
    
    def mean_average_precision_at_k(self, all_recommended_items, all_relevant_items, k):
        """
        Calculate Mean Average Precision@k across multiple queries
        
        Parameters:
            all_recommended_items: List of lists of recommended item IDs for each query
            all_relevant_items: List of lists of relevant item IDs for each query
            k: Number of recommendations to consider
            
        Returns:
            float: MAP@k
        """
        if not all_recommended_items or len(all_recommended_items) == 0:
            return 0.0
            
        # Calculate AP@k for each query
        ap_values = []
        for recommended, relevant in zip(all_recommended_items, all_relevant_items):
            # Trim recommendations to top-k
            recommended_k = recommended[:k]
            # Calculate AP for this query
            ap = self.average_precision(recommended_k, relevant)
            ap_values.append(ap)
        
        # Calculate MAP as the mean of AP values
        return sum(ap_values) / len(ap_values) if ap_values else 0.0
    
    def mean_recall_at_k(self, all_recommended_items, all_relevant_items, k):
        """
        Calculate Mean Recall@k across multiple queries
        
        Parameters:
            all_recommended_items: List of lists of recommended item IDs for each query
            all_relevant_items: List of lists of relevant item IDs for each query
            k: Number of recommendations to consider
            
        Returns:
            float: Mean Recall@k
        """
        if not all_recommended_items or len(all_recommended_items) == 0:
            return 0.0
            
        # Calculate recall@k for each query
        recall_values = []
        for recommended, relevant in zip(all_recommended_items, all_relevant_items):
            # Calculate recall for this query
            recall = self.recall_at_k(recommended, relevant, k)
            recall_values.append(recall)
        
        # Calculate mean recall as the mean of recall values
        return sum(recall_values) / len(recall_values) if recall_values else 0.0