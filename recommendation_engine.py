import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import logging
import time
from typing import Dict, List, Optional, Union, Any
from google import genai
from eval import EvaluationMetrics

# Import Google's Generative AI client
try:
    from google import genai
except ImportError:
    logging.warning("Google genai package not installed. LLM features will be limited.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHLRecommendationEngine:
    # Modified __init__ method
    def __init__(self, collection_name="shl_assessments", model_name="all-MiniLM-L6-v2", 
             host="localhost", port="19530", llm_api_key=None, llm_model="gemini-pro"):
        """
        Initialize the SHL Assessment Recommendation Engine
        """
        # Connect to Milvus
        self._connect_to_milvus(host, port)
        
        # Load the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Load the collection
        logger.info(f"Loading collection: {collection_name}")
        self.collection = Collection(name=collection_name)
        self.collection.load()
        
        # Cache some frequently used values for faster processing
        self.collection_name = collection_name
        
        # Set up LLM API credentials
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        
        # Create a client instance if possible
        self.genai_client = None
        if llm_api_key and 'genai' in globals():
            try:
                from google import genai
                self.genai_client = genai.Client(api_key=llm_api_key)
                logger.info(f"Gemini client initialized for model: {llm_model}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {str(e)}")
        
        logger.info("Recommendation engine initialized")
    
    def _connect_to_milvus(self, host, port):
        """Connect to Milvus server"""
        try:
            logger.info(f"Connecting to Milvus at {host}:{port}")
            connections.connect("default", host=host, port=port)
            logger.info("Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def extract_parameters_with_llm(self, query: str) -> Dict[str, Any]:
        """
        Extract parameters from the query using an LLM
        
        This replaces the simple regex-based extraction methods with a more
        sophisticated LLM-based approach that can understand context better.
        
        Parameters:
            query (str): Natural language query or job description
            
        Returns:
            dict: Dictionary with extracted parameters
        """
        if not self.llm_api_key or 'genai' not in globals():
            logger.warning("LLM API credentials not provided or genai not available, falling back to basic extraction")
            # Fallback to basic extraction
            return {
                "skills": self._basic_extract_skills(query),
                "time_limit": self._basic_extract_time_constraints(query),
                "remote_testing": None,
                "adaptive_testing": None,
                "test_types": [],
                "seniority_level": None,
                "core_query": query
            }
        
        try:
            # Prepare prompt for the LLM
            prompt = f"""
            You are an expert HR assessment analyst. Extract structured information from this job description or query.
            Focus on identifying the right balance of broad skill categories and specific technical skills when relevant.
            
            JOB DESCRIPTION/QUERY:
            {query}
            
            Extract the following information with high precision:
            
            1. Main skill categories being assessed:
               - For general business skills, use umbrella terms like "Accounting", "Project Management", "Business Analysis"
               - For technical domains like programming, focus on extracting ONLY the specific languages, frameworks, and tools mentioned:
                 - Example: If "Python, SQL, JavaScript" are mentioned, extract ONLY ["Python", "SQL", "JavaScript"] WITHOUT adding generic terms like "Programming"
                 - Only include generic term "Programming" if it's explicitly mentioned without specific languages
               - For data skills, extract specific technologies like "SQL", "Tableau", "R" directly WITHOUT generic categories unless explicitly mentioned
               - For soft skills, use general categories like "Communication", "Leadership", "Critical Thinking"
               - Limit to 3-5 most important skill categories/specific skills, prioritizing what's emphasized in the job description
               
            2. Time constraints (in minutes, if mentioned)
            3. Remote testing requirements (Yes/No, if mentioned)
            4. Adaptive testing requirements (Yes/No, if mentioned)
            5. Test types needed (use broad categories like: cognitive, personality, technical knowledge, aptitude)
            6. Seniority level or position type (e.g., entry-level, mid-level, senior, manager)
            7. Job title or role (specific position being hired for)
            8. Core query intent (a concise representation of what the user is looking for)
            
            Format your response ONLY as a valid JSON object with these keys (no explanation, only JSON):
            {{
              "skills": ["skill1", "skill2", ...],
              "time_limit": integer or null,
              "remote_testing": "Yes", "No", or null,
              "adaptive_testing": "Yes", "No", or null,
              "test_types": ["type1", "type2", ...],
              "seniority_level": "string or null",
              "job_title": "string or null",
              "core_query": "string"
            }}
            
            IMPORTANT GUIDELINES:
            - For technical skills like programming languages, databases, and frameworks, extract ONLY specific named skills (Python, SQL, JavaScript, etc.)
            - Do NOT add generic umbrella terms like "Programming" or "Development" for technical skills unless explicitly mentioned
            - For non-technical roles, focus on broader skill categories
            - For specialized fields (finance, healthcare, etc.), include domain-specific knowledge areas
            - Ensure test types are general assessment categories, not specific tests
            - Provide only the structured JSON with no additional text
            
            Query: {query}
            """
                        
            # Call LLM API
            response = self._call_llm_api(prompt)
            logger.debug(f"Processed LLM response: {response[:100]}...")
            print("response:",  response)
            
            response = response.strip()
            if not response:
                raise ValueError("Empty response from LLM")
            
            # Try to parse the JSON response with additional error handling
            try:
                extracted_params = json.loads(response)
            except json.JSONDecodeError as e:
                # Try to clean up common JSON formatting issues
                # 1. Fix single quotes instead of double quotes
                fixed_response = response.replace("'", "\"")
                # 2. Fix unquoted keys
                import re
                fixed_response = re.sub(r'(\w+):', r'"\1":', fixed_response)
                
                try:
                    extracted_params = json.loads(fixed_response)
                except json.JSONDecodeError:
                    # If still failing, raise the original error
                    logger.error(f"Failed to parse JSON response after cleanup attempts: {response}")
                    raise e
            
            logger.info(f"LLM extracted parameters: {extracted_params}")
            
            # Validate and ensure all expected keys are present
            expected_keys = ["skills", "time_limit", "remote_testing", "adaptive_testing", 
                             "test_types", "seniority_level", "core_query"]
            for key in expected_keys:
                if key not in extracted_params:
                    logger.warning(f"Missing expected key in LLM response: {key}")
                    if key == "skills":
                        extracted_params[key] = self._basic_extract_skills(query)
                    elif key == "time_limit":
                        extracted_params[key] = self._basic_extract_time_constraints(query)
                    elif key in ["test_types"]:
                        extracted_params[key] = []
                    else:
                        extracted_params[key] = None
            
            return extracted_params
        except Exception as e:
            logger.error(f"Error in LLM parameter extraction: {str(e)}")
            logger.warning("Falling back to basic extraction")
            
            # Fallback to basic extraction
            return {
                "skills": self._basic_extract_skills(query),
                "time_limit": self._basic_extract_time_constraints(query),
                "remote_testing": None,
                "adaptive_testing": None,
                "test_types": [],
                "seniority_level": None,
                "core_query": query
            }

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call Gemini API using the client instance
        
        Parameters:
            prompt (str): The prompt to send to Gemini
                
        Returns:
            str: LLM response as a JSON string
        """
        if not self.genai_client:
            logger.warning("Gemini client not available, returning empty response")
            return "{}"
        
        try:
            from google import genai
            from google.genai import types
            
            # Use the client to generate content
            response = self.genai_client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature =  0.1,
                    max_output_tokens = 1000,
                    top_p = 0.95,
                    top_k = 40
                ),
            )
            raw_text = ""
            if hasattr(response, 'text'):
                raw_text = response.text
            else:
                # Try to extract text from the response
                raw_text = str(response)
            
            logger.debug(f"Raw LLM response: {raw_text[:100]}...")
            
            # Look for JSON content in the response - it might be wrapped in ```json blocks
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
            if json_match:
                # Extract just the JSON part from code blocks
                return json_match.group(1).strip()
            
            # If no code blocks found, try to find anything that looks like JSON
            json_match = re.search(r'(\{[\s\S]*\})', raw_text)
            if json_match:
                return json_match.group(1).strip()
                
            # Return raw text as a last resort
            return raw_text
                
        except Exception as e:
            logger.error(f"Exception in Gemini API call: {str(e)}")
            return "{}"
    def _extract_json_from_llm_response(self, text):
        """
        Helper method to extract JSON from LLM responses
        
        Parameters:
            text (str): Raw text from LLM responsef
            
        Returns:
            object: Parsed JSON object or None if parsing fails
        """
        if not text:
            return None
        
        try:
            # First try: direct JSON parsing
            return json.loads(text)
        except json.JSONDecodeError:
            # Second try: clean up text by removing # characters
            cleaned_text = text.replace('#', '')
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass
            
            # Third try: look for JSON in code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                try:
                    cleaned_content = json_match.group(1).strip().replace('#', '')
                    return json.loads(cleaned_content)
                except json.JSONDecodeError:
                    pass
            
            # Fourth try: find anything that looks like JSON
            json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
            if json_match:
                try:
                    cleaned_content = json_match.group(1).strip().replace('#', '')
                    return json.loads(cleaned_content)
                except json.JSONDecodeError:
                    pass
            
            # Fifth try: extract numbers using regex
            number_pattern = r'#?(\d+)'
            matches = re.findall(number_pattern, text)
            if matches:
                # Extract just the numbers
                extracted_numbers = [int(num) for num in matches]
                if extracted_numbers:
                    return extracted_numbers
            
            # If all attempts fail, return None
            logger.warning(f"Failed to extract JSON from response: {text[:100]}...")
            return None
    # Keep the basic extraction methods as fallbacks
    def _basic_extract_time_constraints(self, query):
        """Extract time constraints from the query using regex"""
        import re
        time_pattern = r'(\d+)\s*(min|minute|minutes)'
        match = re.search(time_pattern, query.lower())
        
        if match:
            return int(match.group(1))
        return None
    
    def _basic_extract_skills(self, query):
        """Extract mentioned skills from the query using predefined list"""
        skills = ["python", "java", "javascript", "sql", "c++", "c#", ".net", 
                 "ruby", "php", "typescript", "react", "angular", "vue", 
                 "leadership", "management", "communication", "teamwork", 
                 "analytics", "problem solving", "critical thinking", "collaboration"]
        
        found_skills = []
        query_lower = query.lower()
        
        for skill in skills:
            if skill.lower() in query_lower:
                found_skills.append(skill)
                
        return found_skills
    
    def get_query_embedding(self, query):
        """Convert query to embedding vector using the same model as indexing"""
        # Handle potential batching by ensuring we have a single item
        if isinstance(query, list):
            embeddings = self.model.encode(query)
            return embeddings
        else:
            embedding = self.model.encode([query])[0]  # Get first item from batch
            return embedding
    
    def search_similar_assessments(self, query_vector, params=None, top_k=10):
        """
        Search for similar assessments using vector similarity
        
        Parameters:
            query_vector: The embedding vector for the query
            params: Dictionary of parameters extracted from the query
            top_k: Maximum number of results to return
        """
        start_time = time.time()
        
        search_params = {
            "metric_type": "COSINE",  # Use cosine similarity
            "params": {"ef": 100}     # Higher ef value improves recall but increases search time
        }

        
        # Define output fields
        output_fields = ["name", "url", "description", "duration", 
                         "remote_support", "adaptive_support", "test_type_json"]
        
        # Execute search
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k * 2,  # Get more results for filtering
            output_fields=output_fields
        )
        
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.3f} seconds")
        
        return results
    
    def format_search_results(self, results):
        """Format Milvus search results into assessment recommendations"""
        recommendations = []
        
        # Process each hit
        for hit in results[0]:
            try:
                # Access fields directly as attributes
                fields = {}
                for field in ["name", "url", "description", "duration", 
                             "remote_support", "adaptive_support", "test_type_json"]:
                    if hasattr(hit, field):
                        fields[field] = getattr(hit, field)
                
                # Parse test_type from JSON string
                test_type_json = fields.get("test_type_json", "[]")
                test_type = json.loads(test_type_json)
                
                # Create recommendation item
                recommendation = {
                    "url": fields.get("url", ""),
                    "adaptive_support": fields.get("adaptive_support", "No"),
                    "description": fields.get("description", ""),
                    "duration": int(fields.get("duration", 0)),
                    "remote_support": fields.get("remote_support", "No"),
                    "test_type": test_type,
                    "name": fields.get("name", "")
                }
                
                recommendations.append(recommendation)
            except Exception as e:
                logger.error(f"Error processing hit: {str(e)}")
        
        # Return all recommendations after processing all hits
        return recommendations
    
    def filter_by_constraints(self, recommendations, params):
        """
        Filter recommendations by constraints extracted from query
        
        Parameters:
            recommendations: List of recommendation items
            params: Dictionary of parameters extracted from the query
        """
        filtered_recommendations = recommendations
        
        # Filter by time limit if specified
        if params.get("time_limit") is not None:
            filtered_recommendations = [r for r in filtered_recommendations 
                                        if r['duration'] <= params["time_limit"]]
            logger.info(f"Filtered to {len(filtered_recommendations)} results after applying time constraint")
        
        # Filter by remote testing support if specified
        if params.get("remote_testing") in ["Yes", "No"]:
            filtered_recommendations = [r for r in filtered_recommendations 
                                        if r['remote_support'] == params["remote_testing"]]
            logger.info(f"Filtered to {len(filtered_recommendations)} results after applying remote testing filter")
        
        # Filter by adaptive testing support if specified
        if params.get("adaptive_testing") in ["Yes", "No"]:
            filtered_recommendations = [r for r in filtered_recommendations 
                                        if r['adaptive_support'] == params["adaptive_testing"]]
            logger.info(f"Filtered to {len(filtered_recommendations)} results after applying adaptive testing filter")
        
        return filtered_recommendations
    
    def recommend(self, query, top_k=10, min_k=1):
        """
        Generate assessment recommendations based on a natural language query
        with similarity scores between query and assessments
        
        Parameters:
            query (str): Natural language query or job description
            top_k (int): Maximum number of recommendations to return
            min_k (int): Minimum number of recommendations to return
                
        Returns:
            dict: Dictionary with recommended assessments and similarity scores
        """
        logger.info(f"Processing query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Extract parameters using LLM
        params = self.extract_parameters_with_llm(query)
        logger.info(f"Extracted parameters: {params}")
        
        # Original query embedding
        query_vector = self.get_query_embedding(query)
        
        # Generate structured description-based embedding if possible
        if params and "skills" in params:
            # Get test types and skills from params
            test_types = params.get("test_types", [])
            skills = params.get("skills", [])
            
            # Format structured text similar to how we created the original embeddings
            structured_text = f"Assessment for {params.get('seniority_level', '')} with skills {', '.join(skills)} "
            
            # Create embedding for the structured text
            desc_vector = self.get_query_embedding(structured_text)
            logger.info("Generated structured description embedding for matching")
            
            # Search using the structured description embedding
            desc_search_results = self.search_similar_assessments(desc_vector, params, top_k=top_k*3)
            desc_recommendations = self.format_search_results(desc_search_results)
            
            # Compare with regular search
            regular_search_results = self.search_similar_assessments(query_vector, params, top_k=top_k*3)
            regular_recommendations = self.format_search_results(regular_search_results)
            
            # Log comparison info
            logger.info(f"Description-based search returned {len(desc_recommendations)} results")
            logger.info(f"Regular search returned {len(regular_recommendations)} results")
            
            # Optional: Calculate overlap between the two result sets
            desc_urls = set(r['url'] for r in desc_recommendations)
            regular_urls = set(r['url'] for r in regular_recommendations)
            overlap = desc_urls.intersection(regular_urls)
            logger.info(f"Overlap between search methods: {len(overlap)}/{min(len(desc_urls), len(regular_urls))}")
            
            # Choose which recommendations to use (could be a parameter)
            recommendations = desc_recommendations
            search_method = "description_based"
        else:
            # If no generated description, just use regular search
            search_results = self.search_similar_assessments(query_vector, params, top_k=top_k*3)
            recommendations = self.format_search_results(search_results)
            search_method = "regular"

        
        
        # Rerank using LLM
        reranked_recommendations = self.rerank_with_llm(query, recommendations, params, top_k=top_k*2)
        
        # Apply filters based on extracted parameters
        if params and params.get("skills"):
            reranked_recommendations = self.boost_specialized_matches(reranked_recommendations, params.get("skills"))
    
        # Apply filters based on extracted parameters
        if params:
            filtered_recommendations = self.filter_by_constraints(reranked_recommendations, params)
        else:
            filtered_recommendations = reranked_recommendations
        
        # Ensure we have at least min_k results
        if len(filtered_recommendations) < min_k:
            logger.warning(f"Not enough results after filtering ({len(filtered_recommendations)}), relaxing constraints")
            filtered_recommendations = recommendations[:max(min_k, min(top_k, len(recommendations)))]
        else:
            # Limit to top_k
            filtered_recommendations = filtered_recommendations[:min(top_k, len(filtered_recommendations))]
        
        # Include metadata about the search
        result = {
            "search_method": search_method,
            "recommended_assessments": filtered_recommendations
        }
        
        # Add generated description if available
        if params and "generated_description" in params:
            result["generated_description"] = params["generated_description"]
        
        logger.info(f"Returning {len(filtered_recommendations)} recommendations using {search_method} search")
        return result

    def rerank_with_llm(self, query, recommendations, params=None, top_k=10):
        """
        Rerank the retrieved assessments using LLM to better match query intent
        
        Parameters:
            query (str): Original user query
            recommendations: List of assessment recommendations
            params: Extracted parameters from query
            top_k: Maximum number of results to return
            
        Returns:
            list: Reranked list of recommendations
        """
        if not self.genai_client or len(recommendations) <= 1:
            logger.info("Skipping reranking: LLM not available or too few results")
            return recommendations
        
        try:
            # Get the generated description if available
            query_description = params.get("generated_description", "")
            enhanced_query = f"{query}\n\nGenerated assessment profile: {query_description}" if query_description else query
            
            # Create a summary of each recommendation for the LLM to evaluate
            assessment_summaries = []
            for i, rec in enumerate(recommendations):
                summary = f"Assessment #{i+1}: {rec['name']}\n"
                summary += f"Description: {rec['description']}\n"
                summary += f"Type: {', '.join(rec['test_type'])}\n"
                summary += f"Duration: {rec['duration']} minutes\n"
                summary += f"Remote support: {rec['remote_support']}\n"
                assessment_summaries.append(summary)
            
            # Combine all summaries
            all_assessments = "\n\n".join(assessment_summaries)
            
            # Create prompt for reranking
            prompt = f"""
            You are an assessment matching expert. Rank these assessments based on how well they match the user's needs.
    
            USER QUERY: {query}
            
            ASSESSMENT REQUIREMENTS:
            {query_description}
            
            AVAILABLE ASSESSMENTS:
            {all_assessments}
            
            Consider these factors in your ranking, Rank the assessments based on how well they match the user's profile, with a strong emphasis on SKILL MATCH.
            1. Overall match to the user's requirements and intent
            2. Alignment between assessment description and the user's needs
            3. . Coverage of specific skills mentioned - REQUIRED SKILLS: {', '.join(params.get('skills', []))}
               IMPORTANT: Ensure that assessments covering ALL required skills, especially SQL, Python, and JavaScript, are prioritized in your ranking.
            4. Appropriate test types for the role
            5. Seniority level alignment
            6. Time and format requirements
            
            RESPOND WITH ONLY A COMMA-SEPARATED LIST OF ASSESSMENT NUMBERS, with the best match first.
            Example correct response: 3,1,5,2,4
            Important: 
            DO NOT include brackets, JSON formatting, explanations, or any other text.
            """
            
            # Call LLM for reranking
            response = self._call_llm_api(prompt)
            logger.debug(f"Raw LLM ranking response: {response[:200]}...")
            
            # Parse ranking from response
            ranking = self._extract_json_from_llm_response(response)
            
            if ranking and isinstance(ranking, list):
                # If we have a nested array (first element is a list), flatten it
                if len(ranking) > 0 and isinstance(ranking[0], list):
                    logger.info("Detected nested array in LLM response, flattening")
                    ranking = ranking[0]  # Take the first list
                
                # Create a list of valid indices
                valid_rankings = []
                for item in ranking:
                    try:
                        # Convert to integer if it's a string or has # prefix
                        if isinstance(item, str):
                            # Remove any # prefix
                            item = item.replace('#', '')
                            if item.isdigit():
                                item = int(item)
                        
                        # Make sure it's an integer
                        if not isinstance(item, int):
                            logger.warning(f"Skipping non-integer ranking value: {item}")
                            continue
                        
                        # Check if it's in range (1-based to 0-based)
                        index = item - 1  # Convert from 1-indexed to 0-indexed
                        if 0 <= index < len(recommendations):
                            valid_rankings.append(index)
                        else:
                            logger.warning(f"Ranking index out of range: {item}")
                    except Exception as e:
                        logger.warning(f"Error processing ranking item {item}: {str(e)}")
                
                # Check if we have any valid rankings
                if valid_rankings:
                    # Create reranked list
                    reranked = [recommendations[i] for i in valid_rankings]
                    
                    # Add any missing recommendations at the end
                    seen_indices = set(valid_rankings)
                    for i, rec in enumerate(recommendations):
                        if i not in seen_indices:
                            reranked.append(rec)
                    
                    logger.info(f"Reranking successful, new order (first 5): {valid_rankings[:5]}...")
                    return reranked[:top_k]
                else:
                    logger.warning("No valid rankings found after filtering")
            else:
                logger.warning(f"Invalid ranking format returned: {ranking}")
            
        except Exception as e:
            logger.error(f"Error in LLM reranking: {str(e)}")
        
        # If reranking fails, return original recommendations
        return recommendations

    def hybrid_search(self, query, query_vector, params, top_k=30):
        """Combine multiple retrieval methods to maximize recall"""
        results_by_url = {}  # Use URL as unique ID to avoid duplicates
        
        # 1. Vector similarity search (semantic matching)
        vector_results = self.search_similar_assessments(query_vector, params, top_k=top_k)
        vector_recommendations = self.format_search_results(vector_results)
        
        # Add vector results to combined results
        for rec in vector_recommendations:
            results_by_url[rec['url']] = rec
        
        # 2. Direct skill name matching (exact matching)
        if params and params.get("skills"):
            for skill in params.get("skills"):
                try:
                    # Search for exact skill name in assessment name or description
                    skill_expr = f"name LIKE '%{skill}%' OR description LIKE '%{skill}%'"
                    keyword_results = self.collection.query(
                        expr=skill_expr,
                        output_fields=["name", "url", "description", "duration", 
                                      "remote_support", "adaptive_support", "test_type_json"],
                        limit=top_k
                    )
                    
                    # Format and add keyword results
                    for hit in keyword_results:
                        rec = {field: getattr(hit, field, "") for field in 
                              ["name", "url", "description", "duration", 
                               "remote_support", "adaptive_support"]}
                        
                        # Parse test type JSON
                        test_type_json = getattr(hit, "test_type_json", "[]")
                        rec["test_type"] = json.loads(test_type_json)
                        
                        # Add to combined results if not already present
                        if rec['url'] not in results_by_url:
                            results_by_url[rec['url']] = rec
                except Exception as e:
                    logger.error(f"Keyword search error for skill '{skill}': {e}")
        
        # 3. Test type matching (role-based matching)
        if params and params.get("test_types"):
            try:
                # Create expression to match any test type
                test_types = params.get("test_types")
                test_type_conditions = []
                
                for test_type in test_types:
                    # Carefully escape the test type for the query
                    safe_test_type = test_type.replace("'", "''")
                    test_type_conditions.append(f"test_type_json LIKE '%{safe_test_type}%'")
                
                if test_type_conditions:
                    test_type_expr = " OR ".join(test_type_conditions)
                    test_type_results = self.collection.query(
                        expr=test_type_expr,
                        output_fields=["name", "url", "description", "duration", 
                                      "remote_support", "adaptive_support", "test_type_json"],
                        limit=top_k
                    )
                    
                    # Format and add test type results
                    for hit in test_type_results:
                        rec = {field: getattr(hit, field, "") for field in 
                              ["name", "url", "description", "duration", 
                               "remote_support", "adaptive_support"]}
                        
                        # Parse test type JSON
                        test_type_json = getattr(hit, "test_type_json", "[]")
                        rec["test_type"] = json.loads(test_type_json)
                        
                        # Add to combined results if not already present
                        if rec['url'] not in results_by_url:
                            results_by_url[rec['url']] = rec
            except Exception as e:
                logger.error(f"Test type search error: {e}")
        
        # 4. Job title/role matching
        if params and params.get("job_title"):
            try:
                # Search for assessments matching job title
                job_title = params.get("job_title").replace("'", "''")
                job_expr = f"name LIKE '%{job_title}%' OR description LIKE '%{job_title}%'"
                job_results = self.collection.query(
                    expr=job_expr,
                    output_fields=["name", "url", "description", "duration", 
                                  "remote_support", "adaptive_support", "test_type_json"],
                    limit=top_k
                )
                
                # Format and add job title results
                for hit in job_results:
                    rec = {field: getattr(hit, field, "") for field in 
                          ["name", "url", "description", "duration", 
                           "remote_support", "adaptive_support"]}
                    
                    # Parse test type JSON
                    test_type_json = getattr(hit, "test_type_json", "[]")
                    rec["test_type"] = json.loads(test_type_json)
                    
                    # Add to combined results if not already present
                    if rec['url'] not in results_by_url:
                        results_by_url[rec['url']] = rec
            except Exception as e:
                logger.error(f"Job title search error: {e}")
        
        # Combine all results
        combined_results = list(results_by_url.values())
        
        # Log statistics
        logger.info(f"Hybrid search retrieved {len(combined_results)} unique results")
        
        return combined_results
    # Add this after reranking but before filtering
    def boost_specialized_matches(self, recommendations, query_skills):
        """Enhanced boosting with primary and secondary skill matches"""
        primary_matches = []
        secondary_matches = []
        remainder = []
        
        for rec in recommendations:
            desc_lower = rec['description'].lower()
            name_lower = rec['name'].lower()
            
            # Calculate skill match score
            skill_match_score = 0
            matched_skills = set()
            
            for skill in query_skills:
                skill_lower = skill.lower()
                # Name match is strongest signal
                if skill_lower in name_lower:
                    skill_match_score += 3
                    matched_skills.add(skill)
                # Description match is secondary signal
                elif skill_lower in desc_lower:
                    skill_match_score += 1
                    matched_skills.add(skill)
            
            # Categorize based on match quality
            if skill_match_score >= 3:
                primary_matches.append((rec, skill_match_score, len(matched_skills)))
            elif skill_match_score > 0:
                secondary_matches.append((rec, skill_match_score, len(matched_skills)))
            else:
                remainder.append(rec)
        
        # Sort primary and secondary matches by score and number of matched skills
        primary_matches.sort(key=lambda x: (x[1], x[2]), reverse=True)
        secondary_matches.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Recombine in priority order
        result = [item[0] for item in primary_matches] + [item[0] for item in secondary_matches] + remainder
        
        logger.info(f"Enhanced boosting: {len(primary_matches)} primary, {len(secondary_matches)} secondary matches")
        return result
    
    def evaluate(self, test_queries, ground_truth, k_values=[1, 3, 5, 10]):
        """
        Evaluate the recommendation engine on test queries
        
        Parameters:
            test_queries: List of test queries
            ground_truth: Dictionary mapping queries to lists of relevant assessment IDs
            k_values: List of k values to evaluate at
            
        Returns:
            dict: Dictionary containing evaluation results (Mean Recall@K and MAP@K)
        """
        
        results = {}
        all_recommended_items = []
        all_relevant_items = []
        
        # Process each test query
        for query in test_queries:
            # Get recommendations
            recommendations = self.recommend(query, top_k=max(k_values))
            
            # Extract IDs of recommended assessments (using URL as identifier)
            recommended_ids = [
                rec.get("url", "")
                for rec in recommendations.get("recommended_assessments", [])
            ]
            
            # Get relevant items for this query
            relevant_ids = ground_truth.get(query, [])
            
            # Store for calculating aggregate metrics
            all_recommended_items.append(recommended_ids)
            all_relevant_items.append(relevant_ids)
        
        # Calculate metrics for each k value
        metrics = EvaluationMetrics()
        for k in k_values:
            results[f"mean_recall@{k}"] = metrics.mean_recall_at_k(
                all_recommended_items, all_relevant_items, k
            )
            results[f"map@{k}"] = metrics.mean_average_precision_at_k(
                all_recommended_items, all_relevant_items, k
            )
        
        return results