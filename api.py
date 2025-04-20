from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
import os
import json
import logging
# Import your recommendation engine
from recommendation_engine import SHLRecommendationEngine
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Initialize the app
app = FastAPI(title="SHL Assessment Recommendation API")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the recommendation engine (with environment variables for configuration)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "shl_assessments")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# Create the recommendation engine on startup
recommendation_engine = None
@app.on_event("startup")
async def startup_event():
    global recommendation_engine
    try:
        logger.info("Initializing SHL Recommendation Engine...")
        recommendation_engine = SHLRecommendationEngine(
            collection_name=COLLECTION_NAME,
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            llm_api_key=GEMINI_API_KEY,
            llm_model=GEMINI_MODEL
        )
        logger.info("SHL Recommendation Engine initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {str(e)}")
        raise
# Define request and response models
class Query(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10, description="Number of recommendations to retrieve (1-10)")
class AssessmentType(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]
    name: str
class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentType]
# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}
# Recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(query: Query):
    """Get assessment recommendations based on query"""
    if not recommendation_engine:
        raise HTTPException(status_code=500, detail="Recommendation engine not initialized")
   
    try:
        logger.info(f"Processing query: {query.query[:50]}... with top_k={query.top_k}")
       
        # Get recommendations using the engine
        recommendations = recommendation_engine.recommend(
            query=query.query,
            top_k=query.top_k,  # Use the user-specified number of recommendations
            min_k=1    # Return at least 1 recommendation
        )
       
        # Format and return results
        return {"recommended_assessments": recommendations.get("recommended_assessments", [])}
   
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
# Run the API server when executed directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)