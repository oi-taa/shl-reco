from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
import os
import json
import logging
from recommendation_engine import SHLRecommendationEngine
from dotenv import load_dotenv
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Initialize the app
app = FastAPI(title="SHL Assessment Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Zilliz Cloud connection parameters - read from environment with no defaults for sensitive data
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_USER = os.getenv("ZILLIZ_USER")
ZILLIZ_PASSWORD = os.getenv("ZILLIZ_PASSWORD")
ZILLIZ_SECURE = os.getenv("ZILLIZ_SECURE", "True").lower() == "true"
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "shl_assessments")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Fallback for local development
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

recommendation_engine = None
@app.on_event("startup")
async def startup_event():
    global recommendation_engine
    try:
        logger.info("Initializing SHL Recommendation Engine...")
        
        # Check if we have the required Zilliz credentials
        if all([ZILLIZ_URI, ZILLIZ_USER, ZILLIZ_PASSWORD]):
            logger.info("Using Zilliz Cloud for vector database")
            recommendation_engine = SHLRecommendationEngine(
                collection_name=COLLECTION_NAME,
                llm_api_key=GEMINI_API_KEY,
                llm_model=GEMINI_MODEL,
                zilliz_uri=ZILLIZ_URI,
                zilliz_user=ZILLIZ_USER,
                zilliz_password=ZILLIZ_PASSWORD,
                zilliz_secure=ZILLIZ_SECURE
            )
        else:
            logger.warning("Zilliz Cloud credentials not found, falling back to local Milvus")
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
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

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
            top_k=query.top_k,  
            min_k=1   
        )
       
        return {"recommended_assessments": recommendations.get("recommended_assessments", [])}
   
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)