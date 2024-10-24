from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from functools import lru_cache
import pandas as pd
import joblib

app = FastAPI(title="Factory Recommendation System", version="1.0")

# CORS Configuration: Restrict to specific domains in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ftl-frontend-eight.vercel.app/", "https://ftl-frontend-j0llkdfas-jumas-projects-d97866b5.vercel.app", "https://ftl-frontend-j0llkdfas-jumas-projects-d97866b5.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load factory dataset and similarity matrix using in-memory caching
@lru_cache
def load_data():
    factories = pd.read_csv('factories.csv')
    similarity = joblib.load('similarity_matrix.pkl')
    return factories, similarity

df, similarity_matrix = load_data()

class UserInput(BaseModel):
    user_type: str
    factory_name: Optional[str] = None
    location: str
    industry_type: str
    production_capacity: int
    byproduct_type: Optional[str] = None
    byproduct_quantity: Optional[int] = None
    byproduct_frequency: Optional[str] = None
    required_byproduct: Optional[str] = None
    utilization_method: Optional[str] = None
    demand_scale: Optional[str] = None
    frequency_of_use: Optional[str] = None

def get_factory_index(factory_name: str) -> int:
    try:
        return df[df['Factory Name'].str.lower() == factory_name.lower()].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Factory not found")

def recommend_factories(factory_index: int, top_n: int = 5) -> List[dict]:
    similarity_scores = list(enumerate(similarity_matrix[factory_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommendations = [
        {"name": df.iloc[i[0]]['Factory Name'], 
         "location": df.iloc[i[0]]['Location'], 
         "industry": df.iloc[i[0]]['Industry Types']}
        for i in similarity_scores[1:top_n + 1]
    ]
    return recommendations

@app.post("/get-recommendations")
def get_recommendations(user_input: UserInput):
    if user_input.factory_name:
        factory_index = get_factory_index(user_input.factory_name)
        recommendations = recommend_factories(factory_index)
    else:
        if user_input.user_type == 'producer' and user_input.byproduct_type:
            matches = df[df['Required Byproduct'].str.contains(
                user_input.byproduct_type.strip(), case=False, na=False)]
        elif user_input.user_type == 'consumer' and user_input.required_byproduct:
            matches = df[df['Byproduct Type'].str.contains(
                user_input.required_byproduct.strip(), case=False, na=False)]
        else:
            raise HTTPException(status_code=400, detail="Invalid input for recommendation.")

        if matches.empty:
            return {"message": "No matching factories found", "factories": []}

        recommendations = matches[['Factory Name', 'Location', 'Industry Types']].to_dict(orient='records')

    return {"factories": recommendations}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Factory Recommendation System"}
