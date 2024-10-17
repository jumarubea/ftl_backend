from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI(title="Factory Recommendation System", version="1.0")

# Enable CORS to allow communication from any origin (frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify domains here, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load factory dataset and pre-trained similarity matrix
df = pd.read_csv('factories.csv')
similarity_matrix = joblib.load('similarity_matrix.pkl')

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
        return df[df['Factory Name'] == factory_name].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Factory not found")

def recommend_factories(factory_index: int, top_n: int = 5) -> List[str]:
    similarity_scores = list(enumerate(similarity_matrix[factory_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return [df.iloc[i[0]]['Factory Name'] for i in similarity_scores[1:top_n + 1]]

@app.post("/get-recommendations")
def get_recommendations(user_input: UserInput):
    if user_input.factory_name:
        factory_index = get_factory_index(user_input.factory_name)
        recommendations = recommend_factories(factory_index)
    else:
        if user_input.user_type == 'producer' and user_input.byproduct_type:
            matches = df[df['Required Byproduct'].str.contains(user_input.byproduct_type, case=False, na=False)]
        elif user_input.user_type == 'consumer' and user_input.required_byproduct:
            matches = df[df['Byproduct Type'].str.contains(user_input.required_byproduct, case=False, na=False)]
        else:
            raise HTTPException(status_code=400, detail="Invalid input.")

        if matches.empty:
            raise HTTPException(status_code=404, detail="No matching factories found.")

        recommendations = matches['Factory Name'].tolist()

    return recommendations

@app.get("/")
def read_root():
    return {"message": "Welcome to the Factory Recommendation System"}
