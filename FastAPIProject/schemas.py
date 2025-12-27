from pydantic import BaseModel, Field
from typing import Optional


class CancionEntrada(BaseModel):
    instrumentalness: float = Field(..., ge=0, le=1)
    speechiness: float = Field(..., ge=0, le=1)
    danceability: float = Field(..., ge=0, le=1)
    valence: float = Field(..., ge=0, le=1)
    tempo: float = Field(..., ge=40, le=250)