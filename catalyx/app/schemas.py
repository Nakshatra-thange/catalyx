from pydantic import BaseModel, Field, field_validator


class SearchParams(BaseModel):
    q: str = Field(..., min_length=1, max_length=300)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=50)

    @field_validator("q")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query cannot be empty or whitespace only")
        return v