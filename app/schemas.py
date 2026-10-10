from pydantic import BaseModel, Field


class Source(BaseModel):
    file: str
    page: int
    score: float
    snippet: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str = "default"  # one ID per chat; keeps chat memory separate


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentsResponse(BaseModel):
    files: list[str]
    chunks: int


class UploadResponse(BaseModel):
    added_files: list[str]
    chunks_added: int
    total_chunks: int