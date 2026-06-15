from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    default_model: str = "mistral"
    similarity_threshold: float = 0.3
    embedding_model: str = "all-MiniLM-L6-v2"
    modelfile_template: str = 'FROM {base_model}\nSYSTEM """{system_prompt}"""'
    # Intermediary (batch comparative) evaluation defaults
    context_window: int = 32000
    evaluator_overlap: int = 5
    evaluator_seed: int = 42

    @property
    def segments_dir(self) -> Path:
        return self.data_dir / "segments"

    @property
    def evaluations_dir(self) -> Path:
        return self.data_dir / "evaluations"

    @property
    def training_dir(self) -> Path:
        return self.data_dir / "training"

    @property
    def similarity_dir(self) -> Path:
        return self.data_dir / "similarity"

    @property
    def intermediary_dir(self) -> Path:
        return self.data_dir / "intermediary"

    def ensure_dirs(self) -> None:
        for d in (self.segments_dir, self.evaluations_dir, self.training_dir, self.similarity_dir, self.intermediary_dir):
            d.mkdir(parents=True, exist_ok=True)
