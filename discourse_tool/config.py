from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    default_model: str = "mistral"
    similarity_threshold: float = 0.3
    embedding_model: str = "all-MiniLM-L6-v2"
    modelfile_template: str = 'FROM {base_model}\nSYSTEM """{system_prompt}"""'

    @property
    def segments_dir(self) -> Path:
        return self.data_dir / "segments"

    @property
    def evaluations_dir(self) -> Path:
        return self.data_dir / "evaluations"

    @property
    def training_dir(self) -> Path:
        return self.data_dir / "training"

    def ensure_dirs(self) -> None:
        for d in (self.segments_dir, self.evaluations_dir, self.training_dir):
            d.mkdir(parents=True, exist_ok=True)
