import yaml
from dataclasses import dataclass, asdict
from typing import List, Union


@dataclass
class PlateConfig:
    # Dataset Configuration
    alphabet: str
    K: int

    # Training Parameters
    gpu: int
    experiment: str
    model_type: str
    val_max_batchs: int  # 节约时间

    # Data Paths
    train_data: Union[str, List[str]]
    val_data: Union[str, List[str]]
    pre_model: str

    # Optimization Parameters
    learning_rate: float
    epochs: int

    # Image Parameters
    image_width: int
    image_height: int
    num_channels: int

    # Batch Parameters
    batch_size: int
    num_workers: int

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PlateConfig":
        """从YAML文件加载配置"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return PlateConfig(**config_dict)

    def save(self, yaml_path: str) -> None:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, allow_unicode=True)


if __name__ == "__main__":
    config = PlateConfig.from_yaml("config.yaml")

    print(f"Image size: {config.image_width}x{config.image_height}")
    print(f"Batch size: {config.batch_size}")

    config.batch_size = 128
    config.save("new_config.yaml")
