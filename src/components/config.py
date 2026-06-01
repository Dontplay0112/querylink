import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from components.decorators import split_line
from typing import Optional, Literal, Union, Dict, Any, Annotated # 1. 引入 Annotated
from pydantic import BaseModel, Field, SecretStr, computed_field, ConfigDict, model_validator
import tomli_w
import tomllib

class MyBaseModel(BaseModel):
    def __str__(self):
        # 参考print_config函数的实现
        data = self.model_dump(mode='json', exclude_none=True)
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = {k: (v if not isinstance(v, SecretStr) else "****") for k, v in value.items()}
            elif isinstance(value, SecretStr):
                data[key] = "****"
        return tomli_w.dumps(data).strip()

# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------
class ExperimentConfig(MyBaseModel):
    task: str
    desc: Optional[str] = None
    repeat: int = Field(default=1, ge=1)
    date: str = ""
    
    base_data_path: Path = Path("../data")
    data_file_name: str
    @computed_field
    def data_path(self) -> Path:
        return self.base_data_path / self.task
    base_output_path: Path = Path("../outputs")
    @computed_field
    def output_path(self) -> Path:
        file_parts = self.data_file_name.split('.')[:-1]
        file_stem = '-'.join(file_parts) if file_parts else self.data_file_name
        return self.base_output_path / f"{self.task}-{file_stem}" / self.date

    max_workers: int = Field(default=10, ge=1)
    max_inflight: int = Field(default=10, ge=1)
    max_thread_workers: int = Field(default=16, ge=1)
    max_thread_inflight: int = Field(default=16, ge=1)

# -----------------------------------------------------------------------------
# Agent Configuration
# -----------------------------------------------------------------------------

class AgentConfigBase(MyBaseModel):
    agent: str

class QueryLinkAgentConfig(AgentConfigBase):
    agent: Literal["QueryLink",
                   "QueryLinkLocomo",
                   "QueryLinkAmem",
                   "QueryLinkAmemLocomo",
                   "QueryLinkMotivation"] = "QueryLink"
    
    llm_model_source: Literal["openai", "huggingface", "ollama"] = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_base_url: Optional[str] = Field(default=None)
    
    embed_model_source: Literal["openai", "huggingface", "ollama"] = "huggingface"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_base_url: Optional[str] = Field(default=None)
    
    max_retries: int = Field(default=5, ge=0)
    backoff_factor: float = Field(default=0.5, ge=0.0)
    top_k: int = Field(default=4, ge=1)
    window_size: int = Field(default=5, ge=1)
    
    memory_version: str = ""

class TamplateAgentConfig(AgentConfigBase):
    agent: Literal["Template"] = "Template"
    argsA: str = "valueA"
    argsB: int = 37

# -----------------------------------------------------------------------------
# Root Config
# -----------------------------------------------------------------------------
class MyConfig(MyBaseModel):
    exp: ExperimentConfig
    agent: Annotated[
        Union[QueryLinkAgentConfig, TamplateAgentConfig], 
        Field(discriminator="agent")
    ]
    other: Optional[Dict[str, Any]] = None
    
    # 计算好日期并输入给exp的date字段和agent的memory_version字段
    @model_validator(mode='after')
    def setup_dates_and_versions(self):
        if not self.exp.date:
            current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.exp.date = current_date
        if isinstance(self.agent, QueryLinkAgentConfig) and not self.agent.memory_version:
            self.agent.memory_version = self.exp.date
        return self

# -----------------------------------------------------------------------------
# Loading Logic
# -----------------------------------------------------------------------------
@split_line(title="Read Config")
def load_config(sys_argv) -> MyConfig:
    if len(sys_argv) >= 2:
        config_path = sys_argv[1]
    else:
        config_path = '../config/locomo.toml'
        # config_path = '../config/longmemevalS.toml'
    
    path = Path(config_path)
    if not path.exists():
        local_path = Path(path.name)
        if local_path.exists():
            path = local_path
        else:
            raise FileNotFoundError(f"Config file not found: {path.absolute()}")
    with open(path, "rb") as f:
        toml_data = tomllib.load(f)
        
    try:
        config = MyConfig(**toml_data)
        print(config)
        save_config(config, config.exp.output_path / "config.toml")
        return config
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        raise

def save_config(config: MyConfig, save_path: Union[str, Path]):
    path = Path(save_path)
    data = config.model_dump(mode='json', exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
    logging.info(f"Config saved to: {path.absolute()}")

# -----------------------------------------------------------------------------
# Test Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        config = load_config(sys.argv)
        # print_config(config)
        print(config)
        
        # 验证解析出来的类型
        print(f"\nLoaded Agent Type: {type(config.agent).__name__}")
        
        save_path = config.exp.output_path / "config_snapshot.toml"
        save_config(config, save_path)
    except Exception as e:
        pass # load_config 已经打印了错误