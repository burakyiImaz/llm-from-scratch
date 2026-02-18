from transformers import AutoConfig, AutoModelForCausalLM
from .config import Config
from .hf_model import CasualLM

AutoConfig.register("turkce_llm", Config)
AutoModelForCausalLM.register(Config, CasualLM)
