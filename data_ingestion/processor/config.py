import yaml

class Config:
    def __init__(self, config_path="../config.yaml"):
        # Abre e carrega o arquivo YAML
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def get(self, key):
        # Retorna o valor da chave no YAML, ou None se não existir
        return self.config.get(key, None)
