import yaml


class Config(object):
    def __init__(self, path):
        with path.open() as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        rel_size = self._config["scene"]["rel_size"]
        max_width = self._config["hand"]["max_width"]
        self._config["scene"]["size"] = rel_size * max_width

    def __getitem__(self, key):
        return self._config[key]
