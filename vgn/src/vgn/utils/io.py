import yaml


def read_yaml(path):
    with path.open("r") as f:
        data = yaml.load(f)
    return data


def write_yaml(data, path):
    with path.open("w") as f:
        yaml.dump(data, f)


def create_csv(path, header):
    with path.open("wb") as f:
        f.write(header)
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("ab") as f:
        f.write(row)
        f.write("\n")
