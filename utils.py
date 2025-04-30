
import json
import yaml
import datetime


def read_json(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def write_json(filepath: str, data: dict) -> None:
    with open(filepath, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file)


def read_yaml(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def read_lines(filepath: str) -> list[str]:
    with open(filepath, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()
    data = [line.replace('\n', '') for line in lines]
    return data


def write_lines(filepath: str, lines: list[str]):
    lines_sep = [f'{line}\n' for line in lines]
    with open(filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.writelines(lines_sep)


def time_now() -> list[str]:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S").split('_')
