import json
from pathlib import Path
from typing import Any


HISTORY_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "training_history.json"
)


def save_training_record(record: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = {"history": []}

    if not isinstance(data, dict) or not isinstance(data.get("history"), list):
        raise ValueError(
            'История обучений должна иметь формат {"history": []}'
        )

    data["history"].append(record)

    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

def load_training_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError("Файл истории обучений не найден")

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Некорректный формат истории обучений")

    history = data.get("history")

    if not isinstance(history, list):
        raise ValueError("Поле history должно быть списком")

    return history