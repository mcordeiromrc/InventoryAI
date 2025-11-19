import os
from ultralytics import YOLO, YOLOWorld

use_world = os.getenv("DETECT_WORLD", "0") == "1"

if use_world:
    modelo = YOLOWorld("yolov8s-world.pt")

    def carregar_classes(caminho: str) -> list[str]:
        with open(caminho, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    office_classes_pt = carregar_classes("classes.txt")
    classes_en_path = "classes_en.txt"
    if os.path.exists(classes_en_path):
        office_classes_en = carregar_classes(classes_en_path)
        modelo.set_classes(office_classes_en)
        modelo.model.names = office_classes_pt[: len(office_classes_en)]
    else:
        modelo.set_classes(office_classes_pt)

    results = modelo.predict(
        source=0,
        show=True,
        save=False,
        conf=0.3,
        line_width=2,
        imgsz=640
    )
else:
    modelo = YOLO("yolov8m.pt")
    names = list(modelo.model.names)
    names[62] = "monitor"
    names[63] = "notebook"
    names[64] = "mouse"
    names[66] = "teclado"
    names[67] = "celular"
    names[56] = "cadeira"
    names[73] = "livro"
    modelo.model.names = names
    office_indices = [62, 63, 64, 66, 67, 56, 73]
    results = modelo.predict(
        source=0,
        show=True,
        save=False,
        conf=0.4,
        line_width=2,
        imgsz=640,
        classes=office_indices
    )