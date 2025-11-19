import os
import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.fitimage import FitImage
from kivy.uix.widget import Widget
from kivy.properties import StringProperty
try:
    from kivy.graphics.svg import Svg as _SvgInstruction
except Exception:
    _SvgInstruction = None
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
try:
    from tflite_runtime.interpreter import Interpreter as _TFLInterpreter
    Interpreter = _TFLInterpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter as _TFInterpreter
        Interpreter = _TFInterpreter
    except Exception:
        Interpreter = None

class Detector:
    def __init__(self):
        self.use_world = os.getenv("DETECT_WORLD", "0") == "1"
        self.use_tflite = os.getenv("DETECT_TFLITE", "0") == "1" and Interpreter is not None
        self.classes = [62, 63, 64, 66, 67, 56, 73]
        if self.use_tflite:
            self.names = [
                'pessoa','bicicleta','carro','moto','avião','ônibus','trem','caminhão','barco','semáforo','hidrante',
                'placa de pare','parquímetro','banco','pássaro','gato','cachorro','cavalo','ovelha','vaca','elefante',
                'urso','zebra','girafa','mochila','guarda-chuva','bolsa','gravata','mala','frisbee','esqui','snowboard',
                'bola','pipa','taco de beisebol','luva de beisebol','skate','prancha de surfe','raquete de tênis',
                'garrafa','taça de vinho','copo','garfo','faca','colher','tigela','banana','maçã','sanduíche','laranja',
                'brócolis','cenoura','cachorro-quente','pizza','rosquinha','bolo','cadeira','sofá','planta em vaso','cama',
                'mesa de jantar','vaso sanitário','monitor','notebook','mouse','controle remoto','teclado','celular',
                'micro-ondas','forno','torradeira','pia','geladeira','livro','relógio','vaso','tesoura','ursinho de pelúcia',
                'secador de cabelo','escova de dentes'
            ]
            self.tflite_path = os.getenv("TFLITE_MODEL", "models/yolov8m_int8.tflite")
            if not os.path.exists(self.tflite_path):
                self.use_tflite = False
            else:
                self.interpreter = Interpreter(model_path=self.tflite_path)
                self.interpreter.allocate_tensors()
                self.input_det = self.interpreter.get_input_details()[0]
                self.output_det = self.interpreter.get_output_details()
        elif self.use_world:
            self.model = YOLOWorld("yolov8s-world.pt")
            self.classes_pt = self._load_classes("classes.txt")
            classes_en_path = "classes_en.txt"
            if os.path.exists(classes_en_path):
                classes_en = self._load_classes(classes_en_path)
                self.model.set_classes(classes_en)
                self.model.model.names = self.classes_pt[: len(classes_en)]
            else:
                self.model.set_classes(self.classes_pt)
        else:
            self.model = YOLO("yolov8m.pt")
            names = list(self.model.model.names)
            names[62] = "monitor"
            names[63] = "notebook"
            names[64] = "mouse"
            names[66] = "teclado"
            names[67] = "celular"
            names[56] = "cadeira"
            names[73] = "livro"
            self.model.model.names = names

    def _load_classes(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    def infer(self, frame):
        if self.use_tflite:
            h, w, _ = frame.shape
            img = cv2.resize(frame, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            self.interpreter.set_tensor(self.input_det[0]["index"], img)
            self.interpreter.invoke()
            outs = [self.interpreter.get_tensor(o["index"]) for o in self.output_det]
            boxes, scores, classes, count = None, None, None, None
            if len(outs) >= 4:
                boxes, classes, scores, count = outs[0], outs[1], outs[2], int(outs[3][0])
            else:
                y = outs[0]
                if y.ndim == 3 and y.shape[-1] >= 6:
                    count = y.shape[1]
                    boxes = y[0, :, 0:4]
                    scores = y[0, :, 4]
                    classes = y[0, :, 5]
            annotated = frame.copy()
            if boxes is not None and scores is not None and classes is not None:
                for i in range(count):
                    cls = int(classes[i]) if isinstance(classes, np.ndarray) else int(classes)
                    if cls not in self.classes:
                        continue
                    conf = float(scores[i])
                    if conf < 0.35:
                        continue
                    x1, y1, x2, y2 = boxes[i]
                    x1 = int(x1 / 640 * w) if x1 <= 1 else int(x1)
                    y1 = int(y1 / 640 * h) if y1 <= 1 else int(y1)
                    x2 = int(x2 / 640 * w) if x2 <= 1 else int(x2)
                    y2 = int(y2 / 640 * h) if y2 <= 1 else int(y2)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.names[cls]} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return annotated
        if self.use_world:
            r = self.model.predict(source=frame, conf=0.3, imgsz=640, show=False)
        else:
            r = self.model.predict(source=frame, conf=0.4, imgsz=640, show=False, classes=self.classes)
        return r[0].plot()

class SvgIcon(Widget):
    source = StringProperty("")

    def on_kv_post(self, base_widget):
        if not self.source or _SvgInstruction is None:
            return
        self.canvas.clear()
        with self.canvas:
            svg = _SvgInstruction(self.source)
        try:
            self.size = svg.width, svg.height
        except Exception:
            pass

class InventoryMDApp(MDApp):
    def build(self):
        Window.size = (360, 640)
        self.theme_cls.primary_palette = "Red"
        self.theme_cls.theme_style = "Light"
        self.collecting = False
        self.current_cam = int(os.getenv("REAR_CAM_INDEX", "0"))
        self.front_cam = int(os.getenv("FRONT_CAM_INDEX", "1"))
        self.detector = Detector()
        return Builder.load_file("inventory.kv")

    def define_posto(self):
        def _confirm(*_):
            name = tf.text.strip()
            if name:
                self.posto_name = name
                dialog.dismiss()
                self.start_collect()
        tf = MDTextField(hint_text="Nome do POSTO", helper_text="Digite o nome do imóvel", helper_text_mode="on_focus")
        dialog = MDDialog(title="Definir Novo Posto", type="custom", content_cls=tf, buttons=[MDRaisedButton(text="Confirmar", on_release=_confirm)])
        dialog.open()

    def start_collect(self):
        if self.collecting:
            return
        self.collecting = True
        self.root.current = "capture"
        self.open_camera(self.current_cam)
        Clock.schedule_interval(self.update, 0)

    def login_govbr(self):
        pass

    def open_camera(self, index):
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(index)

    def toggle_camera(self):
        if not self.collecting:
            return
        self.current_cam = self.front_cam if self.current_cam != self.front_cam else int(os.getenv("REAR_CAM_INDEX", "0"))
        self.open_camera(self.current_cam)

    def stop_collect(self):
        if not self.collecting:
            return
        self.collecting = False
        Clock.unschedule(self.update)
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        self.root.current = "home"

    def update(self, dt):
        if not hasattr(self, "cap") or not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated = self.detector.infer(frame)
        buf = annotated.tobytes()
        tex = Texture.create(size=(annotated.shape[1], annotated.shape[0]), colorfmt="rgb")
        tex.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
        tex.flip_vertical()
        self.root.get_screen("capture").ids.img_view.texture = tex

    def on_stop(self):
        if hasattr(self, "cap") and self.cap:
            self.cap.release()

if __name__ == "__main__":
        InventoryMDApp().run()