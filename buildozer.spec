[app]
title = InventoryAI
package.name = inventoryai
package.domain = org.example
source.dir = .
source.include_exts = py,kv,png,jpg,txt,tflite
requirements = python3,kivy,kivymd,numpy,opencv-python-headless,tflite-runtime
orientation = portrait
android.permissions = CAMERA
android.api = 33
android.ndk = 25b
fullscreen = 0
log_level = 2

[buildozer]
log_level = 2
warn_on_root = 1