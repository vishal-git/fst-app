from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite


model = tflite.Interpreter("static/model.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

class_mapping = {
    0: "Building",
    1: "Forest",
    2: "Glacier",
    3: "Mountain",
    4: "Sea",
    5: "Street",
}


def model_predict(images_arr):
    model.set_tensor(
        input_details[0]["index"], images_arr.reshape((1, 150, 150, 3))
    )
    model.invoke()
    preds = model.get_tensor(output_details[0]["index"]).reshape((6,))

    pred = np.argmax(preds)
    max_phat = max(preds)

    return pred, max_phat


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def resize(image):
    return cv2.resize(image, (150, 150))


@app.post("/uploadfiles/", response_class=HTMLResponse)
async def create_upload_files(files: UploadFile = File(...)):
    '''
    images = []
    for file in files:
        f = await file.read()
        images.append(f)
    '''
    image = await files.read()

    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_resized = resize(image)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    name = files.filename

    pillow_image = Image.fromarray(image_rgb)
    pillow_image.save("static/" + name)

    image_path = "static/" + name

    images_arr = np.array(image_rgb, dtype=np.float32)

    pred, proba = model_predict(images_arr)
    pred_class = class_mapping[pred]

    table_html = get_html_results_table(image_path, pred_class)

    content = (
        head_html
        + """
    <marquee width="525" behavior="alternate"><h1 style="color:red;font-family:monospace">Here's Our Predictions!</h1></marquee>
    """
        + str(table_html)
        + """<br><form method="post" action="/">
    <button style="background-color: #bbb" type="submit">Home</button>
    </form>"""
    )

    return content


@app.post("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def main():
    content = (
        head_html + """<h1 style="color:#0AC663; font-family:monospace; font-weight: extra-bold; font-stretch: extra-expanded; text-align:center">IMAGE CLASSIFICATION</h1>"""
        + """
    <br><br><h3 style="color: white; font-family:monospace">Upload a picture to see which one of the following categories it belongs to:</h3><br>
    """
    )

    original_paths = [
        "building.jpg",
        "forest.jpg",
        "glacier.jpg",
        "mountain.jpg",
        "sea.jpg",
        "street.jpg",
    ]

    full_original_paths = ["static/original/" + x for x in original_paths]

    display_names = ["Building", "Forest", "Glacier", "Mountain", "Sea", "Street"]

    content = content + get_html_table(full_original_paths, display_names)

    content = (
        content
        + """
    <br/>
    <br/>
    <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
    <input style="background-color:#900C3F; padding: .5em; -moz-border-radium: 5px; -webkit-border-radius: 5px; border-radius: 6px; color: #EDEAE6; font-size: 18px; text-decoration: none; border: none;" name="files" type="file">
    <input style="background-color:#900C3F; padding: .5em; -moz-border-radium: 5px; -webkit-border-radius: 5px; border-radius: 6px; color: #EDEAE6; font-size: 20px; text-decoration: none; border: none;" type="submit">
    </form>
    </body>
    """
    )

    return content


# overall page design
head_html = """
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="background-color:#222;">
<center>
"""


def get_html_table(image_paths, names):
    image_size = 180
    tbl = '<table align="center">'
    for i, name in enumerate(names):
        if i == 0:
            tbl += "<tr>"
        tbl += (
            '<td style="color:#0AC663; font-size: 18px; font-family:monospace; font-weight: bold; font-stretch: extra-expanded; text-align:center">'
            + name.upper()
            + "</td>")
    tbl += "</tr>"

    for i, image_path in enumerate(image_paths):
        if i == 0:
            tbl += "<tr>"
        tbl += f'<td><img height="{image_size}" src="/' + image_path + '" ></td>'
    tbl += "</tr></table>"

    return tbl

def get_html_results_table(image_path, name):
    image_size = 300
    tbl = '<table align="center">'
    tbl += "<tr>"
    tbl += (
            '<td style="color:#0AC663; font-size: 18px; font-family:monospace; font-weight: bold; font-stretch: extra-expanded; text-align:center">'
            + name.upper()
            + "</td>")
    tbl += "</tr>"

    tbl += "<tr>"
    tbl += f'<td><img height="{image_size}" src="/' + image_path + '" ></td>'
    tbl += "</tr></table>"

    return tbl
