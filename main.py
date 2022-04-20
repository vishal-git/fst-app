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
    predictions = [0] * len(images_arr)

    for i, _ in enumerate(predictions):
        model.set_tensor(
            input_details[0]["index"], images_arr[i].reshape((1, 150, 150, 3))
        )
        model.invoke()
        predictions[i] = model.get_tensor(output_details[0]["index"]).reshape((6,))

    prediction_probabilities = np.array(predictions)
    argmaxs = np.argmax(prediction_probabilities, axis=1)

    return argmaxs


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def resize(image):
    return cv2.resize(image, (150, 150))


@app.post("/uploadfiles/", response_class=HTMLResponse)
async def create_upload_files(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        f = await file.read()
        images.append(f)

    images = [np.frombuffer(img, np.uint8) for img in images]
    images = [cv2.imdecode(img, cv2.IMREAD_COLOR) for img in images]
    images_resized = [resize(img) for img in images]
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_resized]

    names = [file.filename for file in files]

    for image, name in zip(images_rgb, names):
        pillow_image = Image.fromarray(image)
        pillow_image.save("static/" + name)

    image_paths = ["static/" + name for name in names]

    images_arr = np.array(images_rgb, dtype=np.float32)

    class_indexes = model_predict(images_arr)

    class_predictions = [class_mapping[x] for x in class_indexes]

    table_html = get_html_table(image_paths, class_predictions, 1)

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

    content = content + get_html_table(full_original_paths, display_names, 0)

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


def get_html_table(image_paths, names, pred):
    image_size = 300 if pred else 180
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
