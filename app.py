from flask import Flask, render_template, request
from backend.text_model import detect_text
from backend.image_model import detect_image
from backend.utils import save_file
from PIL import Image

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


# 🔹 TEXT ROUTE
@app.route("/text", methods=["POST"])
def text():
    user_text = request.form.get("text")

    if not user_text:
        return render_template("index.html", text_result="Please enter some text")

    result = detect_text(user_text)

    return render_template(
        "index.html",
        text_result=result["result"],
        ai_prob=result["ai_probability"],
        perplexity=result["perplexity"]
    )


# 🔹 IMAGE ROUTE
@app.route("/image", methods=["POST"])
def image():
    file = request.files.get("file")

    # ❌ No file
    if not file or file.filename == "":
        return render_template("index.html", image_result="Please upload a file")

    try:
        # Validate image
        img = Image.open(file)
        img.verify()

        file.seek(0)
        path = save_file(file)

        result = detect_image(path)

        return render_template("index.html", image_result=result)

    except:
        return render_template("index.html", image_result="Invalid file. Please upload an image.")


if __name__ == "__main__":
    app.run(debug=True)