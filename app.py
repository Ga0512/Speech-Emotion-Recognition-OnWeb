from flask import Flask, render_template, request, redirect, url_for
import os
from SER import predict

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio_file" in request.files:
            audio_file = request.files["audio_file"]
            if audio_file.filename != "":
                audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_file.filename)
                audio_path = audio_path.replace("\\", "/")
                audio_file.save(audio_path)
                print(audio_path)

                emotion = predict(str(audio_path))

                return render_template("index.html", audio_path=audio_path, emotion=emotion)
    return render_template("index.html", audio_path=None, emotion=None)

@app.route("/clear")
def clear():
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
