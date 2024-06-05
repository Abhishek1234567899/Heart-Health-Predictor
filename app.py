    
from src.Hearthealthpredictor.pipelines.prediction_pipeline import CustomData,PredictPipeline
from flask import Flask,request,render_template,jsonify

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data = CustomData(
            age=int(request.form.get("age")),
            sex=int(request.form.get("sex")),
            cp=int(request.form.get("cp")),
            trestbps=int(request.form.get("trestbps")),
            chol=int(request.form.get("chol")),
            fbs=int(request.form.get("fbs")),
            restecg=int(request.form.get("restecg")),
            thalach=int(request.form.get("thalach")),
            exang=int(request.form.get("exang")),
            oldpeak=float(request.form.get("oldpeak")),
            slope=int(request.form.get("slope")),
            ca=int(request.form.get("ca")),
            thal=int(request.form.get("thal"))
        )
        final_data = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()

        pred = predict_pipeline.predict(final_data)

        result = round(pred[0], 2)

        return render_template("result.html", final_result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
