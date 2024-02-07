import gradio as gr
import requests


input_component = gr.inputs.File(label="Upload Image")

# Function to make a prediction using the FastAPI endpoint
def predict_image(image):
    url = 'http://127.0.0.1:8000/predict'
    files = {'image': image}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return "Failed to get prediction"


output_component = gr.outputs.Textbox(label="Prediction")

gr.Interface(fn=predict_image, inputs=input_component, outputs=output_component, title="Watermark Classifier").launch()
