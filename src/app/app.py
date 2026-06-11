import gradio as gr
import pandas as pd

def predict(age, tenure, monthly_charges):
    # dummy example, replace with your model inference
    data = pd.DataFrame([[age, tenure, monthly_charges]],
                        columns=["age", "tenure", "monthly_charges"])
    # model = load_model()  # you would load your trained model here
    return f"Predicted churn risk for Age={age}: LOW"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Tenure"),
        gr.Number(label="Monthly Charges")
    ],
    outputs=gr.Textbox(label="Prediction")
)

if __name__ == "__main__":
    demo.launch()
