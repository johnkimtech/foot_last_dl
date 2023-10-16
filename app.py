import gradio as gr
from pathlib import Path
from inference import inference
from collections import namedtuple
from gui_utils import render_stl, make_csv_infer

def predict(file, foot):
    if file and foot:
        Setting = namedtuple(
            "Setting", ("exp_name", "infer_data_csv", "device", "batch_size", "output_csv_path")
        )
        args = Setting(
            exp_name="attn_ln",
            infer_data_csv=make_csv_infer(file.name, foot),
            device="cuda",
            batch_size=1,
            output_csv_path=None
        )
        result_df = inference(args)
        return render_stl(file), result_df.iloc[:, 2:]
    else:
        return None, None


# Define the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Upload STL File of a Foot"),
        gr.Radio(label="Left/Right Foot?", choices=["Left", "Right"], value="Left"),
    ],
    outputs=[
        gr.Image(label="3D Render of the STL input file"),
        gr.DataFrame(label="Estimated foot parameters"),
    ],  # Display the rendered image
    live=True,
)

demo.launch(server_name="0.0.0.0", server_port=7860)
