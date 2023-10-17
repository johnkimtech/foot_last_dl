import argparse
import gradio as gr
from pathlib import Path
from inference import inference
from collections import namedtuple
from gui_utils import render_stl, make_csv_infer


def predict(model, input_file, foot):
    if model and input_file and foot:
        Setting = namedtuple(
            "Setting",
            ("exp_name", "infer_data_csv", "device", "batch_size", "output_csv_path"),
        )
        args = Setting(
            exp_name=model,
            infer_data_csv=make_csv_infer(input_file.name, foot),
            device="cpu",
            batch_size=1,
            output_csv_path=None,
        )
        result_df = inference(args).iloc[:, 1:]
        render_img = render_stl(input_file)
        return render_img, result_df
    else:
        return None, None


# Define the Gradio interface
with gr.Blocks(title="Foot Parameter Regression", live=False) as demo:
    # inputs
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Radio(
                label="Model",
                choices=["attn_ln", "attn_ln_oct_16"],
                value="attn_ln_oct_16",
                visible=False,
            )
            input_file = gr.File(label="Upload STL File of a Foot")
            foot = gr.Radio(
                label="Left/Right Foot?", choices=["Left", "Right"], value="Left"
            )

            btn_submit = gr.Button("Predict")

        # outputs
        with gr.Column(scale=2):
            img = gr.Image(label="3D Render of the STL input file")
            result = gr.DataFrame(label="Estimated foot parameters")

    # event handler
    btn_submit.click(predict, inputs=[model, input_file, foot], outputs=[img, result])


def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here.")

    # Define command line arguments
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.launch(server_name=args.ip, server_port=args.port, share=args.public)
