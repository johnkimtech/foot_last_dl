# Demo application: Foot 3D -> Last parameters (5)
import argparse
import gradio as gr
from pathlib import Path
from inference import inference
from collections import namedtuple
from gui_utils import render_3d, make_csv_infer
from foot_last_utils import find_last

LAST_DB_CSV = "data/3D_All_Foot/last_db.csv"


def predict(model, input_file, foot):
    if model and input_file and foot:
        Setting = namedtuple(
            "Setting",
            (
                "exp_name",
                "infer_data_csv",
                "device",
                "batch_size",
                "output_csv_path",
                "result_headers",
            ),
        )
        yield input_file.name, None, None
        args = Setting(
            exp_name=model,
            infer_data_csv=make_csv_infer(input_file.name, foot),
            device="cpu",
            batch_size=1,
            output_csv_path=None,
            result_headers=["No.", "발 길이", "발폭", "발볼 높이", "앞코 높이", "힐 높이"],
        )
        result_df = inference(args).iloc[:, 1:]
        yield input_file.name, result_df, None
        last_params_np = result_df.to_numpy().astype(float).squeeze()
        matched_last = find_last(last_params_np, LAST_DB_CSV)
        last_pc_path = matched_last["3D"]
        yield input_file.name, result_df, last_pc_path
        # return input_file.name, result_df, last_pc_path
    else:
        return None, None


# Define the Gradio interface
with gr.Blocks(title="Foot Parameter Regression", live=False) as demo:
    # inputs
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Radio(
                label="Model",
                choices=["attn_ln_oct_17"],
                value="attn_ln_oct_17",
                visible=False,
            )
            input_file = gr.File(label="Upload STL File of a Foot")
            foot = gr.Radio(
                label="Left/Right Foot?", choices=["Left", "Right"], value="Left"
            )

            btn_submit = gr.Button("Predict")

            img = gr.Model3D(label="3D Render of the STL input file")
        # outputs
        with gr.Column(scale=2):
            result = gr.DataFrame(label="Estimated foot parameters")
            # last_img = gr.Image(label="3D Render of the matched last")
            last_img = gr.Model3D(label="3D Render of the matched last")

    # event handler
    btn_submit.click(
        predict, inputs=[model, input_file, foot], outputs=[img, result, last_img]
    )


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
    demo.queue()
    demo.launch(server_name=args.ip, server_port=args.port, share=args.public)
