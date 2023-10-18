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
        args = Setting(
            exp_name=model,
            infer_data_csv=make_csv_infer(input_file.name, foot),
            device="cpu",
            batch_size=32,
            output_csv_path=None,
            result_headers=["No.", "발 길이", "발폭", "발볼 높이", "앞코 높이", "힐 높이"],
        )
        result_df = inference(args).iloc[:, 1:]
        last_params_np = result_df.to_numpy().astype(float).squeeze()
        matched_last = find_last(last_params_np, LAST_DB_CSV)
        last_pc_path = matched_last["3D"]
        return result_df, last_pc_path
    else:
        return None, None


def foot_show(file):
    if file:
        return file.name
    else:
        return None


# Define the Gradio interface
with gr.Blocks(title="Last Parameter Matching", live=False) as demo:
    # inputs
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# Input")
            model = gr.Radio(
                label="Model",
                choices=["attn_ln_oct_18_no_layernorms"],
                value="attn_ln_oct_18_no_layernorms",
                visible=False,
            )
            input_file = gr.File(label="Upload Foot STL File")
            foot = gr.Radio(
                label="Left/Right Foot?", choices=["Left", "Right"], value="Left"
            )

            btn_submit = gr.Button("Predict")

            foot_render = gr.Model3D(label="Preview of Foot STL")
        # outputs
        with gr.Column(scale=2):
            gr.Markdown("# Output")
            result = gr.DataFrame(label="Estimated last parameters")
            last_render = gr.Model3D(label="3D Render of the maching Last")

    # event handler
    input_file.change(foot_show, input_file, foot_render)
    btn_submit.click(
        predict, inputs=[model, input_file, foot], outputs=[result, last_render]
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
