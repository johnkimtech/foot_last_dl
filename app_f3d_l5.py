# Demo application: Foot 3D -> Last parameters (5)
import argparse
import gradio as gr
import pandas as pd
import os
from inference import inference
from collections import namedtuple
from gui_utils import make_csv_infer
from foot_last_utils import find_last

LAST_DB_CSV = "data/3D_Oct30/clean_last_db.csv"
TABLE_HEADERS = ["No.", "라스트 길이", "라스트 폭", "라스트볼 높이", "앞코 높이", "힐 높이"]
CHECKPOINTS = ["attn_24_jan_normal_leather", "attn_24_jan_normal_textile",
               "mlp_24_jan_normal_leather", "mlp_24_jan_normal_textile"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Your program description here.")

    # Define command line arguments
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--multi", type=int, default=1
    )  # Run input multiple times and take the average
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


args = parse_args()


def postprocess_results(result):
    return result.mean(axis=0)


def preprocess_data(input_file):
    pass


def predict(model, input_file, foot):
    global args
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
        config = Setting(
            exp_name=model,
            infer_data_csv=make_csv_infer(
                input_file.name, foot, loop=args.multi),
            device=args.device,
            batch_size=args.batch_size,
            output_csv_path=None,
            result_headers=TABLE_HEADERS,
        )
        result_df = inference(config).iloc[:, 1:]
        last_params_np = result_df.to_numpy().astype(float).squeeze()
        last_params_np = postprocess_results(last_params_np)
        matched_last = find_last(
            last_params_np, LAST_DB_CSV, TABLE_HEADERS[1:], foot=foot
        )
        result_df = result_df.apply(pd.to_numeric).mean(axis=0).round(3)
        last_pc_path = matched_last["3D"]
        return result_df.to_frame().T, last_pc_path
    else:
        return None, None


def foot_show(file):
    if file:
        df = pd.read_csv('/home/hoang/Dataset/2024/foot_all.csv', index_col=-1)
        return file.name, df.loc[[os.path.basename(file.name)]]
    else:
        return None


# Define the Gradio interface
with gr.Blocks(title="Last Parameter Prediction", live=False) as demo:
    print("Using:", args.device)
    # inputs
    gr.Markdown("# Last Parameter Prediction")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            model = gr.Radio(
                label="Model",
                choices=CHECKPOINTS,
                value=CHECKPOINTS[0],
                visible=True,
            )
            input_file = gr.File(
                label="Upload Foot STL File", file_types=[".stl"])
            foot = gr.Radio(
                label="Left/Right Foot?", choices=["Left", "Right"], value="Left"
            )

            btn_submit = gr.Button("Predict")

            foot_render = gr.Model3D(label="Preview of Foot STL")

            foot_params_lookup = gr.DataFrame(
                label="Matched Foot parameters", headers=["발 길이", "발볼 둘레", "발등 둘레", "발 뒤꿈치 둘레", "발가락 둘레", "발폭"]
            )
        # outputs
        with gr.Column(scale=2):
            gr.Markdown("## Output")
            result = gr.DataFrame(
                label="Estimated last parameters", headers=TABLE_HEADERS[1:]
            )
            last_render = gr.Model3D(label="3D Render of the maching Last")

    # event handler
    input_file.change(foot_show, input_file, [foot_render, foot_params_lookup])
    btn_submit.click(
        predict, inputs=[model, input_file,
                         foot], outputs=[result, last_render]
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name=args.ip, server_port=args.port, share=args.public)
