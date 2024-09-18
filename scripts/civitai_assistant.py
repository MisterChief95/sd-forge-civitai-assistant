import os

import gradio as gr

from civitai_assistant.type import ModelType
from civitai_assistant.update import update_metadata, update_preview_images

from modules import script_callbacks


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as civitai_assistant_view:
        with gr.Column(scale=1, min_width=768):
            with gr.Row():
                gr.HTML(
                    """
                    <center>
                        <h1>Civitai Assistant</h1>
                        <p>Use this assistant to update metadata and preview images for Civitai models.</p>
                    </center>
                    """)
            with gr.Row():
                model_checkboxes = gr.CheckboxGroup(
                    [member.value for member in ModelType],
                    label="Models",
                )
            with gr.Row():
                with gr.Group():
                    overwrite_checkbox = gr.Checkbox(
                        label="Overwite Existing Tags/Images",
                    )
                    recalculate_hash = gr.Checkbox(
                        label="Recalculate Hashes",
                    )

            with gr.Row():
                metadata_btn = gr.Button("Update Tags")
                preview_btn = gr.Button("Update Preview Image")

                # TODO: Implement the check for updates functionality
                gr.Button("Check For Updates", interactive=False)

                metadata_btn.click(fn=update_metadata, inputs=[model_checkboxes, overwrite_checkbox, recalculate_hash])
                preview_btn.click(
                    fn=update_preview_images, inputs=[model_checkboxes, overwrite_checkbox, recalculate_hash]
                )

        return [(civitai_assistant_view, "Civitai Assistant", "civitai_assistant_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
