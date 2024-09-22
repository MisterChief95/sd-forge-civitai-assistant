import gradio as gr

from civitai_assistant.type import ModelType
from civitai_assistant.update import update_metadata, update_preview_images
from civitai_assistant.ui import create_progressable_button

from modules import script_callbacks
from modules import shared


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as civitai_assistant_view:
        with gr.Column(scale=1, min_width=768):
            with gr.Row():
                gr.HTML("<center><h1>Civitai Assistant</h1></center>")
            with gr.Row():
                model_checkboxes = gr.CheckboxGroup([member.value for member in ModelType], label="Models")
            with gr.Row():
                with gr.Group():
                    overwrite_checkbox = gr.Checkbox("Overwite Existing Tags/Images")
                    recalculate_hash = gr.Checkbox("Recalculate Hashes")

            inputs = [model_checkboxes, overwrite_checkbox, recalculate_hash]
            with gr.Row():
                create_progressable_button("Update Tags", update_metadata, inputs=inputs)
            with gr.Row():
                create_progressable_button("Update Preview Images", update_preview_images, inputs=inputs)
            with gr.Row():
                # TODO: Implement the check for updates functionality
                gr.Button("Check For Updates", interactive=False)

        return [(civitai_assistant_view, "Civitai Assistant", "civitai_assistant_tab")]


def on_ui_settings():
    CIVITAI_ASSISTANT_SECTION = ("civitai_assistant", "CivitAI Assistant")

    ca_options = {
        "ca_use_html_descriptions": shared.OptionInfo(False, "Save HTML Descriptions").info(
            "Save HTML descriptions for models."
        ),
        "ca_api_key": shared.OptionInfo("", "CivitAI API Key").info("Used for downloading models from CivitAI."),
    }

    # Add normal settings
    for key, opt in ca_options.items():
        opt.section = CIVITAI_ASSISTANT_SECTION
        shared.opts.add_option(key, opt)


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
