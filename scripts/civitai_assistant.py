import gradio as gr

from civitai_assistant.types import ModelType, UpdateOptions, UpdateType
from civitai_assistant.update import update_models
from civitai_assistant.ui import create_progressable_button

from modules import script_callbacks
from modules import shared


MODEL_CHOICES = [member.value for member in ModelType.__members__.values()]
UPDATE_TYPE_CHOICES = [member.value for member in UpdateType.__members__.values()]
UPDATE_OPTION_CHOICES = [member.value for member in UpdateOptions.__members__.values()]


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as civitai_assistant_view:
        with gr.Column(scale=1, min_width=768):
            with gr.Row():
                gr.HTML("<center><h1>Civitai Assistant</h1></center>")
            with gr.Row():
                model_checkboxes = gr.CheckboxGroup(
                    choices=MODEL_CHOICES, label="Models"
                )
            with gr.Row():
                update_types = gr.CheckboxGroup(
                    choices=UPDATE_TYPE_CHOICES, label="Update Types"
                )
            with gr.Row():
                options_group = gr.CheckboxGroup(
                    choices=UPDATE_OPTION_CHOICES, label="Options"
                )

            inputs = [
                model_checkboxes,
                update_types,
                options_group,
            ]

            def update_handler(
                model_types,
                update_selections,
                options,
                pr: gr.Progress = gr.Progress(),  # noqa: B008
            ):
                update_models(
                    model_types=model_types,
                    update_types=update_selections,
                    options=options,
                    pr=pr,
                )

            with gr.Row():
                create_progressable_button("Update", update_handler, inputs=inputs)

        return [(civitai_assistant_view, "Civitai Assistant", "civitai_assistant_tab")]


def on_ui_settings():
    CIVITAI_ASSISTANT_SECTION = ("civitai_assistant", "CivitAI Assistant")

    ca_options = {
        "ca_use_html_descriptions": shared.OptionInfo(
            False, "Save HTML Descriptions"
        ).info("Save HTML descriptions for models."),
        "ca_api_key": shared.OptionInfo("", "CivitAI API Key").info(
            "Used for downloading models from CivitAI."
        ),
        "ca_max_scan_depth": shared.OptionInfo(
            4,
            "Max Scan Depth",
            gr.Slider,
            {"minimum": 1, "maximum": 10, "step": 1},
        ).info("Maximum subdirectory depth to scan when searching for model files."),
        "ca_follow_symlinks": shared.OptionInfo(
            False, "Follow Symlinks"
        ).info("Follow symbolic links when scanning model directories."),
    }

    # Add normal settings
    for key, opt in ca_options.items():
        opt.section = CIVITAI_ASSISTANT_SECTION
        shared.opts.add_option(key, opt)


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
