import gradio as gr

from neutral.api import create_api
from neutral.config import get_settings
from neutral.ui import build_ui

settings = get_settings()

# API and UI share one process, one port, and one loaded model: a single
# container, a single thing to deploy.
app = create_api(settings)
app = gr.mount_gradio_app(app, build_ui(settings, app.state.get_classifier), path="/")
