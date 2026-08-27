import os

# Must be set before gradio is imported anywhere. Gradio otherwise calls
# api.gradio.app for analytics and messaging during startup, which violates
# the "no network call at boot" constraint and delays every cold start.
# Blocks(analytics_enabled=False) alone does not cover the messaging call.
# setdefault so an operator can still turn it back on deliberately.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
