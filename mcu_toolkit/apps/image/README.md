Image-classification main.cpp templates, one per board.

Rendered from NuML_TFLM_Tool/imgclass_codegen/main_cpp_tmpl.jinja2 and patched by the
NuML App Builder v0.1.8 runtime_preflight / usb_cdc_runtime helpers, then centre-cropped
to match Studio training. Regenerate with
`scripts/vendor_mcu_toolkit.py image-templates`; never edit by hand.

Token: @@ARENA@@ -- the Vela tensor arena size in bytes.
