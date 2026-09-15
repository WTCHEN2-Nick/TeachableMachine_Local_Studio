USB CDC serial + UVC overlay sources rendered from NuML App Builder v0.1.8 (Apache-2.0).
uvc_composite/: CDC on interfaces 2/3 beside the UVC camera (image kind on NuGestureAI).
cdc_only/: CDC-only HSUSB device on interfaces 0/1 (audio kinds on NuGestureAI).
numl_cdc_retarget.c is patched so GCC's _write() reaches the CDC tee (see the file header).
