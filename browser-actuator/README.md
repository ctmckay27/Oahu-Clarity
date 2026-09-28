# Mari404 Browser Actuator

Temporary remote Chromium service for operating the existing Mari404ever ChatGPT Site when native Sites actions are not exposed in the current conversation.

The control API is deliberately scoped to:
- https://chatgpt.com
- https://mari404ever.carlitoesblanco.chatgpt.site

It exposes navigation, snapshots, clicks, key presses, screenshots, and browser close. It does not expose an API for entering passwords or arbitrary JavaScript. Authentication to ChatGPT happens only through the noVNC viewer.

Required environment secrets:
- ACTUATOR_TOKEN
- VNC_PASSWORD

Existing Site only:
- Project: appgprj_6aa06a5c8a608191a2d5cf5f9ef5543a
- URL: https://mari404ever.carlitoesblanco.chatgpt.site

Delete or suspend the service and rotate secrets after the publication task. Do not create a replacement Site.
