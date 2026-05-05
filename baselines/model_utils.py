#-*- coding: utf-8 -*-
"""
Simplified model utility module, removed dependencies on torch and fastchat.
"""

class ApiConversation:
    """A simple conversation manager compatible with OpenAI API."""
    def __init__(self, system_message=None):
        self.messages = []
        if system_message:
            self.messages.append({"role": "system", "content": system_message})

    def append_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages

    def copy(self):
        new_conv = ApiConversation()
        new_conv.messages = self.messages.copy()
        return new_conv

def get_template(return_fschat_conv=False, **kwargs):
    """Return a simplified conversation manager for API models."""
    if return_fschat_conv:
        # Return a simplified version that mimics the fastchat conversation object
        return ApiConversation()
    else:
        # For non-chat templates, can return a simple dictionary or string
        return {"prompt": "{instruction}"}