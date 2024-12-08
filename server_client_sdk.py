import requests

class ChatClient:
    def __init__(self, server_url):
        """
        Initializes the ChatClient with the server URL.
        """
        self.server_url = server_url

    def chat_completions_create(self, messages, model='', temperature=0.7, top_p=0.9, max_new_tokens=50):
        prompt = self._build_prompt(messages)
        response = requests.post(
            f"{self.server_url}/predict",
            json={
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")
        
        return response.json()

    def _build_prompt(self, messages):
        """
        Constructs the input prompt from the list of messages using the server's chat template logic.
        
        Args:
            messages (list): List of messages in the format [{"role": "user", "content": "text"}, ...].
        
        Returns:
            str: Constructed prompt.
        """
        # Request the server to apply the chat template
        response = requests.post(
            f"{self.server_url}/apply_chat_template",
            json={"messages": messages}
        )
        
        if response.status_code != 200:
            raise Exception(f"Chat template application failed with status {response.status_code}: {response.text}")
        return response.json()["text"]

