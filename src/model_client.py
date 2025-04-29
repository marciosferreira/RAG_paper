import requests

class MLModelAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def check_status(self):
        url = f"{self.base_url}/"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_embedding(self, text):
        url = f"{self.base_url}/embedding"
        payload = {"text": text}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def llm_generate(self, text):
        url = f"{self.base_url}/generate"
        payload = {"text": text}
        headers={"Content-Type": "application/json"}
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}


if __name__ == "__main__":
    client = MLModelAPIClient()
    print("Status:", client.check_status())

    prompt = [
            {
                "role": "user",
                "content": f"""hi rag"""
            }
        ]

    print('\n\nanswer', client.llm_generate(prompt))