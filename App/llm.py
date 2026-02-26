from codecarbon import EmissionsTracker
import requests
import time
from typing import Dict, Any    
from .utils import safe_token_count

class LLMClient:
    def __init__(self, model, temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.endpoint = "http://localhost:1234/v1/chat/completions"

    def classify_batch(self, prompt: str):
        tracker = EmissionsTracker(
            project_name="mistral-7b-instruct",
            measure_power_secs=1,
            save_to_file=False,
            log_level="error"
        )
        tracker.start()

        start_t = time.time()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature
        }

        tokens_in = safe_token_count(prompt, self.model)
        raw_response = ""

        try:
            resp = requests.post(self.endpoint, json=payload, timeout=300)
            data = resp.json()
            raw_response = data["choices"][0]["message"]["content"]
        except Exception as e:
            raw_response = f"Exception: {e}"

        emissions_kg = tracker.stop()
        latency = time.time() - start_t

        try:
            energy_kwh = tracker.final_emissions_data.energy_consumed
            power_w = (energy_kwh * 3600000) / latency if latency > 0 else 0.0
        except:
            energy_kwh = 0.0
            power_w = 0.0

        return {
            "raw": raw_response,
            "latency_s": latency,
            "energy_kwh": energy_kwh,
            "carbon_kg": emissions_kg,
            "avg_power_watts": power_w,
            "input_tokens": tokens_in
        }