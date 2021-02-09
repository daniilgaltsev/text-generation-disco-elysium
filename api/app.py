from flask import Flask, request, jsonify
from text_generation.text_generator import TextGenerator


app = Flask(__name__)
generator = TextGenerator()


@app.route("/")
def index():
    """Provide simple health check route"""
    return "Text Generation"


@app.route("/generate", methods=["GET", "POST"])
def generate():
    """
    Provides generation API route. Responds to GET and POST requests.
    """

    prompts = _load_prompts()
    generated = generator.generate(prompts)
    json = {"generated": generated}

    return jsonify(json)


def _load_prompts():
    if request.method == "GET":
        return [""]
    if request.method == "POST":
        json = request.get_json()
        n_sequences = json.get("n_sequences", None)
        prompts = json.get("prompts", None)
        if prompts is None:
            if n_sequences is None:
                return "no prompts or n_sequences defined in query string"
            n_sequences = int(n_sequences)
            prompts = [""] * n_sequences
        return prompts


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8080, debug=False)


if __name__ == "__main__":
    main()