from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

app = Flask(__name__)

# Load the GPT-Neo 2.7B model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to("cuda")

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data['prompt']
    max_length = data.get('max_length', 50)  # Adjust as needed

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda")
    output_ids = model.generate(input_ids, max_length=max_length)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
