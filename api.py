from flask import Flask, request, jsonify
from transformers import GPTJForCausalLM, GPT2Tokenizer

app = Flask(__name__)

# Load GPT-J model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data['prompt']
    max_length = data.get('max_length', 50)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=max_length)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
