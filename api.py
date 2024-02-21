from flask import Flask, request, jsonify
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

app = Flask(__name__)

# Load the GPT-Neo 2.7B model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to("cuda")

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data['prompt']
    max_length = data.get('max_length', 50)  # You might adjust the default as needed

    # Encode the prompt to input_ids
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda")

    # Set up the generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "num_return_sequences": 1,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": True,
        "use_cache": True,
        "attention_mask": torch.ones_like(input_ids).to(input_ids.device),  # Set attention mask
        "pad_token_id": tokenizer.pad_token_id,  # Set pad token ID
    }

    # Generate the output
    with torch.no_grad():  # Disable gradient calculations for generation
        output_ids = model.generate(input_ids, **gen_kwargs)

    # Decode the generated ids to a text string
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({'generated_text': generated_text})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
