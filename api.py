from flask import Flask, request, jsonify
import torch
import time
from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel
app = Flask(__name__)

# Load the GPT-Neo 2.7B model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to("cuda")


# Load the GPT-2 model and tokenizer
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
#model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to("cuda")


@app.route('/generate', methods=['POST'])
def generate_text():
    start_time = time.time()


    print("CUDA is available.")

    data = request.json
    prompt = data['prompt']
    max_length = data.get('max_length', 50)  # You might adjust the default as needed

    elapsed_time = time.time() - start_time

    # Encode the prompt to input_ids
    print("prompt recieved", elapsed_time)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda")
    
    # Set up the generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "num_return_sequences": 1,
        "temperature": 0.5,
        "top_k": 120,
        "top_p": 0.5,
        "do_sample": False,
        "use_cache": True,
        "attention_mask": torch.ones_like(input_ids).to(input_ids.device),  # Set attention mask
        "pad_token_id": tokenizer.pad_token_id,  # Set pad token ID
    }

    print("gen_kwargs: ", gen_kwargs)

    elapsed_time = time.time() - start_time
    print("input_ids: ", elapsed_time)

    # Generate the output
    with torch.no_grad():  # Disable gradient calculations for generation
        output_ids = model.generate(input_ids, **gen_kwargs)

    elapsed_time = time.time() - start_time
    print("output_ids: ", elapsed_time)

    # Decode the generated ids to a text string
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    elapsed_time = time.time() - start_time
    print("generated: ", elapsed_time)

    return jsonify({'generated_text': generated_text})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
