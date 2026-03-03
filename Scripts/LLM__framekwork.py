# pip install torch transformers

import json
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

# API Keys:
CHATGPT = ''
CLAUDE  = ''
GEMINI  = ''

def ChatGPT(key='', model='', system='', prompt='', args={}) -> str:
	'''
	Send a prompt to OpenAI's ChatGPT LLM and return the model's response
	Official model list: https://platform.openai.com/docs/models
	# Parameters:
	-------------
	key    : str             OpenAI API authentication key
	model  : str             Model name. Example: 'gpt-4o-mini'
	system : str             The system wide instruction (personality)
	prompt : str             The input text prompt
	args = {
		'temperature':       1.0,    # Randomness [0.0-2.0]
		'top_p':             1.0,    # Nucleus sampling
		'n':                 1,      # Number of output responses
		'stop':              None,   # Stop the sequence
		'seed':              None,   # Deterministic sampling
		'user':              None,   # End-user ID tracking
		'stream':            False,  # Enable streaming
		'stream_options':    None,   # Streaming behavior
		'max_tokens':        None,   # Max output tokens
		'presence_penalty':  0,      # Penalize topic repetition
		'frequency_penalty': 0,      # Penalize token repetition
		'logit_bias':        None,   # Adjust token probabilities
		'response_format':   {},     # JSON mode / schema
		'tools':             None,   # Function/tool calling
		'tool_choice':       'auto'} # Force specific tool
	# Returns:
	----------
	str                      Generated text response from the model
	'''
	history = [{'role':'system', 'content':system}]
	history.append({'role': 'user', 'content':prompt})
	url = 'https://api.openai.com/v1/chat/completions'
	header = {'Authorization':f'Bearer {key}','Content-Type':'application/json'}
	payload = {'model':model, 'messages':history} | args
	response = requests.post(url, headers=header, json=payload)
	if response.status_code != 200:
		error = response.json()['error']['message']
		raise SystemError(error)
	else:
		text = response.json()['content'][0]['text']
		history.append({'role': 'assistant', 'content':text})
		return text

def Claude(key='', model='', system='', prompt='', args={}) -> str:
	'''
	Send a prompt to Anthropic's Claude LLM and return the model's response
	Official docs: https://docs.anthropic.com/claude/reference/messages_post
	# Parameters:
	-------------
	key    : str          Anthropic API key
	model  : str          Model name. Example: 'claude-3-opus-20240229'
	system : str          The system wide instruction (personality)
	prompt : str          The input text prompt
	args = {
		'temperature':    1.0,             # Randomness
		'top_p':          1.0,             # Nucleus sampling
		'top_k':          0,               # Limits token pool
		'stop_sequences': [],              # Custom stop strings
		'stream':         False,           # Enable streaming
		'metadata':       None,            # Arbitrary JSON metadata
		'max_tokens':     200,             # Max output tokens
		'tools':          [],              # Tool definitions
		'tool_choice':    {'type':'auto'}} # Tool calling behavior
	Note 1:               args = {'max_tokens':200} must always be included
	# Returns:
	----------
	str                   Generated text response from the model
	'''
	history = [{'role': 'user', 'content':prompt}]
	url = 'https://api.anthropic.com/v1/messages'
	header = {'x-api-key':key,
	'anthropic-version':'2023-06-01', 'content-type':'application/json'}
	payload = {'model':model, 'messages':history, 'system':system} | args
	response = requests.post(url, headers=header, json=payload)
	if response.status_code != 200:
		text1 = response.json()['error']['message']
		text2 = response.json()['error']['type']
		error = text1 + ' ' + text2
		raise SystemError(error)
	else:
		text = response.json()
		history.append({'role': 'assistant', 'content':text})
		return text

def Gemini(key='', model='', system='', prompt='', args={'stream':False})-> str:
	'''
	Send a prompt to Google's Gemini LLM and return the model's response
	Official model list: hhttps://ai.google.dev/models
	# Parameters:
	-------------
	key    : str                Google's API authentication key
	model  : str                Model name. Example: 'gemini-1.5-pro'
	system : str                The system wide instruction (personality)
	prompt : str                The input text prompt
	args = {
		'stream':               False, # Enable streaming
		'generationConfig':
			{'temperature':     1.0,   # Randomness
			'topP':             1.0,   # Nucleus sampling
			'topK':             0,     # Limits token pool
			'maxOutputTokens':  200,   # Max output tokens
			'stopSequences':    [],    # Custom stop strings
			'candidateCount':   1},    # Number of users
		'tools':                []}    # Tool definitions
	# Returns:
	----------
	str                         Generated text response from the model
	'''
	history = [{'parts':[{'text':prompt}]}]
	s1 = f'https://generativelanguage.googleapis.com/v1beta/models/'
	s2 = f'{model}:generateContent?key={key}'
	url = s1 + s2
	header = {'Content-Type':'application/json'}
	payload = {
		'systemInstruction':{'parts':[{'text':system}]},
		'contents':history}
	try:
		payload['generationConfig'] = args['generationConfig']
		payload['tools'] = args['tools']
	except:
		pass
	response = requests.post(url, headers=header, json=payload,
		stream=args['stream'])
	if response.status_code != 200:
		error = response.json()['error']['message']
		raise SystemError(error)
	else:
		text = response.json()['candidates'][0]['content']['parts'][0]['text']
		history.append({'parts':[{'text':text}]})
		return text

def LocalLLM(device='cpu', model='', system='', prompt='',
	mxtkn=200, temp=1.0, topp=1.0, outn=1, presp=1.0, seed=None) -> str:
	'''
	Send a prompt to a local HuggingFace LLM and return the model's response
	Official docs: https://huggingface.co/models
	# Parameters:
	-------------
	device : str       CPU for CPU or CUDA for GPU
	model  : str       Model name. Example: 'Microsoft/Phi-2'
	system : str       The system wide instruction (personality)
	prompt : str       The input text prompt
	args = {           Use as **args argument
		'temp':  1.0,  # Randomness [0.0-2.0]
		'mxtkn': 200,  # Max output tokens
		'topp':  1.0,  # Nucleus sampling
		'outn':  1,    # Number of output responses
		'presp': 1.0,  # Penalize topic repetition
		'seed':  None} # Deterministic sampling
	# Returns:
	str                Generated text response from the model
	----------
	'''
	# Add to class __init__ later
	if seed is not None:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	tokenizer = AutoTokenizer.from_pretrained(model)
	if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
	local_model = AutoModelForCausalLM.from_pretrained(
		model,
		dtype='auto',
		device_map=None if device == 'cpu' else 'auto')
	local_model.to(device)
	local_model.eval()
	is_encoder_decoder = getattr(local_model.config, 'is_encoder_decoder', False)
	#----------------------------
	history = [{'role':'system', 'content':system}]
	history.append({'role': 'user', 'content':prompt})
	input_text = ''
	if hasattr(tokenizer, 'apply_chat_template') and \
		tokenizer.chat_template is not None:
		try:
			input_text = tokenizer.apply_chat_template(
				prompt,
				tokenize=False,
				add_generation_prompt=True)
		except Exception:
			input_text = ''
	if input_text == '':
		try:
			for line in prompt:
				role = line.get('role', '')
				content = line.get('content', '')
				if role == 'system':
					input_text += content + '\n'
				else:
					input_text += f'{role}: {content}\n'
		except Exception:
			input_text = str(prompt)
	inputs = tokenizer(
		input_text,
		return_tensors='pt',
		padding=True,
		truncation=True)
	inputs = {k: v.to(device) for k, v in inputs.items()}
	if hasattr(local_model.config, 'max_position_embeddings'):
		max_len = local_model.config.max_position_embeddings
		seq_len = inputs['input_ids'].shape[1]
		if seq_len > max_len:
			inputs['input_ids'] = inputs['input_ids'][:, -max_len:]
			if 'attention_mask' in inputs:
				inputs['attention_mask'] = inputs['attention_mask'][:,-max_len:]
	gen_kwargs = dict(
		max_new_tokens=mxtkn,
		do_sample=temp > 1e-6,
		temperature=temp,
		top_p=topp,
		pad_token_id=tokenizer.eos_token_id,
		repetition_penalty=presp,
		num_return_sequences=outn)
	if is_encoder_decoder:
		gen_kwargs['decoder_start_token_id'] = \
		local_model.config.decoder_start_token_id
	with torch.no_grad():
		if is_encoder_decoder:
			output = local_model.generate(
				input_ids=inputs['input_ids'],
				attention_mask=inputs.get('attention_mask', None),
				**gen_kwargs)
		else:
			output = local_model.generate(**inputs, **gen_kwargs)
	generated_tokens = output[0][inputs['input_ids'].shape[1]:]
	text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
	if device == 'cuda' and torch.cuda.is_available(): torch.cuda.empty_cache()
	history = [{'role':'assistant', 'content':text}]
	return text

def main():
	''' Main Function '''
	# --- ChatGPT --- #
#	print(ChatGPT(CHATGPT, 'gpt-4o-mini', 'you are assistant', 'hello'))
	# --- Claude --- #
#	args = {'max_tokens':200}
#	print(Claude(CLAUDE, 'claude-3-haiku-latest', 'you person', 'hello', args))
	# --- Gemini --- #
#	print(Gemini(GEMINI, 'gemini-3-flash-preview', 'you assistant', 'hello'))
	# --- Local Model --- #
#	model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' # Excellent robustness benchmark for your code.
#	model = 'Microsoft/Phi-2'                    # Surprisingly strong reasoning for ~2.7B scale.
#	model = 'mistralai/Mistral-7B-Instruct-v0.2' # Borderline 24GB if unquantized, but very popular test model.
#	model = 'Qwen/Qwen1.5-1.8B-Chat'             # Good tokenizer edge-case tester.
#	model = 'stabilityai/StableLM-3B-4E1T'       # Lightweight general-purpose baseline.
#	model = 'openlm-research/open_llama_3b_v2'   # Good architecture compatibility check.
#	model = 'bigscience/Bloom-3B'                # Multilingual tokenizer robustness test.
#	model = 'facebook/OPT-2.7B'                  # Older but good inference stability benchmark.
#	model = 'google/gemma-2B'                    # Modern lightweight model from Google.
#	model = 'tiiuae/Falcon-7B'                   # Test memory pressure and attention behavior.
	model = 'runwayml/stable-diffusion-v1-5'     # Image generator
	print(LocalLLM('cpu', model, 'your are an assistant', 'draw me a cat'))


'''
[ ] RAG
[ ] Vector DB
[ ] Agents
[ ] Images
[ ] Function Calling
'''


#if __name__ == '__main__': main()



import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    pipeline
)

def LocalLLM(
    device='cpu',
    model='',
    system='',
    prompt='',
    output_type='auto',   # auto | text | image
    mxtkn=200,
    temp=1.0,
    topp=1.0,
    outn=1,
    presp=1.0,
    seed=None
):
    """
    Universal Local Model Inference Function

    Supports:
    - Text generation
    - Image generation (diffusion pipelines)
    - Vision-language models
    """

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -----------------------------
    # Detect device
    # -----------------------------
    device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'

    # -----------------------------
    # Load tokenizer if exists
    # -----------------------------
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        tokenizer = None

    # -----------------------------
    # Auto output mode detection
    # -----------------------------
    if output_type == 'auto':
        if "stable-diffusion" in model.lower() or "sd" in model.lower():
            output_type = 'image'
        else:
            output_type = 'text'

    # =====================================================
    # TEXT GENERATION MODE
    # =====================================================
    if output_type == 'text':

        local_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float32 if device == 'cpu' else torch.float16,
            device_map=None if device == 'cpu' else 'auto'
        )

        local_model.to(device)
        local_model.eval()

        # Chat history formatting
        input_text = ""

        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass

        if input_text == "":
            input_text = f"{system}\nUser: {prompt}\nAssistant:"

        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=mxtkn,
            do_sample=temp > 1e-6,
            temperature=temp,
            top_p=topp,
            repetition_penalty=presp,
            num_return_sequences=outn,
            pad_token_id=tokenizer.eos_token_id
        )

        # Encoder-decoder models support decoder start token
        if getattr(local_model.config, "is_encoder_decoder", False):
            gen_kwargs["decoder_start_token_id"] = \
                local_model.config.decoder_start_token_id

        with torch.no_grad():
            output = local_model.generate(**inputs, **gen_kwargs)

        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return text

    # =====================================================
    # IMAGE GENERATION MODE
    # =====================================================
    elif output_type == 'image':

        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        pipe.to(device)

        image = pipe(prompt).images[0]

        return image

    else:
        raise ValueError("Invalid output_type. Choose auto | text | image")



LocalLLM(
    device='cuda',
    model='gpt2',
    prompt='Explain quantum physics'
)

img = LocalLLM(
    device='cuda',
    model='runwayml/stable-diffusion-v1-5',
    prompt='A futuristic city at sunset',
    output_type='image'
)

img.save("output.png")
