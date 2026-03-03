# pip install torch transformers

import json
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

# API Keys:
CHATGPT = ''
CLAUDE  = ''
GEMINI  = ''
HGFACE  = ''

def ChatGPT(key, model, prompt, temp=1.0, topp=1.0, outn=1, stop=None,
	seed=None, strm=False, user=None, mxtkn=None, presp=0, freqp=0,
	logit=None, resfm={}, tools=None, tchse='auto', strop=None) -> str:
	'''
	Send a prompt to OpenAI's ChatGPT LLM and return the model's response
	Official model list: https://platform.openai.com/docs/models
	# Parameters:
	-------------
	key    : str              OpenAI API authentication key
	model  : str              Model identifier string. Example: 'gpt-4o-mini'
	prompt : list
			[
				{'role': 'system',    'content': '...'}, # Personality
				{'role': 'user',      'content': '...'}, # User
				{'role': 'assistant', 'content': '...'}, # LLM
				{'role': 'user',      'content': '...'}, # User
				{'role': 'assistant', 'content': '...'}, # LLM
			]
	temp                      Randomness [0.0-2.0]
	topp                      Nucleus sampling
	outn                      Number of output responses
	stop                      Stop the sequence
	seed                      Deterministic sampling
	user                      End-user ID tracking
	strm                      Enable streaming
	strop                     Streaming behavior
	mxtkn                     Max output tokens
	presp                     Penalize topic repetition
	freqp                     Penalize token repetition
	logit                     Adjust token probabilities
	resfm                     JSON mode / schema
	tools                     Function/tool calling
	tchse                     Force specific tool
	# Returns:
	----------
	String, generated text response from the model
	String, runtime errors
	'''
	url = 'https://api.openai.com/v1/chat/completions'
	header = {
		'Authorization':f'Bearer {key}',
		'Content-Type':'application/json'}
	payload = {
		'model':             model,
		'messages':          [prompt],
		'temperature':       temp,
		'top_p':             topp,
		'n':                 outn,
		'stop':              stop,
		'seed':              seed,
		'user':              user,
		'stream':            strm,
		'stream_options':    strop,
		'max_tokens':        mxtkn,
		'presence_penalty':  presp,
		'frequency_penalty': freqp,
		'logit_bias':        logit,
		'response_format':   resfm,
		'tools':             tools,
		'tool_choice':       tchse}
	response = requests.post(url, headers=header, json=payload)
	if response.status_code == 200:
		text = response.json()['content'][0]['text']
		return text
	else:
		text = response.json()['error']['message']
		return text

def Claude(key, model, prompt, mxtkn=200, prsnlty='', temp=1.0,
	topp=1.0, topk=0, stop=[], strm=False, rsfm=None, tools=[],
	tchse={'type':'auto'}) -> str:
	'''
	Send a prompt to Anthropic's Claude LLM and return the model's response
	Official docs: https://docs.anthropic.com/claude/reference/messages_post
	# Parameters:
	-------------
	key    : str      Anthropic API key
	model  : str      Model identifier string. Example: 'claude-3-opus-20240229'
	prompt : list
			[
				{'role': 'user',      'content': '...'},
				{'role': 'assistant', 'content': '...'},
				{'role': 'user',      'content': '...'},
			]
	prsnlty           The system wide instruction
	temp              Randomness
	topp              Nucleus sampling
	topk              Limits token pool
	stop              Custom stop strings
	strm              Enable streaming
	rsfm              Arbitrary JSON metadata
	mxtkn             Max output tokens
	tools             Tool definitions
	tchse             Tool calling behavior
	# Returns:
	----------
	String, generated text response from the model
	String, runtime errors
	'''
	url = 'https://api.anthropic.com/v1/messages'
	header = {
		'x-api-key':key,
		'anthropic-version':'2023-06-01',
		'content-type':'application/json'}
	payload = {
		'model':          model,
		'messages':       prompt,
		'system':         prsnlty,
		'temperature':    temp,
		'top_p':          topp,
		'top_k':          topk,
		'stop_sequences': stop,
		'stream':         strm,
		'metadata':       rsfm,
		'max_tokens':     mxtkn,
		'tools':          tools,
		'tool_choice':    tchse}
	response = requests.post(url, headers=header, json=payload)
	if response.status_code == 200:
		text = response.json()
		return text
	else:
		text1 = response.json()['error']['message']
		text2 = response.json()['error']['type']
		text = text1 + ' ' + text2
		return text

def Gemini(key, model, prompt, prsnlty, strm=False, tmp=1.0, topp=1.0,
	topk=0, mxtkn=200, stop=[], user=1, tools=[]) -> str:
	'''
	Send a prompt to Google's Gemini LLM and return the model's response
	Official model list: hhttps://ai.google.dev/models
	# Parameters:
	-------------
	key    : str              Google's API authentication key
	model  : str              Model identifier string. Example: 'gemini-1.5-pro'
	prompt : dict
				{
	'systemInstruction': {'parts': [{'text': '...'}]}, # Personality
	'contents': [
		{'parts': [{'text': '...'}]},                  # User
		{'parts': [{'text': '...'}]},                  # LLM
		{'parts': [{'text': '...'}]},                  # User
		{'parts': [{'text': '...'}]},                  # LLM
		{'parts': [{'text': '...'}]},                  # User
				]}
	prsnlty                   The system wide instruction
	strm                      Enable streaming
	tmp                       Randomness
	topp                      Nucleus sampling
	topk                      Limits token pool
	mxtkn                     Max output tokens
	stop                      Custom stop strings
	user                      Number of users
	tools                     Tool definitions
	# Returns:
	----------
	String, generated text response from the model
	String, runtime errors
	'''
	s1 = f'https://generativelanguage.googleapis.com/v1beta/models/'
	s2 = f'{model}:generateContent?key={key}'
	url = s1 + s2
	header = {'Content-Type':'application/json'}
#	payload = prompt
	payload = {
			'systemInstruction': {'parts': [{'text':prsnlty}]},
			'contents':             prompt,
			'generationConfig':{
				'temperature':      tmp,
				'topP':             topp,
				'topK':             topk,
				'maxOutputTokens':  mxtkn,
				'stopSequences':    stop,
				'candidateCount':   user},
			'tools':                tools
			}
	response = requests.post(url, headers=header, json=payload, stream=strm)
	if response.status_code == 200:
		text = response.json()['candidates'][0]['content']['parts'][0]['text']
		return text
	else:
		text = response.json()['error']['status']
		return response.json()#text

def LocalLLM(model, prompt, device='cpu', mxtkn=200, temp=1.0, topp=1.0,
	outn=1, presp=1.0, seed=None) -> str:
	'''
	Send a prompt to a local HuggingFace LLM and return the model's response
	Official docs: https://huggingface.co/models
	# Parameters:
	-------------
	model  : str              Model name. Example: 'Microsoft/Phi-2'
	prompt : list
			[
				{'role': 'user',      'content': '...'},
				{'role': 'assistant', 'content': '...'},
				{'role': 'user',      'content': '...'},
			]
	device                    CPU for CPU or CUDA for GPU
	temp                      Randomness [0.0-2.0]
	topp                      Nucleus sampling
	mxtkn                     Max output tokens
	outn                      Number of output responses
	seed                      Deterministic sampling
	presp                     Penalize topic repetition
	# Returns:
	----------
	String, generated text response from the model
	String, runtime errors
	'''
	# Add to class __init__ later
	if seed is not None:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	tokenizer = AutoTokenizer.from_pretrained(model)
	if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
	dtype = torch.float16 if device == 'cuda' else torch.float32
	model = AutoModelForCausalLM.from_pretrained(
		model,
		dtype=dtype,
		device_map=None if device == 'cpu' else 'auto')
	model.to(device)
	model.eval()
	is_encoder_decoder = getattr(model.config, 'is_encoder_decoder', False)
	#----------------------------
	input_text = ''
	if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
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
		truncation=True).to(device)
	gen_kwargs = dict(
		max_new_tokens=mxtkn,
		do_sample=True if temp > 0 else False,
		temperature=temp,
		top_p=topp,
		pad_token_id=tokenizer.eos_token_id,
		repetition_penalty=presp,
		num_return_sequences=outn)
	if is_encoder_decoder:
		gen_kwargs['decoder_start_token_id'] = model.config.decoder_start_token_id
	with torch.no_grad():
		if is_encoder_decoder:
			output = model.generate(
				input_ids=inputs['input_ids'],
				attention_mask=inputs.get('attention_mask', None),
				**gen_kwargs)
		else:
			output = model.generate(**inputs, **gen_kwargs)
	generated_tokens = output[0]
	if generated_tokens.shape[0] > inputs["input_ids"].shape[1]:
		generated_tokens = generated_tokens[inputs["input_ids"].shape[1]:]
	return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def main():
	''' Main Function '''
	# --- ChatGPT --- #
	prompt = [{'role': 'user', 'content': 'Hello'}]
#	print(ChatGPT(CHATGPT, 'gpt-4o-mini', prompt))
	# --- Claude --- #
	prompt = [{'role': 'user', 'content': 'Hello'}]
#	print(Claude(CLAUDE, 'claude-3-haiku-latest', prompt))
	# --- Gemini --- #
	prompt = [{'parts': [{'text': 'Hello'}]}]
#	print(Gemini(GEMINI, 'gemini-3-flash-preview', prompt, 'you are assistant'))
	# --- Local Model --- #
	model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' # Excellent robustness benchmark for your code.
#	model = 'Microsoft/Phi-2'                    # Surprisingly strong reasoning for ~2.7B scale.
#	model = 'mistralai/Mistral-7B-Instruct-v0.2' # Borderline 24GB if unquantized, but very popular test model.
#	model = 'Qwen/Qwen1.5-1.8B-Chat'             # Good tokenizer edge-case tester.
#	model = 'stabilityai/StableLM-3B-4E1T'       # Lightweight general-purpose baseline.
#	model = 'openlm-research/open_llama_3b_v2'   # Good architecture compatibility check.
#	model = 'bigscience/Bloom-3B'                # Multilingual tokenizer robustness test.
#	model = 'facebook/OPT-2.7B'                  # Older but good inference stability benchmark.
#	model = 'google/gemma-2B'                    # Modern lightweight model from Google.
#	model = 'tiiuae/Falcon-7B'                   # Test memory pressure and attention behavior.
	prompt = [
		{'role': 'system', 'content': 'you are an assistant'},
		{'role': 'user', 'content': 'Hello, are you online?'},
		{'role': 'assistant', 'content': 'yes, how can i help you?'},
		{'role': 'user', 'content': 'tell me your name'}
		]
	print(LocalLLM(model, prompt))


'''
[ ] History
[ ] Temperature
[ ] Tokens

[ ] RAG
[ ] Vector DB
[ ] Agents
[ ] Images
[ ] Embeddings
[ ] Function Calling
'''


if __name__ == '__main__': main()
