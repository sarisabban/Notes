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

def ChatGPT(key, model, prompt) -> str:
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
		'model':model,
		'messages':[prompt],
		'temperature':0.9,#temp 0-2 randomness
		'top_p':'',#float nucleaus sampleing
		'n':'',#int number of responses
		'max_tokens':'',#int max token output
		'stop':'',#str or list stop sequence
		'presence_penalty':'',#float Penalize topic repetition
		'frequency_penalty':'',#float Penalize token repetition
		'logit_bias':'',#dict Adjust token probabilities
		'seed':'',#int Deterministic sampling
		'response_format':'',#dict JSON mode / schema
		'tools':'',#list Function/tool calling
		'tool_choice':'',#str/dict Force specific tool
		'stream':'',#bool Enable streaming
		'stream_options':'',#dict Streaming behavior
		'user':'',}#str End-user ID tracking
	response = requests.post(url, headers=header, json=payload)
	if response.status_code == 200:
		text = response.json()['choices'][0]['message']['content']
		return text
	else:
		text = response.json()['error']['code']
		return text

def Claude(key, model, personality, prompt) -> str:
	'''
	Send a prompt to Anthropic's Claude LLM and return the model's response
	Official docs: https://docs.anthropic.com/claude/reference/messages_post
	# Parameters:
	-------------
	key    : str    # Anthropic API key
	model  : str    # Model identifier string. Example: 'claude-3-opus-20240229'
	prompt : list
			[
				{'role': 'user',      'content': '...'},
				{'role': 'assistant', 'content': '...'},
				{'role': 'user',      'content': '...'},
			]
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
	payload = {'model':model,
		'system':personality,
		'max_tokens':200,
		'messages':prompt}
	response = requests.post(url, headers=header, json=payload)
	if response.status_code == 200:
		text = response.json()
		return text
	else:
		text1 = response.json()['error']['message']
		text2 = response.json()['error']['type']
		text = text1 + ' ' + text2
		return text

def Gemini(key, model, prompt) -> str:
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
	# Returns:
	----------
	String, generated text response from the model
	String, runtime errors
	'''
	s1 = f'https://generativelanguage.googleapis.com/v1beta/models/'
	s2 = f'{model}:generateContent?key={key}'
	url = s1 + s2
	header = {'Content-Type':'application/json'}
	payload = prompt
	response = requests.post(url, headers=header, json=payload)
	if response.status_code == 200:
		text = response.json()['candidates'][0]['content']['parts'][0]['text']
		return text
	else:
		text = response.json()['error']['status']
		return text

def LocalLLM(model, prompt, device='cpu', max_tokens=200, temp=0.01, top_p=0.9):
	# Add to class __init__ later
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
	embedings = tokenizer(
		input_text,
		return_tensors='pt',
		padding=True,
		truncation=True).to(device)
	gen_kwargs = dict(
		max_new_tokens=max_tokens,
		do_sample=True if temp > 0 else False,
		temperature=max(temp, 1e-5),
		top_p=top_p,
		pad_token_id=tokenizer.eos_token_id,
		repetition_penalty=1.1)
	if is_encoder_decoder:
		gen_kwargs['decoder_start_token_id'] = model.config.decoder_start_token_id
	with torch.no_grad():
		if is_encoder_decoder:
			output = model.generate(
				input_ids=embedings['input_ids'],
				attention_mask=embedings.get('attention_mask', None),
				**gen_kwargs)
		else:
			output = model.generate(**embedings, **gen_kwargs)
	generated_tokens = output[0]
	if generated_tokens.shape[0] > embedings["input_ids"].shape[1]:
		generated_tokens = generated_tokens[embedings["input_ids"].shape[1]:]
	return tokenizer.decode(generated_tokens, skip_special_tokens=True)









def main():
	''' Main Function '''
	# --- ChatGPT --- #
	prompt = [{'role': 'user', 'content': 'Hello'}]
	text = ChatGPT(CHATGPT, 'gpt-4o-mini', prompt)
	print(text)
	# --- Claude --- #
	prompt = [{'role': 'user', 'content': 'Hello'}]
#	text = Claude(CLAUDE, 'claude-3-opus-20240229', 'you are assistant', prompt)
#	print(text)
	# --- Gemini --- #
	prompt = {'contents': [{'parts': [{'text': 'Hello'}]}]}
#	text = Gemini(GEMINI, 'gemini-3-flash-preview', prompt)
#	print(text)
	# --- Local Model --- #
	# Small models:
	model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' # Excellent robustness benchmark for your code.
	model = 'Microsoft/Phi-2'                    # Surprisingly strong reasoning for ~2.7B scale.
	#model = 'mistralai/Mistral-7B-Instruct-v0.2' # Borderline 24GB if unquantized, but very popular test model.
	model = 'Qwen/Qwen1.5-1.8B-Chat'             # Good tokenizer edge-case tester.
	model = 'stabilityai/StableLM-3B-4E1T'       # Lightweight general-purpose baseline.
	#model = 'openlm-research/open_llama_3b_v2'    # Good architecture compatibility check.
	model = 'bigscience/Bloom-3B'                # Multilingual tokenizer robustness test.
	model = 'facebook/OPT-2.7B'                  # Older but good inference stability benchmark.
	#model = 'google/gemma-2B'                    # Modern lightweight model from Google.
	model = 'tiiuae/Falcon-7B'                   # Test memory pressure and attention behavior.
	
	prompt = [
		{'role': 'system', 'content': 'you are an assistant'},
		{'role': 'user', 'content': 'Hello, are you online?'},
		{'role': 'assistant', 'content': 'yes, how can i help you?'},
		{'role': 'user', 'content': 'tell me your name'}
		]
#	text = LocalLLM(model, prompt)
#	print(text)
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
