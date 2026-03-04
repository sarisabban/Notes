# pip install torch requests transformers diffusers

import json
import torch
import base64
import requests
import mimetypes
from pathlib import Path
#from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# API Keys:
CHATGPT = ''
CLAUDE  = ''

class LLM:
	''' A Multimodal LLM framework '''
	def __init__(self, vendor='', model='', key='', memory=None):
		'''
		OpenAI models:       https://developers.openai.com/api/docs/models
		Anthropic models:    https://platform.claude.com/docs/en/about-claude/models/overview
		Hugging Face models: https://huggingface.co/models
		# Arguments:
		------------
		model:  str              Model name. Example: 'gpt-4o-mini'
		key:    str              The API authentication key
		memory: list             The history of the conversation
		args = {                 OpenAI's ChatGPT models
			'temperature':       1.0,             # Randomness [0.0-2.0]
			'top_p':             1.0,             # Nucleus sampling
			'n':                 1,               # Number of output responses
			'size':              1024x1240        # Image size
			'stop':              None,            # Stop the sequence
			'seed':              None,            # Deterministic sampling
			'user':              None,            # End-user ID tracking
			'stream':            False,           # Enable streaming
			'stream_options':    None,            # Streaming behavior
			'max_tokens':        None,            # Max output tokens
			'presence_penalty':  0,               # Penalize topic repetition
			'frequency_penalty': 0,               # Penalize token repetition
			'logit_bias':        None,            # Adjust token probabilities
			'response_format':   {},              # JSON mode / schema
			'tools':             None,            # Function/tool calling
			'tool_choice':       'auto'}          # Force specific tool
		args = {                 Anthropic's models
			'temperature':       1.0,             # Randomness
			'top_p':             1.0,             # Nucleus sampling
			'top_k':             0,               # Limits token pool
			'stop_sequences':    [],              # Custom stop strings
			'stream':            False,           # Enable streaming
			'metadata':          None,            # Arbitrary JSON metadata
			'max_tokens':        200,             # Max output tokens
			'tools':             [],              # Tool definitions
			'tool_choice':       {'type':'auto'}} # Tool calling behavior
		Note 1 Claude Models:    args = {'max_tokens':200} must be included
		args = {                 Hugging Face transformer models
			'max_new_tokens':        200,  # Max output tokens
			'add_generation_prompt': True, # Append tokens to start model output
			'tokenize':              True, # Returns the raw string
			'return_dict':           True, # Return dictionary of token outputs
			'return_tensors':        'pt', # Returns pt or np tensors
			'do_sample':             True, # If False temperature has no effect
			'temperature':           1.0,  # Randomness [0.0-2.0]
			'num_return_sequences':  1}    # Number of output responses
		Note 2 Local Models:         args = {'max_tokens':200} must be included
		'''
		list_of_vendors = ['openai', 'anthropic', 'transformer', 'diffusion']
		assert vendor.lower() in list_of_vendors, 'Unsupported vendor'
		if vendor.lower() == 'openai' or vendor.lower() == 'anthropic':
			assert key != '', 'API key is required'
		self.model       = model
		self.key         = key
		self.memory      = memory if memory is not None else []
		self.vendor      = vendor.lower()
		self.device      = 'mps' if torch.backends.mps.is_available() else 'cpu'
		self.personality = ''
#		elif self.vendor == 'transformer':
#			self.tokenizer = AutoTokenizer.from_pretrained(self.model)
#			self.HGmodel = AutoModelForCausalLM.from_pretrained(self.model)
#		elif self.vendor == 'diffusion':
#			dtype = torch.float16 if self.device != 'cpu' else torch.float32
#			self.pipe = DiffusionPipeline.from_pretrained(
#				self.model,
#				torch_dtype=dtype).to(self.device)

	def file_encode(self, path=''):
		''' Encode images and files into base64 '''
		p = Path(path)
		mime, _ = mimetypes.guess_type(str(p))
		if mime is None: raise SystemError(f'Unrecognised file format: {mime}')
		b64 = base64.b64encode(p.read_bytes()).decode('utf-8')
		b64 = f'data:{mime};base64,{b64}'
		return b64

	def system(self, personality):
		''' Declare system-wide instructions '''
		if self.vendor == 'openai':
			self.memory.append({'role':'system', 'content':personality})
		elif self.vendor == 'anthropic':
			self.personality = personality
#		elif self.vendor == 'transformer':
#			self.memory.append({'role':'system', 'content':personality})
#		elif self.vendor == 'diffusion':
#			self.memory.append({'role':'system', 'content':personality})

	def stream(self, response=''):
		''' Stream packets from LLM models '''
		text = ''
		for line in response.iter_lines():
			decoded = line.decode('utf-8')
			if decoded.startswith('data: '):
				data = decoded[6:]
				if data == '[DONE]': break
				chunk = json.loads(data)
				if self.vendor == 'openai':
					delta = chunk['choices'][0]['delta'].get('content')
					if delta:
						text += delta
						print(delta, end='', flush=True)
				elif self.vendor == 'anthropic':
					delta = None
					t = chunk.get("type")
					if t == "content_block_delta":
						d = chunk.get("delta", {})
						if d.get("type") == "text_delta":
							delta = d.get("text")
					if delta is None:
						delta = chunk.get("completion")
					if delta:
						text += delta
						print(delta, end='', flush=True)
		print()
		return text

	def chat(self, *args, **kwargs):
		''' Text completion '''
		if self.vendor == 'openai':
			text = self.chat_openai(*args, **kwargs)
		elif self.vendor == 'anthropic':
			text = self.chat_anthropic(*args, **kwargs)
#		elif self.vendor == 'transformer':
#			text = self.chat_local_transformer(prompt)
		return text

	def chat_openai(self, mode='chat', prompt='', filename=None, args=None):
		''' OpenAI's ChatGPT models '''
		args = args or {}
		if mode == 'image':
			url = 'https://api.openai.com/v1/images/generations'
			payload = {
				'model':self.model,
				'prompt':prompt,} | args
		elif mode == 'chat':
			url = 'https://api.openai.com/v1/chat/completions'
			if filename != None:
				content = [{'type':'text', 'text':prompt}]
				b64 = self.file_encode(filename)
				content.append({
					'type':'image_url',
					'image_url':{'url':b64, 'detail':'auto'}})
				self.memory.append({'role':'user', 'content':content})
			else:
				self.memory.append({'role':'user', 'content':prompt})
			payload = {
				'model':self.model,
				'messages':self.memory} | args
		header = {
			'Authorization':f'Bearer {self.key}',
			'Content-Type':'application/json'}
		response = requests.post(
			url,
			headers=header,
			json=payload,
			stream=args.get('stream', False))
		if args.get('stream', False):
			text = self.stream(response)
			self.memory.append({'role':'assistant', 'content':text})
			return text
		if mode == 'chat':
			if response.status_code != 200:
				error = response.json()['error']['message']
				raise SystemError(error)
			else:
				text = response.json()['choices'][0]['message']['content']
				self.memory.append({'role':'assistant', 'content':text})
				return text
		elif mode == 'image':
			b64 = response.json()['data'][0]['b64_json']
			file_bytes = base64.b64decode(b64)
			with open(filename, 'wb') as f: f.write(file_bytes)
			content = [{'type':'text', 'text':f'I generated this image from the prompt: {prompt}'}]
			content.append({
				'type':'image_url',
				'image_url':{'url':b64, 'detail':'auto'}})
			self.memory.append({'role':'assistant', 'content':content})
			return 'Image generated'

	def chat_anthropic(self, mode='chat', prompt='', filename=None, args=None):
		''' OpenAI's ChatGPT models '''
		args = args or {}
		url = 'https://api.anthropic.com/v1/messages'
		if filename != None:
			content = [{'type':'text', 'text':prompt}]
			b64 = self.file_encode(filename)
			header, b64 = b64.split(",", 1)
			media_type = header.split(";")[0].split(":", 1)[1]
			content.append({
				'type':'image',
				'source':{
					'type':'base64',
					'media_type':media_type,
					'data':b64}})
			self.memory.append({'role':'user', 'content':content})
		else:
			self.memory.append({'role':'user', 'content':prompt})
		header = {
			"x-api-key": self.key,
			"anthropic-version":"2023-06-01",
			"content-type":"application/json"}
		payload = {
			'model':self.model,
			'system':self.personality,
			'messages':self.memory} | args
		response = requests.post(
			url,
			headers=header,
			json=payload,
			stream=args.get('stream', False))
		if args.get('stream', False):
			text = self.stream(response)
			self.memory.append({'role':'assistant', 'content':text})
			return text
		if response.status_code != 200:
			error = response.json()['error']['message']
			raise SystemError(error)
		else:
			text = response.json()['content'][0]['text']
			self.memory.append({'role':'assistant', 'content':text})
			return text
























#	def chat_local_transformer(self, prompt):
#		''' Hugging Face's open source locally run transformer-based models '''
#		self.memory.append({'role':'user', 'content':prompt})
#		inputs = self.tokenizer.apply_chat_template(
#			self.memory,
#			return_dict=True,
#			return_tensors='pt',
#			**self.args).to(self.HGmodel.device)
#		outputs = self.HGmodel.generate(
#			**inputs,
#			max_new_tokens=self.args['max_new_tokens'])
#		text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])
#		self.memory.append({'role':'assistant', 'content':text})
#		return text

#	def image_local_diffusion(self, prompt, filename):
#		''' Hugging Face's open source locally run diffusion-based models '''
#		image = self.pipe(prompt).images[0]
#		image.save(filename)




#model_id = 'stabilityai/sd-turbo'  # good lightweight choice
#device = 'mps' if torch.backends.mps.is_available() else 'cpu'
#pipe = DiffusionPipeline.from_pretrained(
#    model_id,
#    torch_dtype=torch.float16 if device != 'cpu' else torch.float32,)
#pipe = pipe.to(device)
#image = pipe('a futuristic cyberpunk city at night').images[0]
#image.save('output.png')



def main():
#	# --- ChatGPT Models --- #
#	llm = LLM('OpenAI', 'gpt-4o-mini', CHATGPT)
#	llm.system('you are a helpful assistant')
#	llm.chat(mode='chat', prompt='write me 300 charachter tweet about love', args={'stream':True})

#	llm = LLM('OpenAI', 'gpt-4o-mini', CHATGPT)
#	llm.system('you are a helpful assistant')
#	print(llm.chat(mode='chat', prompt='describe this image', filename='out.png'))

#	args = {'size':'1024x1024', 'n':1}
#	llm = LLM('OpenAI', 'gpt-image-1', CHATGPT)
#	print(llm.chat(mode='image', prompt='generate me an image of a fantasy planet', filename='planet.png', args=args))

#	# --- Claude Models --- #
#	llm = LLM('Anthropic', 'claude-opus-4-6', CLAUDE)
#	llm.system('you are a helpful assistant')
#	llm.chat(mode='chat', prompt='hello, are you online?', args={'max_tokens':200, 'stream':True})

#	llm = LLM('Anthropic', 'claude-opus-4-6', CLAUDE)
#	llm.system('you are a helpful assistant')
#	print(llm.chat(mode='chat', prompt='describe this image', filename='out.png', args={'max_tokens':200}))


	pass
	# --- Hugging Face Local Models --- #
#	llm = LLM('transformer', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', args={'max_new_tokens':200})
#	llm.system('you are a helpful assistant')
#	print(llm.chat('hello, are you online?'))
#	M = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
#	M = 'stabilityai/sd-turbo'
#	M = 'stabilityai/sdxl-turbo'
#	llm = LLM('diffusion', M)#'Tongyi-MAI/Z-Image-Turbo')
#	llm.system('you are a helpful assistant')
#	llm.image('Astronaut in a jungle, cold color palette, muted colors, detailed, 8k', 'out.png')

if __name__ == '__main__': main()
