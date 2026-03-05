# pip install transformers pillow torch torchvision diffusers accelerate

import json
import torch
import base64
import requests
import warnings
import mimetypes
import threading
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor
from transformers import AutoModelForImageTextToText, TextIteratorStreamer
warnings.filterwarnings('ignore', message='.*CUDA is not available.*')
from diffusers import DiffusionPipeline

# API Keys:
CHATGPT = ''
CLAUDE  = ''

class LLM:
	''' A Multimodal LLM framework '''
	def __init__(self, vendor='', model='', key='', memory=None, args=None):
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
		Note 2 Local Models:     args = {'max_new_tokens':200} must be included
		'''
		list_of_vendors = ['openai', 'anthropic', 'local']
		assert vendor.lower() in list_of_vendors, 'Unsupported vendor'
		if vendor.lower() == 'openai' or vendor.lower() == 'anthropic':
			assert key != '', 'API key is required'
		self.vendor      = vendor.lower()
		self.model       = model
		self.key         = key
		self.memory      = memory if memory is not None else []
		self.args        = args
		if torch.backends.mps.is_available(): self.device = 'mps'
		elif torch.cuda.is_available() and torch.cuda.device_count() > 0:       self.device = 'cuda'
		else:                                 self.device = 'cpu'
		self.personality = ''
		if self.vendor == 'local':
			try:
				self.processor = AutoProcessor.from_pretrained(self.model)
				self.tokenizer = self.processor.tokenizer
				self.HGmodel = AutoModelForImageTextToText.from_pretrained(\
				self.model).to(self.device)
			except:
				dtype = torch.float16 if self.device != 'cpu' else torch.float32
				self.pipe = DiffusionPipeline.from_pretrained(
					self.model,
					dtype=dtype,
					device_map=self.device)
	def system(self, personality):
		''' Declare system-wide instructions '''
		if self.vendor == 'openai':
			self.memory.append({'role':'system', 'content':personality})
		elif self.vendor == 'anthropic':
			self.personality = personality
		elif self.vendor == 'local':
			self.memory.append(
				{'role':'system',
				'content':[{'type':'text', 'text':personality}]})
	def chat(self, *args, **kwargs):
		''' Image-text-text completion model '''
		if self.vendor == 'openai':
			text = self.chat_openai(*args, **kwargs)
		elif self.vendor == 'anthropic':
			text = self.chat_anthropic(*args, **kwargs)
		elif self.vendor == 'local':
			text = self.chat_local(*args, **kwargs)
		return text
	def image(self, *args, **kwargs):
		''' Image generation models '''
		if self.vendor == 'openai':
			text = self.image_openai(*args, **kwargs)
		elif self.vendor == 'anthropic':
			raise SystemError('No image generation with Anthropic models')
		elif self.vendor == 'local':
			text = self.image_local(*args, **kwargs)
		return text
	def file_encode(self, path=''):
		''' Encode images and files into base64 '''
		p = Path(path)
		mime, _ = mimetypes.guess_type(str(p))
		if mime is None: raise SystemError(f'Unrecognised file format: {mime}')
		b64 = base64.b64encode(p.read_bytes()).decode('utf-8')
		b64 = f'data:{mime};base64,{b64}'
		return b64
	def stream(self, inputs=''):
		''' Stream packets from LLM models '''
		if self.vendor == 'openai' or self.vendor == 'anthropic':
			text = ''
			for line in inputs.iter_lines():
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
						t = chunk.get('type')
						if t == 'content_block_delta':
							d = chunk.get('delta', {})
							if d.get('type') == 'text_delta':
								delta = d.get('text')
						if delta is None:
							delta = chunk.get('completion')
						if delta:
							text += delta
							print(delta, end='', flush=True)
			print()
		elif self.vendor == 'local':
			streamer = TextIteratorStreamer(
				self.processor.tokenizer,
				skip_special_tokens=True,
				skip_prompt=True)
			gen_kwargs = dict(
				**inputs,
				max_new_tokens=self.args['max_new_tokens'],
				streamer=streamer)
			threading.Thread(
				target=self.HGmodel.generate,
				kwargs=gen_kwargs,
				daemon=True).start()
			text = ''
			for chunk in streamer:
				text += chunk
				print(chunk, end='', flush=True)
			print()
		return text
	def chat_openai(self, prompt='', filename=None):
		''' OpenAI's ChatGPT image-text->text models '''
		self.args = self.args or {}
		url = 'https://api.openai.com/v1/chat/completions'
		if filename != None:
			content = []
			b64 = self.file_encode(filename)
			content.append({
				'type':'image_url',
				'image_url':{'url':b64, 'detail':'auto'}})
			content.append({'type':'text', 'text':prompt})
			self.memory.append({'role':'user', 'content':content})
		else:
			self.memory.append({'role':'user', 'content':prompt})
		payload = {
			'model':self.model,
			'messages':self.memory} | self.args
		header = {
			'Authorization':f'Bearer {self.key}',
			'Content-Type':'application/json'}
		response = requests.post(
			url,
			headers=header,
			json=payload,
			stream=self.args.get('stream', False))
		if self.args.get('stream', False):
			text = self.stream(response)
			self.memory.append({'role':'assistant', 'content':text})
			return text
		if response.status_code != 200:
			error = response.json()['error']['message']
			raise SystemError(error)
		else:
			text = response.json()['choices'][0]['message']['content']
			self.memory.append({'role':'assistant', 'content':text})
			return text
	def image_openai(self, prompt='', filename='out.png'):
		''' OpenAI's ChatGPT text->image models '''
		self.args = self.args or {}
		url = 'https://api.openai.com/v1/images/generations'
		payload = {
			'model':self.model,
			'prompt':prompt,} | self.args
		header = {
			'Authorization':f'Bearer {self.key}',
			'Content-Type':'application/json'}
		response = requests.post(
			url,
			headers=header,
			json=payload,
			stream=self.args.get('stream', False))
		b64 = response.json()['data'][0]['b64_json']
		file_bytes = base64.b64decode(b64)
		with open(filename, 'wb') as f: f.write(file_bytes)
		content = []
		content.append({
			'type':'image_url',
			'image_url':{'url':b64, 'detail':'auto'}})
		content.append({'type':'text',
		'text':f'I generated this image from the prompt: {prompt}'})
		self.memory.append({'role':'assistant', 'content':content})
		return 'Image generated'
	def chat_anthropic(self, prompt='', filename=None):
		''' Anthropic's Claud and Sonnet image-text->text models '''
		self.args = self.args or {}
		if 'max_tokens' not in self.args:
			raise ValueError('Anthropic models require args with max_tokens')
		url = 'https://api.anthropic.com/v1/messages'
		if filename != None:
			content = []
			b64 = self.file_encode(filename)
			header, b64 = b64.split(',', 1)
			media_type = header.split(';')[0].split(':', 1)[1]
			content.append({
				'type':'image',
				'source':{
					'type':'base64',
					'media_type':media_type,
					'data':b64}})
			content.append({'type':'text', 'text':prompt})
			self.memory.append({'role':'user', 'content':content})
		else:
			self.memory.append({'role':'user', 'content':prompt})
		header = {
			'x-api-key': self.key,
			'anthropic-version':'2023-06-01',
			'content-type':'application/json'}
		payload = {
			'model':self.model,
			'system':self.personality,
			'messages':self.memory} | self.args
		response = requests.post(
			url,
			headers=header,
			json=payload,
			stream=self.args.get('stream', False))
		if self.args.get('stream', False):
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
	def chat_local(self, prompt='', filename=None):
		''' Hugging Face's local transformer-based image-text->text models '''
		self.args = self.args or {}
		if 'max_new_tokens' not in self.args:
			raise ValueError('Local models require args with max_new_tokens')
		if filename != None:
			content = []
			img = Image.open(filename).convert('RGB')
			content.append({'type':'image', 'image':img})
			content.append({'type':'text', 'text':prompt})
			self.memory.append({'role':'user', 'content':content})
		else:
			self.memory.append(
			{'role':'user', 'content':[{'type':'text', 'text':prompt}]})
		inputs = self.processor.apply_chat_template(
			self.memory,
			add_generation_prompt=True,
			tokenize=True,
			return_dict=True,
			return_tensors='pt').to(self.HGmodel.device)
		outputs = self.HGmodel.generate(
			**inputs,
			max_new_tokens=self.args['max_new_tokens'])
		text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])
		if self.args.get('stream', False):
			text = self.stream(inputs)
			self.memory.append(
			{'role':'assistant', 'content':[{'type':'text', 'text':text}]})
			return text
		self.memory.append(
		{'role':'assistant', 'content':[{'type':'text', 'text':text}]})
		return text
	def image_local(self, prompt='', filename='out.png'):
		''' Hugging Face's local diffusion-based text->image models '''
		self.args = self.args or {}
		if 'size' in self.args:
			size = self.args['size'].split('x')
			H, W = int(size[0]), int(size[1])
			assert H % 8 == 0, 'Height must be divisible by 8 [32, 64, 128]'
			assert W % 8 == 0, 'Width must be divisible by 8 [32, 64, 128]'
		else:
			H, W = 512, 512
		image = self.pipe(prompt, height=H, width=W).images[0]
		image.save(filename)
		img = Image.open(filename).convert('RGB')
		content = []
		content.append({
			'type':'image_url',
			'image_url':{'url':img, 'detail':'auto'}})
		content.append({'type':'text',
		'text':f'I generated this image from the prompt: {prompt}'})
		self.memory.append({'role':'assistant', 'content':content})
		return 'Image generated'

def main():
#	# ----- ChatGPT Models ----- #
#	args = {'stream':True}
#	llm = LLM('OpenAI', 'gpt-4o-mini', CHATGPT, args=args)
#	llm.system('you are a helpful assistant')
#	# Chat
#	llm.chat(prompt='write me 300 charachter tweet about love')
#	# Analyse image
#	llm.chat(prompt='describe this image', filename='out.png')
#	# Generate image
#	args = {'size':'1024x1024', 'n':1}
#	llm = LLM('OpenAI', 'gpt-image-1', CHATGPT, args=args)
#	llm.system('you are a helpful assistant')
#	print(llm.image_openai(prompt='generate me an image of a fantasy planet', filename='planet.png'))
#	# ----- Claude Models ----- #
#	args={'max_tokens':200, 'stream':True}
#	llm = LLM('Anthropic', 'claude-opus-4-6', CLAUDE, args=args)
#	llm.system('you are a helpful assistant')
#	# Chat
#	llm.chat(prompt='hello, are you online?')
#	# Analyse image
#	llm.chat(prompt='describe this image', filename='out.png')
#	# ----- Hugging Face Local Models ----- #
#	args = {'max_new_tokens':200, 'stream':True}
#	llm = LLM(vendor='local', model='Qwen/Qwen3-VL-2B-Instruct', args=args)
#	llm.system('you are a helpful assistant')
#	# Chat
#	llm.chat(prompt='hello, are you online?')
#	llm.chat(prompt='are you sure you are online? count 1-10')
#	# Analyse image
#	llm.chat(prompt='what is the object in this image?', filename='out.png')
#	# Generate image
	llm = LLM('local', 'stable-diffusion-v1-5/stable-diffusion-v1-5')
	llm.system('you are a helpful assistant')
	print(llm.image('Alien in a jungle, warm color palette, detailed, 8k', 'out.png'))

if __name__ == '__main__': main()
