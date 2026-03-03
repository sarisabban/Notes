# pip install requests transformers

import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

# API Keys:
CHATGPT = ''
CLAUDE  = ''

class LLM:
	''' A Multimodal LLM framework '''
	def __init__(self, vendor='', model='', key='', memory=None, args=None):
		'''
		# Arguments:
		------------
		model:  str              Model name. Example: 'gpt-4o-mini'
		key:    str              The API authentication key
		memory: list             The history of the conversation
		args = {                 OpenAI's ChatGPT models
			'temperature':       1.0,             # Randomness [0.0-2.0]
			'top_p':             1.0,             # Nucleus sampling
			'n':                 1,               # Number of output responses
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
		Note 1:                  args = {'max_tokens':200} must be included
		args = {                 Hugging Face transformer models
			'max_new_tokens':        200,  # Max output tokens
			'add_generation_prompt': True, # Append tokens to start model output
			'tokenize':              True, # Returns the raw string
			'return_dict':           True, # Return dictionary of token outputs
			'return_tensors':        'pt', # Returns pt or np tensors
			'do_sample':             True, # If False temperature has no effect
			'temperature':           1.0,  # Randomness [0.0-2.0]
			'num_return_sequences':  1}    # Number of output responses
		Note 2:                      args = {'max_tokens':200} must be included
		'''
		list_of_vendors = ['openai', 'anthropic', 'transformer', 'diffusion']
		assert vendor.lower() in list_of_vendors, 'Unsupported vendor'
		if vendor.lower() == 'openai' or vendor.lower() == 'anthropic':
			assert key != '', 'API key is required'
		self.model  = model
		self.key    = key
		self.memory = memory if memory is not None else []
		self.args   = args   if args   is not None else {}
		self.vendor = vendor.lower()
		if self.vendor == 'openai':
			self.url = 'https://api.openai.com/v1/chat/completions'
		elif self.vendor == 'anthropic':
			self.url = 'https://api.anthropic.com/v1/messages'
		else:
			self.tokenizer = AutoTokenizer.from_pretrained(self.model)
			self.HGmodel = AutoModelForCausalLM.from_pretrained(self.model)

	def system(self, personality):
		''' Declare system-wide instructions '''
		if self.vendor == 'openai':
			self.memory.append({'role':'system', 'content':personality})
		elif self.vendor == 'anthropic':
			self.personality = personality
		elif self.vendor == 'transformer':
			self.memory.append({'role':'system', 'content':personality})
		elif self.vendor == 'diffusion':
			self.memory.append({'role':'system', 'content':personality})

	def chat(self, prompt):
		''' Text completion '''
		if self.vendor == 'openai':
			text = self.chat_openai(prompt)
		elif self.vendor == 'anthropic':
			text = self.chat_anthropic(prompt)
		elif self.vendor == 'transformer':
			text = self.chat_local_transformer(prompt)
		elif self.vendor == 'diffusion':
			text = self.chat_local_diffusion(prompt)
		return text

	def chat_openai(self, prompt):
		''' OpenAI's ChatGPT text completion models '''
		self.memory.append({'role':'user', 'content':prompt})
		header = {
			'Authorization':f'Bearer {self.key}',
			'Content-Type':'application/json'}
		payload = {
			'model':self.model,
			'messages':self.memory} | self.args
		response = requests.post(self.url, headers=header, json=payload)
		if response.status_code != 200:
			error = response.json()['error']['message']
			raise SystemError(error)
		else:
			text = response.json()['choices'][0]['message']['content']
			self.memory.append({'role':'assistant', 'content':text})
			return text

	def chat_anthropic(self, prompt):
		''' Anthropic's Claude and Sonnet text completion models '''
		self.memory.append({'role':'user', 'content':prompt})
		header = {
			'x-api-key':self.key,
			'anthropic-version':'2023-06-01',
			'content-type':'application/json'}
		payload = {
			'model':self.model,
			'messages':self.memory,
			'system':self.personality} | self.args
		response = requests.post(self.url, headers=header, json=payload)
		if response.status_code != 200:
			text1 = response.json()['error']['message']
			text2 = response.json()['error']['type']
			error = text1 + ' ' + text2
			raise SystemError(error)
		else:
			text = response.json()['completion']
			self.memory.append({'role':'assistant', 'content':text})
			return text

	def chat_local_transformer(self, prompt):
		''' Hugging Face's open source locally run transformer-based models '''
		self.memory.append({'role':'user', 'content':prompt})
		inputs = self.tokenizer.apply_chat_template(
			self.memory,
			return_dict=True,
			return_tensors='pt',
			**self.args).to(self.HGmodel.device)
		outputs = self.HGmodel.generate(
			**inputs,
			max_new_tokens=self.args['max_new_tokens'])
		text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])
		self.memory.append({'role':'assistant', 'content':text})
		return text

#	def chat_local_diffusion(self, prompt):
#		''' Hugging Face's open source locally run diffusion-based models '''
#		pass




def main():
#	llm = LLM('OpenAI', 'gpt-4o-mini', CHATGPT)
#	llm.system('you are a helpful assistant')
#	print(llm.chat('hello, are you online?'))
#	llm = LLM('Anthropic', 'claude-3-haiku-latest', CLAUDE, args={'max_tokens':200})
#	llm.system('you are a helpful assistant')
#	print(llm.chat('hello, are you online?'))
	llm = LLM('transformer', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', args={'max_new_tokens':200})
	llm.system('you are a helpful assistant')
	print(llm.chat('hello, are you online?'))

#	llm = LLM('diffusion', 'runwayml/stable-diffusion-v1-5')
#	llm.system('you are a helpful assistant')
#	print(llm.chat('hello, are you online?'))

if __name__ == '__main__': main()
