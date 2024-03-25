#https://github.com/CrossRef/rest-api-doc#deep-paging-with-cursors

''' 
This is a script that collects all the Digital Object Identifiers registered under the OMICS Publishing Group and identifies all the authors of each published paper.
The final .csv file output of sorted by author name, but the data itself is still raw and has gaps since not all DOI contain the relevent information.
'''

import requests , lxml.html , urllib.parse , pandas , itertools

#Mine Data
def format_author(author):
	return('{given} {family}'.format(given=author.get('given' , '') , family = author.get('family' , '')).strip())

def format_authors(authors):
	return(';'.join(format_author(author) for author in authors))

email_address = 'EMAIL@EMAIL.com'
cursor = '*'
rows = rows_per_page = 1000
while rows == rows_per_page:
	response = requests.get('https://api.crossref.org/members/2674/works?rows={}&select=DOI,author&cursor={}&mailto={}'.format(rows_per_page , cursor , email_address))
	d = response.json()
	message = d['message']
	items = message['items']
	for item in items:
		output = item['DOI']
		if 'author' in item:
			output += ';{authors}'.format(doi=item['DOI'] , authors=format_authors(item['author']))
		tempfile = open('temp' , 'a')
		tempfile.write(output + '\n')
		tempfile.close()
		print(output)
	rows = len(items)
	cursor = urllib.parse.quote(message['next-cursor'])
'''
#A Better Script?
def format_author(author):
	return ' '.join(author.get(key , '').strip() for key in ['given' , 'family'])

def member_publications(member_id , mailto=None , rows_per_page=1000):
	params = dict(select='DOI,author' , cursor = '*' , rows = rows_per_page , )
	if mailto:
		params['mailto'] = mailto

	base_url = 'https://api.crossref.org/members/{}/works'.format(member_id)
	while True:
		response = requests.get(base_url , params = params)
		message = response.json()['message']
		items = message['items']
		for item in items:
			yield item['DOI'] , item.get('author' , [])
		if len(items) < rows_per_page:
			break
		params['cursor'] = message['next-cursor']

if __name__ == '__main__':
	for doi , authors in member_publications(2674 , mailto = 'EMAIL@EMAIL.com'):
		row = ';'.join(itertools.chain([doi] , (format_author(a) for a in authors)))
		tempfile = open('temp' , 'a')
		tempfile.write(row + '\n')
		tempfile.close()
		print(row)
'''
#Organise Data
tempfile = open('temp' , 'r')
tempfile2 = open('temp2' , 'a')
tempfile2.write('Author;DOI\n')
tempfile2.close()
for line in tempfile:
	line = line.strip().split(';')
	for entry in line[1:]:
		authors = entry + ';' + line[0] + '\n'
		tempfile2 = open('temp2' , 'a')
		tempfile2.write(authors)
		tempfile2.close()
		print(authors)
data = pandas.read_csv('temp2' , sep = ';').sort_values('Author').reset_index(drop = True)
data.to_csv('authors.csv' , sep = ';')
print(data)
os.system('rm temp temp2')

'''
#Compressed
import os , requests , urllib.parse , pandas

#Mine Data
cursor = '*'
rows = rows_per_page = 1000
while rows == rows_per_page:
	response = requests.get('https://api.crossref.org/members/2674/works?rows={}&select=DOI,author&cursor={}&mailto={}'.format(rows_per_page , cursor , 'contact@omicsgroup.com')).json()
	message = response['message']
	items = message['items']
	for item in items:
		output = item['DOI']
		if 'author' in item: output += ';{authors}'.format(doi = item['DOI'] , authors = (';'.join(('{given} {family}'.format(given = author.get('given' , '') , family = author.get('family' , '')).strip()) for author in item['author'])))
		with open('temp' , 'a') as tempfile: tempfile.write(output + '\n')
		print(output)
	rows = len(items)
	cursor = urllib.parse.quote(message['next-cursor'])

#Organise Data
tempfile = open('temp' , 'r')
with open('temp2' , 'a') as tempfile2: tempfile2.write('Author;DOI\n')
print('\x1b[31m' + 'Organising Authors...' + '\x1b[0m')
for line in tempfile:
	line = line.strip().split(';')
	for entry in line[1:]:
		authors = entry + ';' + line[0] + '\n'
		with open('temp2' , 'a') as tempfile2: tempfile2.write(authors)
print('\x1b[31m' + 'Sorting Table...' + '\x1b[0m')
data = pandas.read_csv('temp2' , sep = ';').sort_values('Author').reset_index(drop = True)
data.to_csv('authors.csv' , sep = ';')
os.remove('temp') ; os.remove('temp2') ; print('\x1b[32m' + 'Done' + '\x1b[0m')
'''
