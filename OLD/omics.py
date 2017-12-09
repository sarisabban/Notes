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

email_address = 'ac.research@icloud.com'
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
		tempfile.write(output)
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
	for doi , authors in member_publications(2674 , mailto = 'ac.research@icloud.com'):
		row = ';'.join(itertools.chain([doi] , (format_author(a) for a in authors)))
		tempfile = open('temp' , 'a')
		tempfile.write(row)
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
