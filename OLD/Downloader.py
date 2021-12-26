# Download YouTube videos
# pip install pytube

from pytube import YouTube

def format_bytes(size):
	power = 2**10
	n = 0
	power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
	while size > power:
		size /= power
		n += 1
	SIZE = '{}\t{}'.format(round(size, 1), power_labels[n])
	return(SIZE)

link = input('Enter URL > ')
yt = YouTube(link)
videos = yt.streams.filter(progressive=True, file_extension='mp4')
videos = videos.order_by('resolution')

print('OPTION\tFORMAT\tRESOLUTION\tFPS\tSIZE')
for option in videos:
	try:
		OPTION      = option.itag
		FORMAT      = option.subtype
		RESOLUTION  = option.resolution
		FPS         = option.fps
		SIZE        = format_bytes(int(option.filesize))
		line = '{}\t{}\t{}\t\t{}\t{}'.format(OPTION, FORMAT, RESOLUTION, FPS, SIZE)
		print(line)
	except: pass

dn_option = int(input('Enter option to download > '))
print('[+] downloading...')
videos.get_by_itag(dn_option).download()
print('[+] Downloaded successfully')
