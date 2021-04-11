# Download YouTube videos
# pip3 install pytube

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
videos = yt.streams.all()
video = list(enumerate(videos))

print('OPTION\tFORMAT\tRESOLUTION\tFPS\tSIZE')
for option in video:
	OPTION      = option[0]
	FORMAT      = option[1].subtype
	RESOLUTION  = option[1].resolution
	FPS         = option[1].fps
	SIZE        = format_bytes(int(option[1].filesize))
	line = '{}\t{}\t{}\t\t{}\t{}'.format(OPTION, FORMAT, RESOLUTION, FPS, SIZE)
	print(line)

dn_option = int(input('Enter option to download > '))
dn_video = videos[dn_option]
dn_video.download()
print('downloaded successfully')
