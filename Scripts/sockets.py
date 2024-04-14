import sys
import socket

SERVER = socket.gethostbyname(socket.gethostname()) #192.168.1.1
PORT   = 5000

def client():
	''' Client side object '''#AF_INET6 IPv6
	cln = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	cln.connect((SERVER, PORT))
	message = ''
	while True:
		msg = cln.recv(8)
		if len(msg) <= 0:
			break
		message += msg.decode('UTF-8')
	print(message)

def server():
	''' Server side object '''#AF_INET6 IPv6
	srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	srv.bind((SERVER, PORT))
	srv.listen(5)
	while True:
		cln, address = srv.accept()
		print(f'[+] Connection from {address} established')
		cln.send('hello'.encode('UTF-8'))
		cln.close()

if   sys.argv[1] == '-s': server()
elif sys.argv[1] == '-c': client()
