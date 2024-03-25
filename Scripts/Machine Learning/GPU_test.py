import tensorflow as tf
import timeit

def Find():
	''' FIND GPU '''
	device_name = tf.test.gpu_device_name()
	if device_name != '/device:GPU:0': raise SystemError('GPU device not found')
	print('\x1b[31mFound GPU at: {}\x1b[0m'.format(device_name))

def Test():
	''' TEST CPU/GPU SPEED '''
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.device('/cpu:0'):
		random_image_cpu = tf.random_normal((100, 100, 100, 3))
		net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
		net_cpu = tf.reduce_sum(net_cpu)
	with tf.device('/gpu:0'):
		random_image_gpu = tf.random_normal((100, 100, 100, 3))
		net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
		net_gpu = tf.reduce_sum(net_gpu)
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	def cpu(): sess.run(net_cpu)
	def gpu(): sess.run(net_gpu)
	print('Time(s) to convolve 32x7x7x3 filter over random 100x100x100x3 images'
		'(batch x height x width x channel). Sum of ten runs.')
	print('CPU(s):')
	cpu_time = timeit.timeit('cpu()',number=10,setup="from __main__ import cpu")
	print(cpu_time)
	print('GPU(s):')
	gpu_time = timeit.timeit('gpu()',number=10,setup="from __main__ import gpu")
	print(gpu_time)
	print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
	sess.close()

def main():
	try:
		Find()
		Test()
	except:
		print('GPU device not found')

if __name__ == '__main__': main()
