import numpy as np

def qarray(q):
	''' Quaternion as a matrix '''
	a, b, c, d = q
	result = np.array([
	[a,-b,-c,-d],
	[b, a,-d, c],
	[c, d, a,-b],
	[d,-c, b, a]
	], dtype=np.float64)
	return(result)

def qatmul(q1, q2):
	''' Quaternion multiplication '''
	a, b, c, d = q1
	e, f, g, h = q2
	result = np.array([
	a*e - b*f - c*g - d*h,
	b*e + a*f - d*g + c*h,
	c*e + d*f + a*g - b*h,
	d*e - c*f + b*g + a*h
	], dtype=np.float64)
	return(result)

def qdot(q1, q2):
	''' Quaternion dot product '''
	a = q1[0]
	x = np.array([q1[1], q1[2], q1[3]])
	e = q2[0]
	y = np.array([q2[1], q2[2], q2[3]])
	s = a*e - np.matmul(x, y)
	v = a*y + e*x + np.cross(x, y)
	result = np.array([s, v[0], v[1], v[2]])
	return(result)
