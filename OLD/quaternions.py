import math
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

def rotate(v, n, theta):
	''' Rotate a vector around axis n '''
	n_ = n / np.linalg.norm(n)
	C = math.cos(math.radians(theta))
	S = math.sin(math.radians(theta))
	nxv = np.cross(n_, v)
	result = np.array(v)*C + nxv*S
	return(result)

def rodrigues(v, n, theta):
	''' Rotate a vector around any axis using Rodrigues Rotation Formula '''
	v = np.array(v)
	n = n / np.linalg.norm(n)
	theta = math.radians(theta)
	C = math.cos(theta)
	S = math.sin(theta)
	nxv = np.cross(n, v)
	ndv = n*(np.dot(n, v))
	result = ndv*(1-C) + v*C + nxv*S
	return(result)

def qrotate(v, n, theta):
	''' Rotate a vector around axis n using quaternions '''
	e = math.e
	theta = math.radians(theta)
	q = np.array([0, v[0], v[1], v[2]])
	n = n / np.linalg.norm(n)
	w  = math.cos(theta/2)
	nx = n[0]*math.sin(theta/2)
	ny = n[1]*math.sin(theta/2)
	nz = n[2]*math.sin(theta/2)
	P  = [w,  nx,  ny,  nz]
	Ps = [w, -nx, -ny, -nz]
	result = qatmul(qatmul(P, q), Ps)
	return(result[1:])

def reflect(v, n):
	''' Reflect a vector around axis n '''
	v = np.array(v)
	n = n / np.linalg.norm(n)
	result = v - 2*(np.dot(v, n))*n
	return(result)

def qreflect(v, n):
	''' Reflect a vector around axis n using quaternions '''
	v = np.array(v)
	n = n / np.linalg.norm(n)
	v = [0, v[0], v[1], v[2]]
	n = [0, n[0], n[1], n[2]]
	result = qatmul(qatmul(n, v), n)
	return(result[1:])


v1 = [2, 1, 1]
v2 = [2, 2, 1]
n = [0, 1, 0]
theta = 90

q1_ = qreflect(v1, n)
print(q1_)
print()

q1_ = qrotate(v1, n, theta)
q2_ = qrotate(v2, n, theta)
print(q1_)
print(q2_)
print()








def QRM(Q):
	''' Quaternion rotation matrix '''
	# https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
	q0 = Q[0]
	q1 = Q[1]
	q2 = Q[2]
	q3 = Q[3]
	r00 = 2 * (q0 * q0 + q1 * q1) - 1
	r01 = 2 * (q1 * q2 - q0 * q3)
	r02 = 2 * (q1 * q3 + q0 * q2)
	r10 = 2 * (q1 * q2 + q0 * q3)
	r11 = 2 * (q0 * q0 + q2 * q2) - 1
	r12 = 2 * (q2 * q3 - q0 * q1)
	r20 = 2 * (q1 * q3 - q0 * q2)
	r21 = 2 * (q2 * q3 + q0 * q1)
	r22 = 2 * (q0 * q0 + q3 * q3) - 1
	rot_matrix = np.array([
	[r00, r01, r02],
	[r10, r11, r12],
	[r20, r21, r22]])
	return(rot_matrix)


m = np.array([
[2, 1, 1],
[2, 2, 1]])
theta = 90
n = [0, 1, 0]

theta = math.radians(-theta)
w  = math.cos(theta/2)
nx = n[0]*math.sin(theta/2)
ny = n[1]*math.sin(theta/2)
nz = n[2]*math.sin(theta/2)
q  = [w,  nx,  ny,  nz]
RM = QRM(q)
m_ = np.matmul(m, RM)
print(m_)
