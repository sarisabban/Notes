import tensorflow as tf

'''
#General structure to preform a calculation
w = tf.Variable([1.0] , tf.float32)
b = tf.Variable([2.0] , tf.float32)
x = tf.placeholder(tf.float32)
y = w * x + b

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(y , {x:[1,2,3,4,5,6,7,8,9,0]}))
'''

#Model
#Loss Function
#Gradient Descent
