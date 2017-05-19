import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

np.set_printoptions(threshold = np.nan)


def add_layer(inputs, in_size, out_size, activation_function=None,drop = 0):
  Weights = tf.Variable(tf.truncated_normal([in_size,out_size]),name='W')
  biases = tf.Variable(tf.zeros([1, out_size])+0.001,name='b')
  Wx_plus_b = tf.add(tf.matmul(inputs, Weights),biases)
  if activation_function is None:
    outputs = Wx_plus_b
  else:
    outputs = activation_function(Wx_plus_b)
  if(drop == 1):
    outputs_d = tf.nn.dropout(outputs,keep_prob)
  else:
    outputs_d = outputs
  return outputs_d

#compute accuracy of one hot model
def compute_accuracy(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
  correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
  return result

#compute accuracy of regression model
'''
def compute_accuracy(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1})
   
  correct_prediction = tf.equal(np.round(y_pre), v_ys)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
  if(result > 0.9999):
    print(y_pre)
    input()
  return result
'''


x_data_train = np.zeros((286,1431))
y_data_train = np.zeros((286,143))
x_data_test = np.zeros((286,1431))
y_data_test = np.zeros((286,143))
#one hot label
y_data_train[:143]=np.identity(143)
y_data_train[143:]=np.identity(143)
y_data_test[:143]=np.identity(143)
y_data_test[143:]=np.identity(143)

'''
#taking half pattern for training data
d1 = 0
count = 0
with open('data_chain1') as file:
  for line in file:
    d2 = 0
    lin = line[:-1].split(', ') 
    for _ in lin:
      x_data_train[d1][d2] += float(_)
      d2 += 1
      if(d2 == 1430):
        d2 = 0
        count+=1
        if(count==98):
          break
    x_data_train[d1][1430] = 98
    d1 += 1


count = 0
with open('data_chain1_') as file:
  for line in file:
    d2 = 0
    lin = line[:-1].split(', ') 
    for _ in lin:
      x_data_train[d1][d2] += float(_)
      d2 += 1
      if(d2 == 1430):
        d2 = 0
        count+=1
        if(count==98):
          break
    x_data_train[d1][1430] = 0
    d1 += 1
#taking half pattern as testing data
d1=0
count = 98
with open('data_chain1') as file:
  for line in file:
    d2 = 0
    lin = line[:-1].split(', ') 
    for _ in lin:
      if(count == 0):
	      x_data_test[d1][d2] += float(_)
	      d2 += 1
	      if(d2 == 1430):
                d2 = 0
      else:
        count = count - 1
    x_data_test[d1][1430] = 98
    d1 += 1
count = 98
with open('data_chain1') as file:
  for line in file:
    d2 = 0
    lin = line[:-1].split(', ') 
    for _ in lin:
      if(count == 0):
	      x_data_test[d1][d2] += float(_)
	      d2 += 1
	      if(d2 == 1430):
                d2 = 0
      else:
        count = count - 1
    x_data_test[d1][1430] = 0
    d1 += 1

x_data_test = x_data_test/98
x_data_train = x_data_train/98

'''
#taking all data into x_data
x_data_train = np.zeros((286,1431))
x_data_test = np.zeros((286,1431))
d1 = 0
#sa0
with open('data_chain1') as file:
  for line in file:
    d2 = 0
    lin = line[:-1].split(', ') 
    for _ in lin:
      x_data_train[d1][d2] += float(_)
      d2 += 1
      if(d2 == 1430):
        d2 = 0
    #extra bit for SA0
    x_data_train[d1][1430] = 196
    d1 += 1


#sa1
with open('data_chain1_') as file:
  for line in file:
    d2 = 0
    lin = line[:-1].split(', ') 
    for _ in lin:
      x_data_train[d1][d2] += float(_)
      d2 += 1
      if(d2 == 1430):
        d2 = 0
    #extra bit for SA1
    x_data_train[d1][1430] = 0
    d1 += 1

d1 = 0
#sa0 p50
with open('data_chain1_p50') as file:
  for line in file:
    d2 = 0
    lin = line[:-1].split(', ') 
    for _ in lin:
      x_data_test[d1][d2] += float(_)
      d2 += 1
      if(d2 == 1430):
        d2 = 0
    #extra bit for SA0
    x_data_test[d1][1430] = 196
    d1 += 1


#sa1 p50
with open('data_chain1_p50_') as file:
  for line in file:
    d2 = 0
    lin = line[:-1].split(', ') 
    for _ in lin:
      x_data_test[d1][d2] += float(_)
      d2 += 1
      if(d2 == 1430):
        d2 = 0
    #extra bit for SA1
    x_data_test[d1][1430] = 0
    d1 += 1
x_data_train = x_data_train/196
x_data_test = x_data_test/196

'''
#taking even and odd as train and test data
x_data_train = x_data[[i_ for i_ in range(len(x_data)) if i_%2==0]]
x_data_test = x_data[[i_ for i_ in range(len(x_data)) if i_%2!=0]]
y_data_train = y_data[[i_ for i_ in range(len(y_data)) if i_%2==0]]
y_data_test = y_data[[i_ for i_ in range(len(y_data)) if i_%2!=0]]
'''

'''
#shuffle data for randomly splitting train and test data
full_data = list(zip(x_data,y_data))
np.random.shuffle(full_data)
full_data = np.concatenate((x_data,y_data),axis=1)

full_data = np.array(full_data)
x_data = full_data[:,0]
y_data = full_data[:,1]
x_data = np.array(x_data)
y_data = np.array(y_data)
x_data_train = np.zeros((143,1431))
x_data_test = np.zeros((143,1431))
y_data_train = np.zeros((143,143))
y_data_test = np.zeros((143,143))

for i_ in range(143):
  for j_ in range(1431):
    x_data_train[i_][j_] = x_data[i_][j_]
for i_ in range(143):
  for j_ in range(1431):
    x_data_test[i_][j_] = x_data[i_+143][j_]
for i_ in range(143):
  for j_ in range(143):
    y_data_train[i_][j_] = y_data[i_][j_]
for i_ in range(143):
  for j_ in range(143):
    y_data_test[i_][j_] = y_data[i_+143][j_]
'''

#xs = tf.placeholder(tf.float32,[None,280280],name='x_input')
xs = tf.placeholder(tf.float32,[None,1431],name='x_input')
ys = tf.placeholder(tf.float32,[None,143],name='y_input')

keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

#l1 = add_layer(xs,280280,50000,tf.nn.elu,1)
#l2 = add_layer(l1,50000,10000,tf.nn.elu,1)
#l3 = add_layer(l2,10000,2500,tf.nn.elu,1)
#l4 = add_layer(l3,2500,1000,tf.nn.elu,1)
#l5 = add_layer(l4,1000,500,tf.nn.elu,1)
#prediction = add_layer(l3,2500,143,tf.nn.softmax,0)
#l1 = add_layer(xs,1431,500,tf.nn.sigmoid,1)
#l2 = add_layer(l1,1100,800,tf.nn.sigmoid,1)
#l3 = add_layer(l2,800,500,tf.nn.sigmoid,1)
#l4 = add_layer(l3,500,280,tf.nn.sigmoid,1)
#l5 = add_layer(l4,600,400,tf.nn.elu,1)


prediction = add_layer(xs,1431,143,tf.nn.softmax,0)

#fixing some possible numeric error version
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction,1e-30,1.0)),reduction_indices=[1]))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

#loss = tf.reduce_mean(tf.squared_difference(prediction,ys))


train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)

#final learning rate
min_learning_rate = 0.000001
#initial learning rate
max_learning_rate = 0.01

decay_speed = 50000.0 
#drop out probability
drop_rate = 0.95

best_ac = 0
counter = 0

for i in range(50000):
  learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)
#  learning_rate = 0.000001
  sess.run(train_step,feed_dict = {xs: x_data_train, ys: y_data_train, keep_prob: drop_rate, lr: learning_rate})
  if(i%100==0):
    accuracy = compute_accuracy(x_data_train,y_data_train)
    print('accuracy: ',accuracy)
    if(accuracy>0.9999):
      counter +=1
      if(counter>10):
        break
#  cross_entro[i] = sess.run(cross_entropy,feed_dict={xs:x_data,ys:y_data})
#  accur[i] = sess.run(accuracy,feed_dict={xs:x_data,ys:y_data})

#x_axis = range(--)
#plt.plot(x_axis,cross_entro,label = cross_entropy)

#plt.plot(x_axis,accur,label = accuracy)
#plt.show

print('compute final')
print(compute_accuracy(x_data_test,y_data_test))

