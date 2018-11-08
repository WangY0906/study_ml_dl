import tensorflow as tf
graph=tf.Graph()
with graph.as_default():
    w = tf.Variable(dtype=tf.float32,initial_value=1.0)
    ema = tf.train.ExponentialMovingAverage(0.9)
    update = tf.assign_add(w, 1.0)

    with tf.control_dependencies([update]):
        ema_op = ema.apply([w])#返回一个op,这个op用来更新moving_average #这句和下面那句不能调换顺序

    ema_val = ema.average(w)#此op用来返回当前的moving_average,这个参数不能是list

with tf.Session(graph=graph) as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(3):
        print (i)
        print ('w_old=',sess.run(w))
        print (sess.run(ema_op))
        print ('w_new=', sess.run(w))
        print (sess.run(ema_val))
        print ('**************')
