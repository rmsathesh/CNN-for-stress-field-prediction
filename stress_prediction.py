import numpy as np
import os
import sys
import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import time

#%%
floc= r'\..\stress prediction\data\data'
fname = os.path.join(floc, 'all_data_s.npy')

input_arr = np.array(np.load(fname)).astype(np.float32)
np.random.shuffle(input_arr)

#%%

input_arr = input_arr[:22000,:]
h = 24
b = 32
train_data = input_arr[:20000,:]
test_data = input_arr[20000:,:]

#%%
def senet(x, nf, scope_name):
    
    with tf.variable_scope(scope_name):
        squeeze = tf.math.reduce_mean(x, axis=[1, 2])
        squeeze = tf.layers.dense(squeeze, units=nf//16)
        squeeze = tf.nn.relu(squeeze)
        squeeze = tf.layers.dense(squeeze, units=nf)
        excite = tf.nn.sigmoid(squeeze)
        excite = tf.reshape(excite, [-1, 1, 1, nf])
        val = x * excite
    return val
        

def conv2d(x, nf, size, scope_name):
    
    with tf.variable_scope(scope_name):
        x = tf.layers.conv2d(x, filters=nf, kernel_size=(size, size), padding='same')
        x = tf.nn.relu(x)
    return x

def resnet(x, nf, scope_name):
    
    with tf.variable_scope(scope_name):
        skip = x
        x = conv2d(x, nf=128, size=3, scope_name=scope_name + '_conv_1')
        x = conv2d(x, nf=128, size=3, scope_name=scope_name + '_conv_2')
        x = senet(x, nf, scope_name=scope_name + '_senet')
    return x


def conv2d_down(x, nf, scope_name):
    
    with tf.variable_scope(scope_name):
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])
        x = tf.layers.conv2d(x, filters=nf, kernel_size=(4, 4), strides=(2,2))
        x = tf.nn.relu(x)
    return x


def conv2d_up(x, a, nf, scope_name):
    
    with tf.variable_scope(scope_name):
        x = tf.layers.conv2d_transpose(x,filters=nf, kernel_size=(3,3), strides=(2,2), padding='same')
        x = tf.concat([x, a], axis=3)
    return x
        
#%%

def data_gen(data, batch_size=32, train=True, buffer=2048):
    
    with tf.variable_scope('data'):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if train:
            dataset = dataset.shuffle(buffer_size=buffer)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)
        iterator = dataset.make_initializable_iterator()
    return iterator
    


def model(input_layer):
    
    with tf.variable_scope('model'):
        a = []
        
        op = conv2d(input_layer, nf=32, size=9, scope_name='conv_1')
        a.append(op)
        op = conv2d_down(op, nf=32, scope_name='down_1')
        
        op = conv2d(op, nf=64, size=3, scope_name='conv_2')
        a.append(op)
        op = conv2d_down(op, nf=64, scope_name='down_2')
        
        op = resnet(op, nf=128, scope_name='resnet_1')
        op = resnet(op, nf=128, scope_name='resnet_2')
        op = resnet(op, nf=128, scope_name='resnet_3')
        
        op = conv2d_up(op, a[-1], nf=64, scope_name='up_1')
        op = conv2d(op, nf=64, size=3, scope_name='conv_3')
        
        op = conv2d_up(op, a[-2], nf=32, scope_name='up_2')
        op = conv2d(op, nf=32, size=3, scope_name='conv_4')
        
        op = conv2d(op, nf=1, size=1, scope_name='conv_5')    
    return op


    

        
#%%
def loss(stress_true, stress_pred):
    with tf.variable_scope('loss'):
        total_loss = tf.losses.mean_squared_error(stress_true, stress_pred)
        tf.summary.scalar('total_loss', total_loss)
    return total_loss

def optimizer(total_loss, global_step, learning_rate=0.001):
    with tf.variable_scope('optimizer'):
        opti = tf.train.AdamOptimizer(learning_rate=learning_rate)
        opti = opti.minimize(total_loss, global_step)   
    return opti

#%%
def data_preprocessing(data):
    n = data.shape[0]
    h = 24
    b = 32
    boundary = data[:,:h*b].reshape(-1, h, b)
    fx = data[:,h*b]
    fy = data[:,h*b + 1]
    stress = data[:,h*b+2:].reshape(-1, h, b, 1)
    bound = []
    force_x = []
    force_y = []
    d_x = []
    d_y = []
    for i in range(n):
        temp = np.where(boundary[i]==2, fx[i], 0)
        force_x.append(temp)
        temp = np.where(boundary[i]==2, fy[i], 0)
        force_y.append(temp)
        dtemp = np.zeros((h, b))
        dtemp[np.where(boundary[i,:,0]==1),0] = -1
        d_x.append(dtemp)
        d_y.append(dtemp)
        btemp = np.where(boundary[i]==2, 1, boundary[i])
        bound.append(btemp)
        
    force_x = np.array(force_x)
    force_y = np.array(force_y)
    d_x = np.array(d_x)
    d_y = np.array(d_y)
    bound = np.array(bound)
    
    return np.stack([bound, force_x, force_y, d_x, d_y], axis=-1), stress
    
    

#%%
train_val = data_preprocessing(train_data)
test_val = data_preprocessing(test_data)
   

#%%
def train(data, learning_rate=0.001):
    
    
    fpath = r'D:\study\ANN\CNN\projects\stress prediction\graph'
    mpath = r'D:\study\ANN\CNN\projects\stress prediction\model'
    
    with tf.variable_scope('main'):
        
        global_step = tf.Variable(1, trainable=False, name='global_step')
        iterator = data_gen(data)
        x, stress_true = iterator.get_next()
        stress_true = tf.cast(stress_true, tf.float64)
        tf.summary.image('boundary', x[:,:,:,0:1])
        tf.summary.image('load_x', x[:,:,:,1:2])
        tf.summary.image('load_y', x[:,:,:,2:3])
        stress_pred = model(x)
        tf.summary.image('stress_pred', stress_pred)
        loss_val = loss(stress_true, stress_pred)
        tf.summary.scalar('loss', loss_val)
        opti = optimizer(loss_val, global_step, learning_rate)
        
        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        
        epochs = 2
        stream_loss = 0
        log_iter = 1
        
        with tf.Session() as sess:
            sess.run(init)
            writer = tf.summary.FileWriter(fpath, sess.graph)
            
            ckpt = tf.train.get_checkpoint_state(mpath)
            if ckpt and ckpt.model_checkpoint_path:
                print(f'Restoring model from {mpath}')
                saver.restore(sess, ckpt.model_checkpoint_path)
                
            for i in range(1, epochs+1):
                
                k=0
                sess.run(iterator.initializer)

                while True:
                    try:
                        print(k)
                        _, total_loss = sess.run([opti, loss_val])
                        k += 1
                        assert not np.isnan(total_loss), 'Model diverged with nan'
                        if k % 50 == 0:
                            summary = sess.run(summary_op)
                            writer.add_summary(summary, i*k)
                            print(f'completed {k} iterations in epochs {i} with loss: {total_loss}')
                        stream_loss += total_loss
                    except tf.errors.OutOfRangeError:
                        print('finished one full cross')
                        saver.save(sess, save_path=mpath+'/model_lr0.0001', global_step=i)
                        break
                    
                if i%log_iter == 0:
                    print(f' Completed {i} epochs with loss: {stream_loss / (log_iter * k) :.2f}')
                    stream_loss = 0



#%%
tf.reset_default_graph()
    
train(train_val, learning_rate=0.0001)


#%%

def evaluate(x_test, stress_test):
    mpath_1 = r'D:\study\ANN\CNN\projects\stress prediction\model'
    rpath = r'D:\study\ANN\CNN\projects\stress prediction\result'
    
    with tf.Graph().as_default():
        with tf.variable_scope('main'):
            x = tf.placeholder(dtype=tf.float64, shape=[None, 24, 32, 5])
            stress_act = tf.placeholder(dtype=tf.float64, shape=[None, 24, 32, 1])
            stress_pred = model(x)
            var_list = tf.trainable_variables()
            test_loss = loss(stress_act, stress_pred)
            saver = tf.train.Saver(var_list)
            
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mpath_1)
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                tf.get_default_graph().as_graph_def()
                total_loss = 0
                for i in x_test.shape[0]:
                    x_b = x_test[i].reshape(-1, 24, 32, 5)
                    s_b = stress_test[i].reshape(-1, 24, 32, 1)
                    
                    pred, loss_val = sess.run([stress_pred, test_loss], feed_dict={x:x_b, stress_act:s_b})
                    total_loss += loss_val
                    
                    stress_plt = np.concatenate([s_b, pred, s_b-pred], axis=2)
                    plt.imshow(stress_plt[0,:,:,0], cmap='jet', interpolation='nearest')
                    plt.colorbar()
                    img_name = rpath + '/image_' + str(i)
                    plt.savefig(img_name)
                    plt.show()
                print('MSE:', total_loss / x_test.shape[0])
            

#%%
x_test, stress_test = test_val
evaluate(x_test, stress_test)

#%%




    
        
