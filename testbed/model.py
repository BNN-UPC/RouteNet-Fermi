import tensorflow as tf
import tensorflow_probability as tfp
from tensorboard.plugins.hparams import api as hp


class RouteNet(tf.keras.Model):
    def __init__(self):
        super(RouteNet, self).__init__()

        self.max_iter = 20
        self.iterations = 8
        self.threshold = 0.05
        self.ip_op_state_dim = 64
        self.path_state_dim = 64
        self.aggr_dropout = 0
        self.readout_dropout = 0

        self.max_num_sizes = 1
        self.max_num_times = 2

        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate")

        self.r_op_update = tf.keras.layers.GRUCell(self.ip_op_state_dim, name="ROpUpdate")

        self.s_op_update = tf.keras.layers.GRUCell(self.ip_op_state_dim, name="SOpUpdate")

        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=113),
            tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu)
        ])

        self.r_op_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=2),
            tf.keras.layers.Dense(self.ip_op_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.ip_op_state_dim, activation=tf.keras.activations.relu)
        ])

        self.s_op_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=2),
            tf.keras.layers.Dense(self.ip_op_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.ip_op_state_dim, activation=tf.keras.activations.relu)
        ])

        self.readout = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.path_state_dim),
            tf.keras.layers.Dense(int(self.path_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.readout_dropout),
            tf.keras.layers.Dense(int(self.path_state_dim / 4),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.readout_dropout),
            tf.keras.layers.Dense(1)
        ])

        self.readout2 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.path_state_dim),
            tf.keras.layers.Dense(int(self.path_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.readout_dropout),
            tf.keras.layers.Dense(int(self.path_state_dim / 4),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.readout_dropout),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus)
        ])

    # @tf.function
    def call(self, inputs):
        traffic = inputs['traffic']
        packets = inputs['packets']
        packet_size = inputs['packet_size']
        ipg = inputs['ipg']

        s_capacity = inputs['s_capacity']
        r_capacity = inputs['r_capacity']

        path_to_s_op = inputs['path_to_s_op']
        path_to_r_op = inputs['path_to_r_op']

        ops_to_path = inputs['ops_to_path']

        path_state = self.path_embedding(
            tf.concat([traffic, packets, packet_size, ipg],
                      axis=1))

        denorm_traffic = tf.math.multiply(traffic, tf.math.subtract(1.1091119e+09, 6677.713)) + 6677.713
        path_gather_traffic_r = tf.gather(denorm_traffic, path_to_r_op[:, :, 0])
        path_gather_traffic_s = tf.gather(denorm_traffic, path_to_s_op[:, :, 0])
        denorm_r_capacity = (tf.math.multiply(r_capacity, tf.math.subtract(80., 1.)) + 1.) * 1e9
        denorm_s_capacity = (tf.math.multiply(s_capacity, tf.math.subtract(80., 1.)) + 1.) * 1e9

        load_r = tf.math.reduce_sum(path_gather_traffic_r, axis=1) / denorm_r_capacity
        load_s = tf.math.reduce_sum(path_gather_traffic_s, axis=1) / denorm_s_capacity

        r_op_state = self.r_op_embedding(tf.concat([r_capacity, load_r], axis=1))
        s_op_state = self.s_op_embedding(tf.concat([s_capacity, load_s], axis=1))

        for it in range(self.iterations):
            ###############
            #  IP AND OP  #
            #   TO PATH   #
            ###############
            states = tf.concat([s_op_state, r_op_state], axis=0)

            ops_gather = tf.gather(states, ops_to_path)

            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True,
                                                  name="PathUpdateRNN")
            previous_path_state = path_state

            path_state_sequence, path_state = path_update_rnn(ops_gather,
                                                              initial_state=path_state)

            path_state_sequence = tf.concat([tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1)

            ##################
            #   PATH TO SOP  #
            ##################
            path_gather = tf.gather_nd(path_state_sequence, path_to_s_op, name="PathToSOp")
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            s_op_state, _ = self.s_op_update(path_sum, [s_op_state])

            ##################
            #   PATH TO ROP  #
            ##################
            path_gather = tf.gather_nd(path_state_sequence, path_to_r_op, name="PathToROp")
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            r_op_state, _ = self.r_op_update(path_sum, [r_op_state])

        r = self.readout(path_state)

        sequence_delay = self.readout2(path_state_sequence[:, 1:])

        delay = tf.math.reduce_sum(sequence_delay, axis=1)
        delay = tf.math.log(delay)

        return delay
