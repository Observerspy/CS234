import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
              
        The first hidden layer convolves 32 filters of 8*8 with stride 4 with the
        input image and applies a rectifier nonlinearity. The second hidden layer convolves
        64 filters of 4*4 with stride 2, again followed by a rectifier nonlinearity.
        This is followed by a third convolutional layer that convolves 64 filters of 3*3 with
        stride 1 followed by a rectifier. The final hidden layer is fully-connected and consists
        of 512 rectifier units. The output layer is a fully-connected linear layer with a
        single output for each valid action. The number of valid actions varied between 4
        and 18 on the games we considered.

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 

        with tf.variable_scope(scope, reuse=reuse) as _:
            X = layers.conv2d(state, 32, 8, stride=4, )
            X = layers.conv2d(X, 64, 4, stride=2, )
            X = layers.conv2d(X, 64, 3, stride=1, )
            X = layers.flatten(X)
            X = layers.fully_connected(X, 512)
            out = layers.fully_connected(X, num_actions, activation_fn=None)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
