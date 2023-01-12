import tensorflow as tf

import matplotlib.pyplot as plt

from tf_agents import environments
from tf_agents.utils import common
from tf_agents.environments import suite_gym
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from keras import optimizers

# Disabling gpu on m1 silocon performs much better
tf.config.set_visible_devices([], 'GPU')

save_dir = 'saved_state'
log_interval = 100

# Hyper parameters
num_iterations = 10000
batch_size = 64
replay_buffer_size = num_iterations
initial_collect_steps = num_iterations // 2
fc_layer_params = (100,)
num_episodes = 100
learning_rate = 1e-3
target_update_period = 200
num_parallel_calls = 3

# TODO: Adapt optuna to optimize hyper parameters
# Above the configuration trains the model below 10k iterations though

train_env = environments.TFPyEnvironment(suite_gym.load('CartPole-v0'))

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    target_update_period=target_update_period
)

agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_size
)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=initial_collect_steps)

collect_driver.run()

dataset = replay_buffer.as_dataset(
    num_steps=agent._n_step_update + 1,
    num_parallel_calls=num_parallel_calls,
    sample_batch_size=batch_size,
)

avg_return_metric = tf_metrics.AverageReturnMetric()

observers = [avg_return_metric, replay_buffer.add_batch]
collect_step_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=observers,
    num_steps=1
)

iterator = iter(dataset)
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

eval_policy = None
try:
    policy = tf.saved_model.load(save_dir)
except:
    print('No saved state found, training...')
    episodes = []
    steps = []
    for i in range(num_iterations):
        collect_step_driver.run()
        experience, _ = next(iterator)
        train_loss = agent.train(experience)
        step = agent.train_step_counter.numpy()
        if step % log_interval == 0:
            episodes.append(
                [avg_return_metric.result().numpy(), train_loss.loss])
            steps.append(step)
            print('step = {0}, loss = {1}, avg return = {2}'.format(
                step, train_loss.loss, avg_return_metric.result().numpy()))

    eval_policy = agent.policy
    policy_saver = policy_saver.PolicySaver(eval_policy)
    policy_saver.save(save_dir)

    # Plot the training metrics
    plt.plot(steps, episodes)
    plt.xlabel('Steps')
    plt.ylabel(['Average Return', 'Loss'])
    plt.show()

# Test the agent in a new environment
test_env = environments.TFPyEnvironment(suite_gym.load('CartPole-v0'))
for _ in range(num_episodes):
    time_step = test_env.reset()
    while not time_step.is_last():
        action_step = eval_policy.action(time_step)
        time_step = test_env.step(action_step)
        test_env.render(mode='human')

train_env.close()
test_env.close()
