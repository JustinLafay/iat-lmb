import sys
import time
import pickle
from dqn_agent import DQNAgent
from epsilon_profile import EpsilonProfile
from game.SpaceInvaders import SpaceInvaders
import game
import numpy as np

from networks import MLP, CNN

# test once by taking greedy actions based on Q values
def test_maze(env: SpaceInvaders, agent: DQNAgent, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = env.reset() if (same) else env.reset()
        if display:
            env.render()

        for step in range(max_steps):
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = env.step(action)

            if display:
                time.sleep(speed)
                env.render()

            sum_rewards += reward
            if terminal:
                n_steps = step+1  # number of steps taken
                break
            state = next_state
    return n_steps, sum_rewards


def main(nn: str= "mlp", mode : str= "test"):
    
    if(mode == "learn"):
        env = SpaceInvaders(display=True)
    if(mode == "test"):
        env = SpaceInvaders(display=True)
    """ INSTANCIE LE LABYRINTHE """ 
    

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    # Hyperparamètres basiques
    n_episodes = 100
    max_steps = 5
    gamma = 0.05
    alpha = 0.5
    eps_profile = EpsilonProfile(1.0, 0.1)

    # Hyperparamètres de DQN
    final_exploration_episode = 30
    batch_size = len(env.get_state())
    replay_memory_size = 100
    target_update_frequency = 100
    tau = 1.0

    """ INSTANCIE LE RESEAU DE NEURONES """
    if (nn == "mlp"):
        model = MLP(len(env.get_state()), 4)
    elif (nn == "cnn"):
        model = CNN(env.ny, env.nx, env.nf, env.na)
    else:
        print("Error : Unknown neural network (" + nn + ").")
    

    print('--- neural network ---')
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('number of parameters:', num_params)
    print(model)

    """  LEARNING PARAMETERS"""
    agent = DQNAgent(model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode)
    print("******* mode = "+mode+"********")
    if(mode == "learn"):
        lines = agent.learn(env, n_episodes, max_steps)
        with open('test_scores.txt', 'w') as f:
            for line in lines:
                f.write(f"{line}\n")
        file = open("./dqn_parameters/dqn.pkl", "wb")
        pickle.dump(agent, f)
        file.close()
    
    elif(mode == "test"):
        print("Début du test")
        f = open("./dqn_parameters/dqn.pkl", "rb")
        agent2 = pickle.load(f)
        f.close()
        state = env.reset()
        print("Test du jeu...")
        while True:
            action = agent2.select_action(state)
            state, reward, is_done = env.step(action)
            time.sleep(0.001)
    
    
    
    
    # test_maze(env, agent, max_steps, speed=1, display=False)

if __name__ == '__main__':
    """ Usage : python main.py [ARGS]
    - First argument (str) : the name of the agent (i.e. 'random', 'vi', 'qlearning', 'dqn')
    - Second argument (int) : the maze hight
    - Third argument (int) : the maze width
    """
    if (len(sys.argv) > 2):
        main(sys.argv[1], sys.argv[2])
    elif (len(sys.argv) > 1):
        main(sys.argv[1], [])
    else:
        main("random", [])