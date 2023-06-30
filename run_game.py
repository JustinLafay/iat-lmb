from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from dqn_agent import DQNAgent
from networks import MLP
from epsilon_profile import EpsilonProfile

def main():

    game = SpaceInvaders(display=True)
    # controller = KeyboardController()
    # controller = RandomAgent(game.na)
    controller = DQNAgent(
        qnetwork=MLP,
        eps_profile= EpsilonProfile,
    )
 
    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)

if __name__ == '__main__' :
    main()
