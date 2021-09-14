import numpy as np

class Sparse_Lander:
    def __init__(self, gym_env):

        self.env = gym_env
        #self.goal = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]) #setting the initial goal
        self.goal = np.array([0, 0, 0, 0, 0])
    def HER_new_goal(self, new_goal): #make the last state into the new goal

        self.goal = new_goal #for calculating rewards when changing goals
        return self.goal

    def reset_goal(self): #resets goal to original state

        x = np.random.uniform(-1, 1)
        y = np.random.uniform(0, 1)

        #self.goal = np.array([x, 0, 0, 0, 0, 0, 0, 1, 1])
        self.goal = np.array([x, y, 0, 0, 0]) #manouver to a given x and y co-ordinate
        

    def sparse_rewards(self, state):
        if abs(np.linalg.norm((state-self.goal))) < self.threshold:#find the absolute value of the distance between the state and the goal... if greater than threshold then failed if not then success
            reward = 1
        else:
            reward = -1 
        return reward

    def step(self, action):

        state, rew, done, _ = self.env.step(action) #take a step in environment
        
        state = np.array([state[0], state[1], state[6], state[7]]) #extract features wanted from state vector
        dist = self.compute_relative_goal_distance(state, self.goal) #compute the relative goal distance
        state = np.array([state[0], state[1], dist, state[2], state[3]]) #make the new state vector
        #state = np.array([state[0], state[1], dist, state[2], state[3], state [4], state[5] ,state[6], state[7]])
        rew = self.sparse_rewards(state) #calculate the reward
        

        if rew == 1:
            done = True #when succeeded terminate env

        return state, rew, done, _ #new step function which calculates sparse rewards

    def compute_relative_goal_distance(self, state, goal): 
        return abs(state[0]-self.goal[0]) #gets distance to x-coordinate to help the lander

    def reset(self, threshold):
        state = self.env.reset() 
        
        self.threshold = threshold #set threshold
        #self.reset_goal()
        state = np.array([state[0], state[1], state[6], state[7]]) #re-create the new state vector
        dist = self.compute_relative_goal_distance(state, self.goal) 
        #state = np.array([state[0], state[1], dist, state[2], state[3], state [4], state[5] ,state[6], state[7]])
        state = np.array([state[0], state[1], dist, state[2], state[3]])
        

         #resets the environment!
        return state