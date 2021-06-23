import numpy as np


class Agent:
    _total_counter = 0
    _total_alive_counter = 0

    def __init__(self, node=0, probability_survival=0.5, beard=False):
        self.node = node
        self.fed = False
        self.alive = True
        self.id = Agent._total_counter
        self.probability_survival = probability_survival
        self.beard = beard

        # increase count of all agents
        Agent._total_counter += 1
        Agent._total_alive_counter += 1

        # increase count of specific type of agent
        type(self)._counter += 1
        type(self)._alive_counter += 1

    def name(self):
        return type(self).__name__

    def reproduce(self):
        # asexual reproduction
        # original Agent becomes N more agents
        n_children = np.random.choice(a=[1, 2], size=1, p=[0.5, 0.5])[0]
        children = []
        for _ in range(n_children):
            # initialize new children of same subclass as parent
            children.append(type(self)(self.probability_survival))
        return children

    def die(self):
        self.alive = False
        # decrease total alive counter of all agents
        Agent._total_alive_counter -= 1
        # decrease total alive counter of specific subclass of agent
        type(self)._alive_counter -= 1

    def feed(self):
        self.fed = True

    def __repr__(self):
        return f"【{self.name()} {self.id}】"


class Coward(Agent):
    _counter = 0  # total created
    _alive_counter = 0  # currently alive

    def behavior(self, pair):
        # Cowards run away, saving themselves but killing another agent
        pair.die()


class Impostor(Coward):
    _counter = 0  # total created
    _alive_counter = 0  # currently alive

    # Impostors are Cowards that have Green Beards
    def __init__(self, _):
        super().__init__(beard=True)


class Altruist(Agent):
    _counter = 0  # total created
    _alive_counter = 0  # currently alive

    def __init__(self, probability_survival):
        super().__init__(probability_survival)

    def behavior(self, _):
        # Altruists risk their own lives
        # to guarantee the survival of another agent,
        # regardless of any characteristic of the other agent
        survival = np.random.choice(
            a=[0, 1],
            size=1,
            p=[1.0 - self.probability_survival, self.probability_survival],
        )[0]

        if not survival:
            self.die()


class GreenBeardAltruist(Agent):
    _counter = 0  # total created
    _alive_counter = 0  # currently alive

    def __init__(self, probability_survival, beard=True):
        super().__init__(probability_survival, beard)

    def behavior(self, pair):
        # Green Beard Altruists risk their lives to save only agents
        # that also have a Green Beard
        if pair.beard:
            survival = np.random.choice(
                a=[0, 1],
                size=1,
                p=[1.0 - self.probability_survival, self.probability_survival],
            )[0]
            if not survival:
                self.die()
        else:
            pair.die()


class BeardlessGreenBeardAltruist(GreenBeardAltruist):
    _counter = 0  # total created
    _alive_counter = 0  # currently alive

    def __init__(self, probability_survival):
        super().__init__(probability_survival, beard=False)
