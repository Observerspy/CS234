import numpy as np
import sys
from six import StringIO, b

from gym import utils
import discrete_env

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {

    "4x4": [
        "SHHH",
        "FHHH",
        "FHHH",
        "FFFG"
    ]
}

class FrozenLakeEnv(discrete_env.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SHHH
        FHHH
        FHHH
        FFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, you cannot move to these place
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole or reach max steps
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=False):

        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        self.a_true = []
        for s in range(nS):
            a_true_table = np.arange(4)
            np.random.shuffle(a_true_table)
            self.a_true.append(a_true_table)

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            a_true_table = self.a_true[to_s(row, col)]

            if a_true_table[a]==0: # left
                col = max(col-1,0)
            elif a_true_table[a]==1: # down
                row = min(row+1,nrow-1)
            elif a_true_table[a]==2: # right
                col = min(col+1,ncol-1)
            elif a_true_table[a]==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                if desc[row, col] == b"H":
                    continue
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]

                                 # if meet hole, stay at original place
                                if newletter == b'H':
                                    li.append((1.0, s, 0.0, False))
                                    continue
                                
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((0.8 if b==a else 0.1, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            # if meet hole, stay at original place
                            if newletter == b'H':
                                li.append((1.0, s, 0.0, False))
                                continue

                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        return outfile
