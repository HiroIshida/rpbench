import matplotlib.pyplot as plt

from rpbench.two_dimensional.dummy import DummyConfig, DummySolver, DummyTask

task = DummyTask.sample(1, standard=True)
res = task.solve_default()[0]
assert res.traj is not None

conf = DummyConfig(500)
online_solver = DummySolver.init(conf)

count = 0
for _ in range(100):
    task = DummyTask.sample(1, standard=False)
    problem = task.export_problems()[0]
    online_solver.setup(problem)
    result = online_solver.solve(res.traj)
    if result.traj is not None:
        count += 1
        print(count)

task.visualize()
plt.show()
