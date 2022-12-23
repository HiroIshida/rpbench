### design

## staticmethod MixIn based design
Most of the abstract method of the `Problem` is `@staticmethod`. Those are expected to be implemented by injecting a mixin classes.

### rejected idea
*implement it as a has-a relationship*: To do this, each class must implement own `get_sdfcreator`, `get_descriptions_sampler` etc.. and when combination of them are large, writing it becomes much cumbersome.

## separate problem and solver
motivation: to compare performance for different planners, the `solve` function should not be implemented in the problem.

