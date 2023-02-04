from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct    


@struct.dataclass
class Metrics(metrics.Collection):
  
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: Metrics
