from typing import List, overload
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology , use_flops = False) -> List[Partition]:
    

      nodes = list(topology.all_nodes())
     
      ##################################
      if use_flops :
          nodes.sort(key=lambda x: (x[1].memory* sum(x[1].flops.to_dict().values()) /  3 , x[0] ), reverse=True)
      else:
          nodes.sort(key=lambda x: (x[1].memory, x[0]), reverse=True)


      total_memory = sum(node[1].memory  for node in nodes)
      partitions = []
      start = 0
      for node in nodes:
        end = round(start + (node[1].memory/total_memory), 5)
        partitions.append(Partition(node[0], start, end))
        start = end
      return partitions
    






