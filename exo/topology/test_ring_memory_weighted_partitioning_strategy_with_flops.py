"""
    This script for testing the new stratgy of using partitioning which the FLOPS of each device where added as a parameter 
    
    Changes :
        -   Instead of using only the memory of the device the function were modified :
                from : nodes_order = sort_in_desc( node.memory   )
                to   : nodes_order = sort_in_desc(node_memory * (sum_of_flops_accross_different_structure / 3 ))

        -   New argument where added to the function `RingMemoryWeightedPartitioningStrategy` which is `use_flops` to use the new methodology


    
    The test script shows that the function works as intended and pass all the tests successfully 
"""





import unittest
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition



class TestRingMemoryWeightedPartitioningStrategy(unittest.TestCase):
  def test_partition(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
    topology = Topology()
    TFLOPS = 1
    topology.update_node(
      "node1",
      DeviceCapabilities(
        model="NVIDIA GEFORCE RTX 2060",
        chip="test1",
        memory=2000 , 
        flops=DeviceFlops(fp32=6.45*TFLOPS, fp16=12.9*TFLOPS, int8=25.8*TFLOPS), ###  15.05 * 2000 = 30100   --> 2    3
      ),
    )
    topology.update_node(
      "node2",
      DeviceCapabilities(
        model="NVIDIA QUATRO RTX A2000",
        chip="test1",
        memory=3000,
        flops=DeviceFlops(fp32=7.99*TFLOPS, fp16=7.99*TFLOPS, int8=31.91*TFLOPS) ### 15.96 * 3000 = 47880  --->  1   2
      ),
    )
    topology.update_node(
      "node3",
      DeviceCapabilities(
        model="AMD Radeon RX 6900 XT",
        chip="test3",
        memory=1000,
        flops=DeviceFlops(fp32=23.04*TFLOPS, fp16=46.08*TFLOPS, int8=92.16*TFLOPS), ### 53.76  * 1000 = 53760 ---> 3   1
      ),
    )
    topology.add_edge("node1", "node2")
    topology.add_edge("node2", "node3")
    topology.add_edge("node3", "node1")
    topology.add_edge("node1", "node3")

    strategy = RingMemoryWeightedPartitioningStrategy()
    ## MODIFY TO USE FLOPS
    partitions = strategy.partition(topology , use_flops=True)

    self.assertEqual(len(partitions), 3)
    self.assertEqual(
      partitions,
      [
        Partition("node3", 0.0, 0.16667),
        Partition("node2", 0.16667, 0.66667),
        Partition("node1", 0.66667, 1.0),
      ],
    )

  def test_partition_rounding(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
    TFLOPS = 1
    topology = Topology()
    topology.update_node(
      "node1",
      DeviceCapabilities(
        model="NVIDIA GEFORCE RTX 2060",
        chip="test1",
        memory=2000 , 
        flops=DeviceFlops(fp32=6.45*TFLOPS, fp16=12.9*TFLOPS, int8=25.8*TFLOPS), ###  15.05 * 2000 = 30100   --> 2    3
      ),
    )
    topology.update_node(
      "node2",
      DeviceCapabilities(
        model="NVIDIA QUATRO RTX A2000",
        chip="test1",
        memory=3000,
        flops=DeviceFlops(fp32=7.99*TFLOPS, fp16=7.99*TFLOPS, int8=31.91*TFLOPS) ### 15.96 * 3000 = 47880  --->  1   2
      ),
    )
    topology.update_node(
      "node3",
      DeviceCapabilities(
        model="AMD Radeon RX 6900 XT",
        chip="test3",
        memory=1000,
        flops=DeviceFlops(fp32=23.04*TFLOPS, fp16=46.08*TFLOPS, int8=92.16*TFLOPS), ### 53.76  * 1000 = 53760 ---> 3   1
      ),
    )

    strategy = RingMemoryWeightedPartitioningStrategy()
    partitions = strategy.partition(topology , use_flops = True)
    print("-"*50)
    print("PARITIONS" , partitions ,"\n")

    self.assertEqual(len(partitions), 3)
    self.assertEqual(
      partitions,
      [
        Partition("node3", 0.0, 0.16667),
        Partition("node2", 0.16667, 0.66667),
        Partition("node1", 0.66667, 1.0),
      ],
    )


if __name__ == "__main__":
  unittest.main()

