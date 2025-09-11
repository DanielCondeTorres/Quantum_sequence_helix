import numpy as np
from pennylane import numpy as qnp
import pennylane as qml
from pennylane import qaoa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
import itertools
import argparse
import sys
# Clasico y helice a mas vecinos
# Calculo estadistico de combinaciones posibles intentant hacer 4 qubits por posicion,  intentan juntar helices mas parecidos juntarlos
# Also support Qiskit
try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.algorithms import QAOA, VQE
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit.opflow import PauliSumOp
    from qiskit.providers.aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not available. Using PennyLane only.")
  
