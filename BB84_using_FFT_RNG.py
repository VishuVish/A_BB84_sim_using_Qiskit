import numpy as np, matplotlib.pyplot as plt
from scipy import signal
import hashlib
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

#Creating the Noise Signal
dt = 0.001
t = np.arange(0, 10000, dt)
f = 2.5*np.random.randn(len(t)) 

n = len(t)
fhat = np.fft.fft(f, n)  
PSD = fhat * np.conj(fhat) / n  
freq = (1/(dt*n)) * np.arange(n)  
L = np.arange(1, np.floor(n/2), dtype='int')  

# Using this to generate Bit Stream (0s and 1s)

threshold = np.median(PSD)  
bits = (PSD > threshold).astype(int)  
half = len(bits) // 2
folded_bits = bits[:half] ^ bits[half:half*2] 

def von_neumann_extraction(folded_bits):
    extracted_bits = []
    for i in range (0, len(folded_bits) - 1, 2):
        pairs = folded_bits[i], folded_bits[i + 1]
        if pairs == (0,1) or pairs == (1,0):
            extracted_bits.append(folded_bits[i])
    return np.array(extracted_bits)

#Randomness Extraction using SHA256 Cryptographic Hash Function
def bits_to_bytes(bits): 
    
    pad_len =(8 - len(bits) % 8) % 8    
    bits_padded = np.concatenate((bits, np.zeros(pad_len, dtype=int)))  
    bytes_arr = np.packbits(bits_padded)  
    return bytes_arr.tobytes() 

def sha256_extractor(bits):
    block_size = 256 
    final_bits = [] 
    
    for i in range (0, len(bits), block_size):
        block = bits[i:i + block_size] 
        if len(block) < block_size: 
            break 
        
        block_bytes = bits_to_bytes(block) 
        hash_bytes = hashlib.sha256(block_bytes).digest() 
        hash_bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8)) 
        final_bits.extend(hash_bits[:block_size]) 
        
    return np.array(final_bits)

final_bits = sha256_extractor(von_neumann_extraction(folded_bits))
print("Final Bit Stream Length:", len(final_bits))

#BB84 Part
# The final_bits can be used as a key for BB84 protocol or any other cryptographic application.

mid = len(final_bits) // 2
key_bits = final_bits[:mid] 
basis_bits = final_bits[mid:] 
max_qubits = 28  # Or whatever your simulator supports
key_bits = key_bits[:max_qubits]
basis_bits = basis_bits[:max_qubits]

#Encoding the bits for Alice for BB84 protocol
def bits_for_BB84(key_bits, basis_bits):
    n = len(key_bits)
    qc_alice = QuantumCircuit(n, n) 
    
    for i in range(n):
        bit = key_bits[i]
        basis = basis_bits[i]
        
        if basis == 0:
            if bit == 1:
                qc_alice.x(i) #Encode |1> state
                
        else:
            if bit == 0:
                qc_alice.h(i) # Encode |+> state
            else:
                qc_alice.x(i) # Apply X gate 
                qc_alice.h(i) # Encode |-> state
     
    return qc_alice
    
qc_alice = bits_for_BB84(key_bits, basis_bits)

#Eve presence 
def eve_is_present_and_measures(qc_alice, key_bits, basis_bits):
    n= len(key_bits)
    eve_basis_bits = np.random.randint(0, 2, size=n)  
    eve_measured_bits = []  
    
    qc_eve = QuantumCircuit(n, n)  
    
    for i in range(n):
        if eve_basis_bits[i] == 1:
            qc_eve.h(i)
            
        qc_eve.measure(i,i)  
        
        simulator = AerSimulator()
        transpiled_qc = transpile(qc_eve, simulator)
        result = simulator.run(transpiled_qc).result()
        counts = result.get_counts()
        measured_bit = int(max(counts, key=counts.get)[0])
        eve_measured_bits.append(measured_bit)
        
        #Reconstructing the key bits based on Eve's measurements
        if eve_basis_bits[i] == 0:
            if measured_bit == 1:
                qc_eve.x(i)  
        else:
            if measured_bit == 0:
                qc_eve.h(i)
            else:
                qc_eve.x(i)
                qc_eve.h(i)
                
    return qc_eve, eve_measured_bits, eve_basis_bits     

qc_eve, eve_measured_bits, eve_basis_bits = eve_is_present_and_measures(qc_alice, key_bits, basis_bits)

#Bob's measurement
def bob_measures(qc_eve, eve_basis_bits):
    n = len(eve_basis_bits)
    bob_basis_bits = np.random.randint(0, 2, size=n)  
    bob_measured_bits = []  
    
    qc_bob = QuantumCircuit(n, n)  
    for i in range(n):
        if bob_basis_bits[i] == 1:
            qc_bob.h(i)  
            
    qc_bob.measure(range(n), range(n))  
        
    simulator = AerSimulator()
    transpiled_qc = transpile(qc_bob, simulator)
    result_bob = simulator.run(transpiled_qc).result()
    counts_bob = result_bob.get_counts()

    for i in range(n):
        most_likely_outcome = max(counts_bob, key=counts_bob.get)
        bob_measured_bits = [int(bit) for bit in most_likely_outcome]
    return bob_measured_bits, bob_basis_bits

bob_measured_bits, bob_basis_bits = bob_measures(qc_eve, eve_is_present_and_measures(qc_alice, key_bits, basis_bits)[2])

#Sifting 
def sifting(key_bits, bob_basis_bits, basis_bits, bob_measured_bits):
    shared_key = []
    
    for i in range(len(basis_bits)):
        if basis_bits[i] == bob_basis_bits[i]:  
            shared_key.append(bob_measured_bits[i])
                
    return shared_key

shared_key = sifting(key_bits, bob_basis_bits, basis_bits, bob_measured_bits)
print("Shared Key:", shared_key)
matching_bases = sum(1 for i in range(len(basis_bits)) if basis_bits[i] == bob_basis_bits[i])
print("Number of Matching Bases:", matching_bases)