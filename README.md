# BB84 Quantum Key Distribution  

This project is a **pedagogical demonstration of the BB84 quantum key distribution protocol** implemented in Qiskit.  

The random bitstream used here does **not** come from built-in pseudorandom generators.  
Instead, it is sourced from my other project, [Chaos2Crypto](https://github.com/VishuVish/Chaos2Crypto-FFT-Based-Random-Number-Generator),  
where I developed an FFT-based random number generator with bias correction and cryptographic extraction.    

---

##  What this project demonstrates
- **Alice encodes qubits** according to her random key bits and chosen bases.
-  **Eve** may intercept, measure in random bases, and resend the qubits.  
  - This introduces errors in the sifted key.  
- **Bob measures qubits** in his own random bases.  
- **Sifting** is performed to keep only bits where Alice’s and Bob’s bases matched.  
- The result is a **shared sifted key**.  

This project is **for educational purposes only** - it is not a production-ready cryptographic implementation.  

---

##  Simulator Limitations
I used Qiskit’s `AerSimulator` as backend constraints, this demo is limited to **28 qubits** at a time.  

- With 28 qubits:  
  - On average, **12–16 bases match** between Alice and Bob.  
  - That’s roughly **47–53% agreement**, which is exactly what theory predicts (basis matches occur with probability ½).  
  - This confirms the implementation behaves as expected.
 
---
Feel free to clone, modify, and extend this project to suit your needs! 
*(Specially, I wanted to try making a Batch wise measurement to cover more number of qubits but I couldn't figure out a nice way to implement this, hence maybe anyone interested can try that as well)* 


