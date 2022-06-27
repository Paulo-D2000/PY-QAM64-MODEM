import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

################################## HELPER FUNCTIONS #####################################################

def CharToBits(p_chr):
    return [x for x in bin(ord(p_chr))[2:].zfill(8)]

def BitsToChars(bits):
    return [chr(int(c,2)) for c in bits]

def qam_bytes(bytes_in):
    pbytes = np.reshape([[y for y in bin(x)[2:].zfill(6)] for x in bytes_in],len(bytes_in)*6)
    pbytes = pbytes[:8*round(len(pbytes)/8)]
    bits = np.split(pbytes,len(pbytes)/8)
    return np.array([''.join( y for y in x[:]) for x in bits])

################################## SYMBOL MAPPING / DE-MAPPING ##########################################

qam64_map = dict(zip(range(64),[2*(x+y*1j)-7-7j for x,y in np.ndindex((8,8))]))
symbol_mapper = np.vectorize(lambda t: qam64_map[t])

qam64_demap = dict(zip([2*(x+y*1j)-7-7j for x,y in np.ndindex((8,8))],range(64)))
symbol_demapper = np.vectorize(lambda t: qam64_demap[t])

def qam64_th(x):
    return qam64_map[np.argmin(np.abs(np.tile(x,64)-symbol_mapper(range(64))))]
qam_sample = np.vectorize(lambda t: qam64_th(t))

################################## FILTERS & TAPS #######################################################

def butter_lowpass_taps(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass_taps(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def root_raised_cosine_taps(num_taps,beta,Ts):
    t = np.arange(-num_taps//2, num_taps//2) # remember it's not inclusive of final number
    h = (1/np.sqrt(Ts))*(np.sin(np.pi*t/Ts*(1-beta))+4*beta*t/Ts*np.cos(np.pi*t/Ts*(1+beta)))/(np.pi*t/Ts*(1-(4*beta*t/Ts)**2))
    h[t==0] = (1/np.sqrt(Ts))*(1-beta+4*(beta/np.pi))
    return h

def root_raised_cosine(x,taps):
    return np.convolve(x,taps)

################################## MODULATOR #######################################################

debug_sym = []
def qam_mod(bits, taps, xsps=2):
    sym = symbol_mapper(bits)
    debug_sym.append(sym)
    sym = np.insert(sym, np.repeat(range(len(sym)),sps), 0)
    rrc_sym = np.sqrt(xsps)*root_raised_cosine(sym,taps)
    return rrc_sym

################################## NOISE GENERATORS #######################################################
        
def awgn_noise(size, pwr_db=-3):
    pwr = 10**(pwr_db/10)
    return pwr*(np.random.randn(size) + 1j*np.random.randn(size))/np.sqrt(2)

def phase_noise(size,strength = 0.1):
    return np.exp(1j*np.random.randn(size) * strength )

################################## MAIN PROGRAM #######################################################

num_symbols = 10
sps = 64

chars = 'CQ TEST QAM64 PU4THZ PYTHON MOD-DEMOD 73\n'

#### CONVERT STRING TO BITS AND PAD TO LENGTH... ###

bits = np.tile(np.reshape([CharToBits(x) for x in chars],8*len(chars)),10)
padsz = int(6*np.ceil(len(bits)/6))-len(bits)

bits = np.split(np.pad(bits, (0, padsz), 'constant'),(len(bits)+padsz)//6) 
bits = np.array([int(''.join( y for y in x[:]),2) for x in bits])

#### SIMPLE SCRAMBLER - REMOVE LONG SEQUENCES OF 1's OR 0's Eg:  [0, 0 ,0 ... 1, 1 ,1 ,1]  ###

num_symbols = len(bits)

scrambler = np.random.randint(0,64,num_symbols)

sbits = np.bitwise_xor(bits,scrambler)

#### ROOT RAISED COSINE FILTER TAPS ###

rrc_taps = root_raised_cosine_taps(11*sps, 0.35, sps)

#### MODULATION ###

sym = qam_mod(sbits,rrc_taps,sps)
i,q = sym.real, sym.imag

#### PLOT TX CONSTELLATION ###

plt.figure(figsize=(6,6))
plt.title('TX QAM 64 Constellation Map')
plt.plot(i,q,'.',markersize=1)
plt.xticks(np.arange(-10,10,2))
plt.yticks(np.arange(-10,10,2)) 
plt.grid(True)

#### ADD AWGN + PHASE NOISE - CHANNEL MODEL ###

sym += awgn_noise(len(sym),3)
sym *= phase_noise(len(sym),0.1)

#### RECIEVER SIDE -> APPLY RRC AND MANUALLY SYNC THE SAMPLE CLOCK ###

sym2 = root_raised_cosine(sym, rrc_taps)/8
i,q = sym2.real[11*sps:][:num_symbols*(sps+1)], sym2.imag[11*sps:][:num_symbols*(sps+1)]

sym2 = i+q*1j
sym3 = sym2.copy()

#### RX SYMBOLS PLOT ###

plt.figure(2)
plt.title("RX Clock Sync")
plt.plot(sym2.real[:1000],'r')

#### INITIALIZE DOWNSAMPLED SYMBOLS ARRAYS ###

pi, pq = [], []

#### EYE DIAGRAM PLOT ###

fig,ax = plt.subplots(2,1)
fig.suptitle("Eye Diagram")
ax[0].title.set_text("In Phase")
ax[1].title.set_text("Quadrature")

#### DOWNSAMPLE RX SYMBOLS WITH CORRECT TIMMING ###

for i in range(64,len(i),65):
    pi.append(sym2.real[i])
    pq.append(sym2.imag[i])
    ax[0].plot(sym3.real[i-64:i+64],color='red', alpha=.5, lw=.5)
    ax[1].plot(sym3.real[i-64:i+64],color='blue', alpha=.5, lw=.5)

pi,pq = np.array(pi),np.array(pq)

#### SOFT DECISION / GRAY CODING ###

sym2 = qam_sample(pi+pq*1j)

bi, bq = sym2.real, sym2.imag

di,dq = np.array(debug_sym).real[0,:], np.array(debug_sym).imag[0,:]

#### SYMBOL DE-MAPPING INTO BITS ###

out_bits = symbol_demapper(sym2)

#### DE-SCRAMBLING ###

out_bits = np.bitwise_xor(out_bits,scrambler)

#### PRINT TX & RX STRINGS AND ERRORS ###

print('input: ',''.join(x for x in BitsToChars(qam_bytes(bits))))
print('output:',''.join(x for x in BitsToChars(qam_bytes(out_bits))))
print('error',np.sum(out_bits-bits))

#### PLOT RX CONSTELLATION ###

plt.figure(figsize=(6,6))
plt.title('RX QAM 64 Constellation Map')
plt.plot(pi,pq,'o')
plt.plot(bi,bq,'r.')
plt.xlim(-9,9)
plt.ylim(-9,9)
plt.grid(True)

#### PLOT RX SYMBOL SYNC ###

plt.figure(2)
plt.plot(np.insert(pi, np.repeat(range(len(pi)),sps), 0)[:1000],'k.-')

#### PLOT CHANNEL CONSTELLATION ###

plt.figure(figsize=(6,6))
plt.title('Channel Constellation Map')
plt.plot(sym.real,sym.imag,'.',markersize=1)
plt.xlim(-9,9)
plt.ylim(-9,9)
plt.grid(True)

#### END ###

plt.show()
