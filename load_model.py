
import pickle

class loadmodel :
  def load_weights():
    with open('weights.dat','rb') as f:
        weights = pickle.load(f)
        net1,net2=weights[0],weights[1]
    return net1,net2